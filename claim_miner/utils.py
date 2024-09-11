from uuid import UUID
from base64 import urlsafe_b64decode, urlsafe_b64encode
import asyncio
from functools import partial, wraps
from contextvars import copy_context
import inspect
from copy import deepcopy
from typing import (
    Callable,
    Coroutine,
    Any,
    Generator,
    AsyncGenerator,
    Type,
    TypeVar,
    Tuple,
    Optional,
    Iterable,
    List,
)
import re

from pydantic import BaseModel, create_model
from pydantic.fields import FieldInfo
from Levenshtein import distance
from langdetect import detect, LangDetectException


def encode_uuid(u: UUID) -> str:
    return urlsafe_b64encode(u.bytes).decode("ascii").rstrip("=")


def decode_uuid(id: str) -> UUID:
    return UUID(bytes=urlsafe_b64decode(id + "=" * (24 - len(id))))


def as_list(val):
    if isinstance(val, (tuple, list)):
        return val
    return [val]


def as_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"true", "yes", "on", "1", "checked"}


def filter_dict(d):
    return dict(filter(lambda x: x[1] is not None, d.items()))


# Copied from Quart, so as to not load quart in kafka worker


def run_sync(func: Callable[..., Any]) -> Callable[..., Coroutine[None, None, Any]]:
    """Ensure that the sync function is run within the event loop.
    If the *func* is not a coroutine it will be wrapped such that
    it runs in the default executor (use loop.set_default_executor
    to change). This ensures that synchronous functions do not
    block the event loop.
    """

    @wraps(func)
    async def _wrapper(*args: Any, **kwargs: Any) -> Any:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, copy_context().run, partial(func, *args, **kwargs)
        )
        if inspect.isgenerator(result):
            return run_sync_iterable(result)  # type: ignore
        else:
            return result

    return _wrapper


def run_sync_iterable(
    iterable: Generator[Any, None, None],
) -> AsyncGenerator[Any, None]:
    async def _gen_wrapper() -> AsyncGenerator[Any, None]:
        # Wrap the generator such that each iteration runs
        # in the executor. Then rationalise the raised
        # errors so that it ends.
        def _inner() -> Any:
            # https://bugs.python.org/issue26221
            # StopIteration errors are swallowed by the
            # run_in_exector method
            try:
                return next(iterable)
            except StopIteration:
                raise StopAsyncIteration()

        loop = asyncio.get_running_loop()
        while True:
            try:
                yield await loop.run_in_executor(None, copy_context().run, _inner)
            except StopAsyncIteration:
                return

    return _gen_wrapper()


class classproperty(object):
    def __init__(self, f):
        self.f = f

    def __get__(self, obj, owner):
        return self.f(owner)


BaseModelT = TypeVar("BaseModelT", bound=BaseModel)


def make_field_optional(field: FieldInfo, default: Any = None) -> Tuple[Any, FieldInfo]:
    new = deepcopy(field)
    new.default = default
    new.annotation = Optional[field.annotation]  # type: ignore
    return (new.annotation, new)


def to_optional(model: Type[BaseModelT]) -> Type[BaseModelT]:
    """Transform a schema into an equivalent optional schema"""
    # https://github.com/pydantic/pydantic/issues/3120#issuecomment-1528030416
    return create_model(  # type: ignore
        f"Partial{model.__name__}",
        __base__=model,
        __module__=model.__module__,
        **{
            field_name: make_field_optional(field_info)
            for field_name, field_info in model.model_fields.items()
        },
    )


def deduplicate(l: Iterable, keyf: Optional[Callable] = None) -> Generator:
    # Deduplicate a list while keeping the order
    keyf = keyf or (lambda x: x)
    vals = set()
    for e in l:
        k = keyf(e)
        if k not in vals:
            yield e
            vals.add(k)


def find_relative_position(
    text: str, subtext: str, first_pos: int = 0, fuzzy_tolerance=0.33
) -> Tuple[Optional[int], int, Optional[int]]:
    """
    This function finds the relative position of subtext in text, starting at the given start_index.
    Discounts deletions, and allows to go further.
    """
    lsub = len(subtext)
    ltxt = len(text) - first_pos
    pos = text.find(subtext, first_pos)
    cutoff = max(1, int(lsub * fuzzy_tolerance))
    if pos >= 0:
        return (pos, pos + lsub, None)
    step = cutoff
    best_score = cutoff + 1
    best_pos = None
    start = first_pos
    end = start + step * ((ltxt - lsub) // step) + 1
    while True:
        for i in range(start, end, step):
            score = distance(text[i : i + lsub], subtext, score_cutoff=best_score)
            if score < best_score:
                best_score = score
                best_pos = i
        if step == 1:
            break
        step = max(step // 3, 1)
        if best_pos is not None:
            start = max(start, best_pos - 2 * step)
            end = min(end, best_pos + 2 * step + 1)
    if best_score >= cutoff:
        return (None, None, cutoff)
    # It is possible that the subtext skips text.
    step = cutoff
    best_end_pos = best_pos + lsub
    best_score2 = distance(text[best_pos : best_pos + lsub], subtext, weights=(8, 1, 8))
    start = best_pos + step
    end = first_pos + ltxt
    while True:
        for i in range(start, end, step):
            score = distance(
                text[best_pos:i], subtext, weights=(8, 1, 8), score_cutoff=best_score2
            )
            if score < best_score2:
                best_score2 = score
                best_end_pos = i
        if step == 1:
            break
        step = max(step // 3, 1)
        start = max(start, best_end_pos - 2 * step)
        end = min(end, best_end_pos + 2 * step + 1)
    if best_score >= cutoff:
        return (None, None, cutoff)

    return (best_pos, best_end_pos, best_score)


_sentry_sdk = None


def get_sentry_sdk(process):
    from . import db_config_get, target_db

    global _sentry_sdk
    if _sentry_sdk is None and (sentry_dsn := db_config_get("sentry_dsn", None)):
        import sentry_sdk

        _sentry_sdk = sentry_sdk
        sentry_sdk.init(
            dsn=sentry_dsn,
            enable_tracing=True,
            traces_sample_rate=0.1,
            environment=f"{target_db}_{process}",
        )
    return _sentry_sdk


def safe_lang_detect(text):
    try:
        return detect(text)
    except LangDetectException:
        return "und"
