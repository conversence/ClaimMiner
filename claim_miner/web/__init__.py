"""
In this package, the web application's routes are implemented.
"""
# Copyright Society Library and Conversence 2022-2024

from typing import Optional, Union, Set
from io import IOBase

from orjson import loads
from starlette.convertors import Convertor, register_url_convertor
from fastapi import Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import StreamingResponse, Response, FileResponse
from fastapi.staticfiles import StaticFiles

from ..app import app, app_router, spa_router, NotFound
from ..pyd_models import (
    fragment_type_names,
    link_type_names,
    visible_statement_type_names,
    UserModel,
    permission,
    process_status,
    topic_type,
)
from ..models import CollectionScope, model_names_s
from .. import production, Session


def register_url_enum_convertor(enum_class):
    class EnumPathConvertor(Convertor):
        regex = "|".join([e.name for e in enum_class])

        def convert(self, value: str) -> enum_class:
            return enum_class[value]

        def to_string(self, value: enum_class) -> str:
            return value.name

    register_url_convertor(enum_class.__name__, EnumPathConvertor())


register_url_enum_convertor(topic_type)

templates = Jinja2Templates("templates", auto_reload=not production)
templates.env.globals |= dict(
    visible_statement_type_names=visible_statement_type_names,
    fragment_type_names=fragment_type_names,
    link_type_names=link_type_names,
)


def overlaps(f1, f2):
    return max(f1.char_position, f2.char_position) < min(
        f1.char_position + len(f1.text), f2.char_position + len(f2.text)
    )


def render_with_spans(text, fragments):
    if not fragments:
        return text
    use_fragments = fragments[:]
    use_fragments.sort(key=lambda x: (x[1].char_position, len(x[1].text)))
    # eliminate overlaps
    previous = None
    fragments = []
    for fragment in use_fragments:
        if previous and overlaps(previous[1], fragment[1]):
            continue
        fragments.append(fragment)
        previous = fragment
    position = 0
    render = ""
    for _, fragment in fragments:
        fstart = fragment.char_position
        # fudge factor
        if len(fragment.text) > 5 and fragment.text[:5] != text[fstart : fstart + 5]:
            for correction in range(-2, 3):
                if (
                    fragment.text[:5]
                    == text[fstart + correction : fstart + 5 + correction]
                ):
                    fstart = fstart + correction
                    break
        if fstart > position:
            render += text[position:fstart]
            position = fstart
        render += f'<span class="boundary" id="fragment_{fragment.id}">'
        render += fragment.text
        render += "</span>"
        position += len(fragment.text)
        if (
            len(fragment.text) > 5
            and fragment.text[-5:] != text[position - 5 : position]
        ):
            for correction in range(-2, 3):
                if (
                    fragment.text[-5:]
                    == text[position - 5 + correction : position + correction]
                ):
                    position += correction
                    break
    render += text[position:]
    return f"<span>{render}</span>"


def update_fragment_selection(
    request: Request, selection_changes: Optional[str] = None, reset_fragments=False
) -> Set[int]:
    if reset_fragments:
        selection = set()
    else:
        selection = set(request.session.get("fragments_selection", ()))
    if selection_changes:
        selection_changes = loads(selection_changes)
        for k, v in (selection_changes or {}).items():
            k = int(k)
            if v:
                selection.add(k)
            else:
                selection.discard(k)
    if selection_changes or reset_fragments:
        request.session["fragments_selection"] = list(selection)
    return selection


collection_path = CollectionScope.collection_path
get_collection = CollectionScope.get_collection


async def get_base_template_vars(
    request: Request,
    current_user: Optional[UserModel] = None,
    collection=None,
    session=None,
):
    if session is None:
        async with Session() as session:
            return await get_base_template_vars(
                request, current_user, collection, session
            )

    try:
        collection_ob = await get_collection(
            collection, session, current_user.id if current_user else None
        )
    except ValueError:
        raise NotFound()
    collection_names = await CollectionScope.get_collection_names(session)

    def user_can(perm: Union[str, permission]):
        return collection_ob.user_can(current_user, perm)

    return dict(
        collection=collection_ob,
        collection_names=collection_names,
        model_names=model_names_s,
        process_status=process_status,
        user_can=user_can,
    )


@spa_router.get("/favicon.ico")
def favicon():
    return FileResponse("static/favicon.ico")


app.mount("/static", StaticFiles(directory="static"), name="static")


def send_as_attachment(content, media_type, filename):
    # TODO: length, etag...
    if isinstance(content, IOBase):
        return StreamingResponse(
            content,
            media_type=media_type,
            headers={"content-disposition": f"attachment;filename={filename}"},
        )
    else:
        return Response(
            content,
            media_type=media_type,
            headers={"content-disposition": f"attachment;filename={filename}"},
        )
