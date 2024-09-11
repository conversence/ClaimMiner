"""
Copyright Society Library and Conversence 2022-2024
"""

from typing import List, Dict, Optional, Union, Literal
import re
from enum import Enum, StrEnum

from redis import Redis
import langchain
from langchain_community.cache import RedisCache
from langchain.schema import BaseOutputParser
from langchain.globals import set_llm_cache


set_llm_cache(RedisCache(redis_=Redis(db=6)))


class processing_models(Enum):
    gpt_4o = "gpt-4o"  # Currently gpt_4o_2024_05_13
    gpt_4o_2024_05_13 = "gpt-4o-2024-05-13"
    gpt_4_turbo = "gpt-4-turbo"  # Currently gpt_4_turbo_2024_04_09
    gpt_4_turbo_2024_04_09 = "gpt-4-turbo-2024-04-09"
    gpt_4 = "gpt-4"  # Currently gpt_4_0613
    gpt_4_0613 = "gpt-4-0613"
    gpt_4_turbo_preview = "gpt-4-turbo-preview"  # Currently gpt_4_0125_preview
    gpt_4_0125_preview = "gpt-4-0125-preview"
    gpt_4_1106_preview = "gpt-4-1106-preview"
    gpt_4_32K = "gpt-4-32k"  # Currently gpt-4-32k-0613
    gpt_4_32K_0613 = "gpt-4-32k-0613"
    gpt_3_5_turbo = "gpt-3.5-turbo"  # Currently gpt_3_5_turbo_0125
    gpt_3_5_turbo_0125 = "gpt-3.5-turbo-0125"
    gpt_3_5_turbo_1106 = "gpt-3.5-turbo-1106"
    text_moderation_latest = "text-moderation-latest"  # Currently text_moderation_007
    text_moderation_stable = "text-moderation-stable"  # Currently text_moderation_007
    text_moderation_007 = "text-moderation-007"
    davinci_002 = "davinci-002"
    babbage_002 = "babbage-002"
    # Legacy
    gpt_3_5_turbo_16K = "gpt-3.5-turbo-16k"  # Currently gpt_3_5_turbo_16K_0613
    gpt_3_5_turbo_16K_0613 = "gpt-3.5-turbo-16k-0613"
    gpt_3_5_turbo_0613 = "gpt-3.5-turbo-0613"
    # Obsolete
    gpt_3_5_turbo_0301 = "gpt-3.5-turbo-0301"


processing_models_legacy = {
    "text-davinci-002": processing_models.davinci_002,
    "text-davinci-003": processing_models.davinci_002,
    "text-babbage-002": processing_models.babbage_002,
    "text-curie-001": processing_models.davinci_002,
    "text-ada-001": processing_models.babbage_002,
}
# Those are legacy names, not legacy models.


def processing_model_from_name(name: str) -> processing_models:
    if value := processing_models_legacy.get(name):
        return value
    if value := processing_models._value2member_map_.get(name):
        return value
    raise ValueError(f"Unknown processing model: {name}")


DEFAULT_MODEL = processing_models.gpt_3_5_turbo_0125


def get_base_llm(
    model_name: Union[processing_models, str] = DEFAULT_MODEL, temperature=0
):
    from langchain_openai import ChatOpenAI

    if isinstance(model_name, processing_models):
        model_name = model_name.value
    return ChatOpenAI(model_name=model_name, n=2, temperature=temperature)


class SinglePhraseParser(BaseOutputParser):
    """Class to parse the output into a simple dictionary with text."""

    @property
    def _type(self) -> str:
        """Return the type key."""
        return "single_phrase"

    def parse(self, text: str) -> List[Dict[str, str]]:
        """Parse the output of an LLM call."""
        return [dict(text=text.strip())]


class BulletListParser(BaseOutputParser):
    """Class to parse the output into a list of dictionaries."""

    regex_pattern = re.compile(r"^\s*[-\+\*•]+\s+(.*)\s*$")

    @property
    def _type(self) -> str:
        """Return the type key."""
        return "bullet_list"

    def parse(self, text: str) -> List[Dict[str, str]]:
        """Parse the output of an LLM call."""
        lines = re.split(r"[\r\n]+", text)
        matches = [re.match(self.regex_pattern, line) for line in lines]
        if matches := list(filter(None, matches)):
            return [dict(text=r.group(1)) for r in matches]
        else:
            raise ValueError("No answer")


class BulletListWithRefsParser(BaseOutputParser):
    """Class to parse the output into a list of dictionaries."""

    regex_pattern = re.compile(r"^\s*[-\+\*•]+\s+(.*)\s+\((\d+(,\s*\d+)*)\)\s*\.?\s*$")

    @property
    def _type(self) -> str:
        """Return the type key."""
        return "bullet_list_with_refs"

    def parse(self, text: str) -> List[Dict]:
        """Parse the output of an LLM call."""
        lines = re.split(r"[\r\n]+", text)
        matches = [re.match(self.regex_pattern, line) for line in lines]
        if matches := list(filter(None, matches)):
            return [
                dict(text=r.group(1), sources=[int(x) for x in r.group(2).split(",")])
                for r in matches
            ]
        else:
            raise ValueError("No answer")


parsers = [SinglePhraseParser(), BulletListParser(), BulletListWithRefsParser()]

parsers_by_name = {p._type: p for p in parsers}

parser_type = StrEnum("parser_type", [p._type for p in parsers])
