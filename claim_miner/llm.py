"""
Copyright Society Library and Conversence 2022-2024
"""

from typing import List, Dict, Optional, Union, Literal
import re
from enum import Enum, StrEnum

from openai import OpenAI, AsyncOpenAI

from . import config


class processing_models(Enum):
    o1_preview_2024_09_12 = "o1-preview-2024-09-12"
    o1_preview = "o1-preview"  # Currently o1-preview-2024-09-12
    o1_mini_2024_09_12 = "o1-mini-2024-09-12"
    o1_mini = "o1-mini"  # Currently "o1-mini-2024-09-12"
    gpt_4o = "gpt-4o"  # Currently gpt_4o_2024_05_13
    gpt_4o_latest = "gpt-4o-latest"  # Currently gpt_4o_2024_05_13
    gpt_4o_2024_08_06 = "gpt-4o-2024-08-06"
    gpt_4o_2024_05_13 = "gpt-4o-2024-05-13"
    gpt_4o_mini = "gpt-4o-mini"
    gpt_4o_mini_2024_07_18 = "gpt-4o-mini-2024-07-18"
    gpt_4_turbo = "gpt-4-turbo"  # Currently gpt-4-turbo-2024-04-09
    gpt_4_turbo_2024_04_09 = "gpt-4-turbo-2024-04-09"
    gpt_4_turbo_preview = "gpt-4-turbo-preview"  # Currently gpt_4_0125_preview
    gpt_4 = "gpt-4"  # Currently gpt_4_0613
    gpt_4_0613 = "gpt-4-0613"
    gpt_4_0125_preview = "gpt-4-0125-preview"
    gpt_4_1106_preview = "gpt-4-1106-preview"
    gpt_3_5_turbo = "gpt-3.5-turbo"  # Currently gpt_3_5_turbo_0125
    gpt_3_5_turbo_0125 = "gpt-3.5-turbo-0125"
    gpt_3_5_turbo_1106 = "gpt-3.5-turbo-1106"
    gpt_3_5_turbo_instruct = "gpt-3.5-turbo-instruct"
    gpt_3_5_turbo_instruct_0914 = "gpt-3.5-turbo-instruct-0914"
    text_moderation_latest = "text-moderation-latest"  # Currently text_moderation_007
    text_moderation_stable = "text-moderation-stable"  # Currently text_moderation_007
    text_moderation_007 = "text-moderation-007"
    davinci_002 = "davinci-002"
    babbage_002 = "babbage-002"
    # Legacy
    gpt_4_turbo_0314 = "gpt-4-turbo-0314"


processing_models_legacy = {
    "text-davinci-002": processing_models.davinci_002,
    "text-davinci-003": processing_models.davinci_002,
    "text-babbage-002": processing_models.babbage_002,
    "text-curie-001": processing_models.davinci_002,
    "text-ada-001": processing_models.babbage_002,
    "gpt-3.5-turbo-16k-0613": processing_models.gpt_3_5_turbo,
    "gpt-3.5-turbo-0613": processing_models.gpt_3_5_turbo,
    "gpt-3.5-turbo-0311": processing_models.gpt_3_5_turbo,
    "gpt-3.5-turbo-16k": processing_models.gpt_3_5_turbo,
    "gpt-4-32k": processing_models.gpt_4,
    "gpt-4-32k-0613": processing_models.gpt_4,
}
# Those are legacy names, not legacy models.


def processing_model_from_name(name: str) -> processing_models:
    if value := processing_models_legacy.get(name):
        return value
    if value := processing_models._value2member_map_.get(name):
        return value
    raise ValueError(f"Unknown processing model: {name}")


DEFAULT_MODEL = processing_models.gpt_4o_mini_2024_07_18

SCHEMA_CAPABLE_MODELS = {
    processing_models.gpt_4o_mini_2024_07_18,
    processing_models.gpt_4o_2024_08_06,
}

OPEN_AI_CLIENT = None


def get_openai_client():
    global OPEN_AI_CLIENT
    if not OPEN_AI_CLIENT:
        OPEN_AI_CLIENT = AsyncOpenAI(
            api_key=config.get("openai", "api_key"),
            organization=config.get("openai", "organization"),
        )
    return OPEN_AI_CLIENT
    # if isinstance(model_name, processing_models):
    #     model_name = model_name.value
    # return ChatOpenAI(model_name=model_name, n=2, temperature=temperature)


class SinglePhraseParser:
    """Class to parse the output into a simple dictionary with text."""

    @property
    def _type(self) -> str:
        """Return the type key."""
        return "single_phrase"

    def parse(self, text: str) -> List[Dict[str, str]]:
        """Parse the output of an LLM call."""
        return [dict(text=text.strip())]


class BulletListParser:
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


class BulletListWithRefsParser:
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
