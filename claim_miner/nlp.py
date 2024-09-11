"""
Copyright Society Library and Conversence 2022-2024
"""

import re

import spacy

from . import config

nlp = spacy.load(config.get("base", "spacy_model", fallback="en_core_web_sm"))


def breakup_para(para, limit=999):
    if len(para) > limit:
        chunks = []
        offset = 0
        nlp_analysis = nlp(para)
        sents = list(nlp_analysis.sents)
        for s in sents:
            if s.end_char - offset > limit:
                end = s.start_char
                chunks.append(para[offset:end])
                offset = end
        chunks.append(para[offset:])
    else:
        chunks = [para]
    return chunks


def breakup_paras(paras, limit=999):
    chunks = []
    for para in paras:
        if len(para) > limit:
            chunks.extend(breakup_para(para, limit=limit))
        else:
            chunks.append(para)
    return chunks


def breakup_sentences(txt, limit=999):
    # For text that does not have paragraphs
    txt = re.sub(r"[\n\s]+", " ", txt)
    nlp_analysis = nlp(txt)
    chunks = []
    offset = 0
    sents = list(nlp_analysis.sents)
    for s in sents:
        if s.end_char - offset > limit:
            end = s.start_char
            chunks.append(txt[offset:end])
            offset = end
    return chunks


min_wlen = 12

PIVOT_PATTERNS = [
    (("ccomp", "nsubj"), "R"),
    (("ccomp", "csubj"), "R"),
    (("ccomp"), "L"),
    ((), "+"),
]


def check_pivot_pattern(pivot, pattern, command):
    if pattern:
        dep = pattern[0]
        for c in pivot.children:
            if c.dep_ == dep:
                return check_pivot_pattern(c, pattern[1:], command)
        return None
    if command == "R":
        return pivot.right_edge.i + 1
    elif command == "L":
        return pivot.left_edge.i
    elif command == "+":
        return pivot.i + 1
    elif command == "*":
        return pivot.i
    else:
        raise RuntimeError(f"Unknown pattern command: {command}")


def check_pivot_patterns(r, pivot_patterns=PIVOT_PATTERNS):
    for pattern, command in pivot_patterns:
        pos = check_pivot_pattern(r, pattern, command)
        if pos is not None:
            return pattern, command, pos
    assert False


def roots(ana):
    return [t for t in ana if t.dep_ == "ROOT"]


def as_prompts(para):
    ana = nlp(para)
    for r in roots(ana):
        if r.right_edge.i - r.left_edge.i < min_wlen:
            continue
        _, _, pos = check_pivot_patterns(r)
        yield (ana[r.left_edge.i : pos].text, ana[pos : r.right_edge.i + 1].text)


def print_tree(ana, r, indent=0, direction=""):
    before = True
    children = list(r.children)
    child_dir = "v "
    for c in children:
        if before and c.i > r.i:
            before = False
            child_dir = "^ "
            print("  " * indent, direction, r.dep_, ": ", ana[r.i], sep="")
        print_tree(ana, c, indent + 1, child_dir)
    if before:
        print("  " * indent, direction, r.dep_, ": ", ana[r.i], sep="")
