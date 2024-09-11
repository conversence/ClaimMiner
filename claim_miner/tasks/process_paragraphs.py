from typing import List, Union, Tuple, Dict, Optional
from ..utils import find_relative_position
from ..models import Document, Fragment, fragment_type


async def store_processed_para_data(
    session,
    doc: Document,
    paras: List[Tuple[int, int, str, List[Tuple[int, int, str, bool]]]] = [],
):
    fragments = []
    for pos, char_pos, text, sentences in paras:
        para_db = Fragment(
            text=text,
            position=pos,
            char_position=char_pos,
            scale=fragment_type.paragraph,
            document=doc,
            language=doc.language,
        )
        session.add(para_db)
        fragments.append(para_db)
        if len(sentences) > 1:
            for pos, char_pos, text, hallucination in sentences:
                sentence_db = Fragment(
                    text=text,
                    position=pos,
                    char_position=char_pos,
                    scale=fragment_type.sentence,
                    document=doc,
                    language=doc.language,
                    part_of_fragment=para_db,
                )
                if hallucination:
                    sentence_db.generation_data = dict(hallucination=True)
                session.add(sentence_db)
                fragments.append(sentence_db)
        elif sentences:
            if sentences[0][3]:
                para_db.generation_data = dict(hallucination=True)
        else:
            para_db.generation_data = dict(hallucination=True)

    return fragments
