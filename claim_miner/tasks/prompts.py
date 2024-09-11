from typing import Optional
from logging import getLogger

from sqlalchemy.orm import joinedload, subqueryload, contains_eager
from sqlalchemy.orm.attributes import flag_modified
from langchain.prompts import PromptTemplate

from .. import Session
from ..pyd_models import process_status, fragment_type, link_type
from ..models import Analysis, ClaimLink, Statement, Topic, with_polymorphic, select
from ..llm import get_base_llm, parsers_by_name, processing_models
from .kafka import sentry_sdk
from .tasks import (
    PromptAnalysisModel,
    SimplePromptAnalysisModel,
    FragmentPromptAnalysisModel,
    PromptTaskTemplateModel,
)

logger = getLogger(__name__)


async def do_prompt_analysis(analysis_id: int):
    async with Session() as session:
        theme_alias = with_polymorphic(Topic, "*", flat=True, aliased=True)
        target_alias = with_polymorphic(Topic, "*", flat=True, aliased=True)
        analysis: Optional[Analysis] = await session.scalar(
            select(Analysis)
            .filter_by(id=analysis_id)
            .outerjoin(Analysis.theme.of_type(theme_alias))
            .outerjoin(Analysis.target.of_type(target_alias))
            .options(
                joinedload(Analysis.task_template),
                joinedload(Analysis.collection),
                subqueryload(Analysis.context),
                contains_eager(Analysis.target.of_type(target_alias)),
                contains_eager(Analysis.theme.of_type(theme_alias)),
            )
        )
        if not analysis:
            logger.error(f"Analysis {analysis_id} not found")
            return False
        analysis_model: PromptAnalysisModel = analysis.as_model(session)
        task_template: PromptTaskTemplateModel = analysis_model.get_task_template()
        prompt_template = task_template.prompt

        if analysis.analyzer_name == "fragment_prompt_analyzer":
            statement = analysis.theme
            prompt_t = PromptTemplate(
                input_variables=["theme", "fragments"], template=prompt_template
            )
            # partial_variables=dict(format_instructions=parser.get_format_instructions())
            fragment_texts = "\n\n".join(
                f"({f.id}): {f.text})" for f in analysis.context
            )
            prompt = prompt_t.format(theme=statement.text, fragments=fragment_texts)
            target = analysis.theme
        else:
            statement = analysis.target
            prompt_t = PromptTemplate(
                input_variables=["theme"], template=prompt_template
            )
            prompt = prompt_t.format(theme=statement.text)
            target = analysis.target
        collections = [analysis.collection] if analysis.collection_id else []
        logger.debug("%s", prompt)
        llm = get_base_llm(model_name=task_template.model.value)  # temperature...
        try:
            resp = await llm.ainvoke(prompt)
        except Exception as e:
            logger.exception("", exc_info=e)
            if sentry_sdk:
                sentry_sdk.capture_exception(e)
            analysis.status = process_status.error
            await session.commit()
            return False
        result = resp.content
        logger.debug("%s", result)
        parser = parsers_by_name[task_template.parser]
        result = parser.parse(result)
        logger.debug("%s", result)
        analysis.results = result
        analysis.status = process_status.complete
        await session.commit()
        if not analysis_model.autosave:
            return analysis

        for i, new_statement_data in enumerate(analysis.results):
            generation_data = {}
            if analysis.context:
                sources = new_statement_data.get("sources", [])
                if sources:
                    generation_data["sources"] = sources
                else:
                    logger.warning("Missing sources! %s", new_statement_data)
            # TODO: Check if a fragment with that text exists, reuse fragment and add analysis instead
            statement = Statement(
                text=new_statement_data["text"],
                scale=task_template.node_type,
                language="en",
                from_analyses=[analysis],
                generation_data=generation_data,
                collections=collections,
            )
            if analysis.context:
                slinks = [
                    ClaimLink(
                        target_topic=source,
                        source_topic=statement,
                        link_type=link_type.quote,
                        from_analyses=[analysis],
                    )
                    for source in sources
                ]
                session.add_all(slinks)
            if task_template.backwards_link:
                flink = ClaimLink(
                    source_topic=statement,
                    target_topic=target,
                    link_type=task_template.link_type,
                    from_analyses=[analysis],
                )
            else:
                flink = ClaimLink(
                    source_topic=target,
                    target_topic=statement,
                    link_type=task_template.link_type,
                    from_analyses=[analysis],
                )
            session.add(flink)
            await session.flush()
            new_statement_data["fragment_id"] = statement.id
            analysis.results[i] = new_statement_data
        flag_modified(analysis, "results")
        await session.commit()
    return analysis
