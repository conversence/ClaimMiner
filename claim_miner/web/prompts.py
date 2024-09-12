"""
Copyright Society Library and Conversence 2022-2024
"""

from logging import getLogger
from typing import Annotated, Optional

from fastapi import Request, Form, status
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import joinedload, subqueryload
from sqlalchemy.orm.attributes import flag_modified

from .. import Session, select
from . import update_fragment_selection, get_base_template_vars, app_router
from ..models import (
    Statement,
    StatementOrFragment,
    ClaimLink,
    TopicCollection,
    Analysis,
    Collection,
)
from ..pyd_models import fragment_type, link_type, process_status
from ..app import BadRequest, NotFound, Unauthorized
from ..auth import user_with_permission_c_dep
from ..llm import get_openai_client, parsers_by_name, processing_models
from ..task_registry import TaskRegistry

logger = getLogger(__name__)

prompt_analyzer_names = ("fragment_prompt_analyzer", "simple_prompt_analyzer")


@app_router.post("/claim/{statement_id}/simple_prompt")
@app_router.post("/c/{collection}/claim/{statement_id}/simple_prompt")
@app_router.post("/claim/{statement_id}/prompt_fragments")
@app_router.post("/c/{collection}/claim/{statement_id}/prompt_fragments")
async def analyze_prompt(
    request: Request,
    current_user: user_with_permission_c_dep("openai_query"),
    statement_id: int,
    template_nickname: Annotated[str, Form()],
    model: Annotated[processing_models, Form()],
    selection_changes: Annotated[Optional[str], Form()] = None,
    reset_fragments: Annotated[bool, Form()] = False,
    collection: Optional[str] = None,
):
    sources = []
    use_fragments = request.url.path.endswith("_fragments")

    # TODO: Use the task!
    # TODO: Does an analysis with those params already exist?
    # Decide: is model part of the analyzer or analysis? I say former.
    if use_fragments:
        sources = list(
            update_fragment_selection(request, selection_changes, reset_fragments)
        )
        if not sources:
            raise BadRequest("No sources")
    analyzer_name = (
        "fragment_prompt_analyzer" if use_fragments else "simple_prompt_analyzer"
    )
    task_registry = TaskRegistry.get_registry()
    analyzer = task_registry.analyzer_by_name[analyzer_name]
    prompt_template = task_registry.task_template_by_nickname.get(template_nickname)
    if not prompt_template:
        return NotFound(f"No prompt called {template_nickname}")
    async with Session() as session:
        base_vars = await get_base_template_vars(
            request, current_user, collection, session
        )
        collection_ob: Collection = base_vars["collection"]
        r = await session.execute(
            select(StatementOrFragment).filter(
                StatementOrFragment.id.in_(sources + [statement_id])
            )
        )
        fragments = {f.id: f for (f,) in r}
        statement = fragments.pop(statement_id)

    if use_fragments:
        # partial_variables=dict(format_instructions=parser.get_format_instructions())
        fragment_texts = "\n\n".join(
            f"({id}): {f.text})" for (id, f) in fragments.items()
        )
        prompt = prompt_template.prompt.format(theme=statement.text, fragments=fragment_texts)
    else:
        prompt = prompt_template.prompt.format(theme=statement.text)
    logger.debug("%s", prompt)
    client = get_openai_client()  # temperature...
    messages = [dict(role="user", prompt=prompt)]
    if prompt_template.system_prompt:
        messages = [dict(role="system", prompt=prompt_template.system_prompt)] + messages

    resp = await client.beta.chat.completions.parse(
        model=prompt_template.model.value,
        messages = messages)
    result = resp.content
    logger.debug("%s", result)
    parser = parsers_by_name[prompt_template.params["parser"]]
    result = parser.parse(result)
    logger.debug("%s", result)
    async with Session() as session:
        params = dict(smodel=model)
        if use_fragments:
            theme = statement
            target = None
            params["source_ids"] = sorted(fragments.keys())
        else:
            theme = None
            target = statement
        analysis = Analysis(
            analyzer_id=analyzer.id,
            task_template_id=prompt_template.id,
            theme=theme,
            target=target,
            results=result,
            params=params,
            creator_id=current_user.id,
            context=list(fragments.values()),
        )
        session.add(analysis)
        await session.commit()

    return RedirectResponse(f"/f{collection_ob.path}/analysis/{analysis.id}")


@app_router.post("/analysis/{analysis_id}/prompt")
@app_router.post("/c/{collection}/analysis/{analysis_id}/prompt")
async def process_prompt_analysis_post(
    request: Request,
    current_user: user_with_permission_c_dep("openai_query"),
    analysis_id: int,
    saving: Annotated[Optional[int], Form()] = None,
    exporting: Annotated[bool, Form()] = False,
    delete_results: Annotated[bool, Form()] = False,
    collection: Optional[str] = None,
):
    changed = False
    async with Session() as session, request.form() as form:
        base_vars = await get_base_template_vars(
            request, current_user, collection, session
        )
        collection_ob: Collection = base_vars["collection"]

        analysis = await session.scalar(
            select(Analysis)
            .filter_by(id=analysis_id)
            .options(
                joinedload(Analysis.analyzer),
                joinedload(Analysis.task_template),
                joinedload(Analysis.theme),
                joinedload(Analysis.target),
                subqueryload(Analysis.context),
                subqueryload(Analysis.generated_topics),
            )
        )
        analyzer = analysis.analyzer
        prompt_template = analysis.task_template
        fragments = {f.id: f for f in analysis.context}
        result_nodes = {f.id: f for f in analysis.generated_topics}

        assert analyzer.name in ("fragment_prompt_analyzer", "simple_prompt_analyzer")

        target = analysis.target or analysis.theme

        for i, r in enumerate(analysis.results):
            new_text = form.get(f"text_{i+1}")
            if new_text and new_text.strip() != r["text"].strip():
                if "old_text" not in r:
                    r["old_text"] = r["text"]
                elif new_text == r["old_text"]:
                    del r["old_text"]
                r["text"] = new_text
                changed = True

        if saving:
            new_statement_data = analysis.results[saving - 1]
            if new_statement_data.get("fragment_id", None):
                raise BadRequest("Already saved")
            new_node_type = getattr(
                fragment_type, prompt_template.params["node_type"], None
            )
            new_link_type = getattr(
                link_type, prompt_template.params["link_type"], None
            )
            generation_data = {}
            if fragments:
                # We had a stack trace here, with sources undefined. How?
                sources = new_statement_data.get("sources", [])
                if sources:
                    generation_data["sources"] = sources
                else:
                    logger.warning("Missing sources! %s", new_statement_data)
            # TODO: Check if a fragment with that text exists, reuse fragment and add analysis instead
            statement = Statement(
                text=new_statement_data["text"],
                scale=new_node_type,
                language="en",
                from_analyses=[analysis],
                generation_data=generation_data,
                created_by=current_user.id,
            )
            if fragments:
                slinks = [
                    ClaimLink(
                        target_topic=statement,
                        source_topic=fragments[source],
                        link_type=link_type.quote,
                        created_by=current_user.id,
                        from_analyses=[analysis],
                    )
                    for source in sources
                ]
                session.add_all(slinks)
            if prompt_template.params.get("backwards_link", False):
                flink = ClaimLink(
                    source_topic=statement,
                    target_topic=target,
                    link_type=new_link_type,
                    created_by=current_user.id,
                    from_analyses=[analysis],
                )
            else:
                flink = ClaimLink(
                    source_topic=target,
                    target_topic=statement,
                    link_type=new_link_type,
                    created_by=current_user.id,
                    from_analyses=[analysis],
                )
            session.add(flink)
            await session.flush()
            if collection_ob:
                session.add(
                    TopicCollection(
                        collection_id=collection_ob.id, topic_id=statement.id
                    )
                )
            new_statement_data["fragment_id"] = statement.id
            analysis.results[int(saving) - 1] = new_statement_data
            await session.commit()
            changed = True

        elif delete_results:
            if not collection_ob.user_can(current_user, "admin"):
                raise Unauthorized()
            registry = TaskRegistry.get_registry()
            task = registry.task_from_analysis(analysis)
            await registry.delete_task_data(session, task, True)
            analysis.status = process_status.not_requested
            await session.commit()

        if changed:
            flag_modified(analysis, "results")
        await session.commit()
    return RedirectResponse(
        f"/f{collection_ob.path}/analysis/{analysis.id}",
        status_code=status.HTTP_303_SEE_OTHER,
    )
