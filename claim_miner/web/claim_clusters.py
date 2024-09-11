"""
Copyright Society Library and Conversence 2022-2024
"""

# clustering
from collections import defaultdict
from itertools import groupby
from typing import Annotated, Optional, Union, List

import numpy as np
from sqlalchemy.future import select
from sqlalchemy.sql import func, delete
from sqlalchemy.orm import joinedload, subqueryload
from sklearn.cluster import DBSCAN
from fastapi import Form, Request, status
from fastapi.responses import RedirectResponse
from ..utils import run_sync

from . import get_base_template_vars, templates, app_router
from .. import Session, dispatcher
from ..app import BadRequest
from ..auth import user_with_coll_permission_c_dep, UserModel
from ..pyd_models import fragment_type, embedding_model
from ..models import (
    Collection,
    VisibleClaim,
    Statement,
    Analysis,
    ClusterData,
    InClusterData,
    User,
    aliased,
    embed_db_model_by_name,
    poly_type_clause,
)
from ..tasks.tasks import ClusterTask


async def claim_cluster(
    request: Request,
    current_user: UserModel,
    collection,
    eps: float = 0.1,
    min_samples: int = 5,
    model: Optional[embedding_model] = None,
    scale: Optional[fragment_type] = None,
    algorithm: Optional[str] = "dbscan",
    save: Optional[bool] = False,
    recalc: Optional[bool] = False,
):
    if save:
        # TODO: Should I create the analysis object here?
        await dispatcher.trigger_task(
            "cluster",
            collection_name=collection,
            algorithm=algorithm,
            model=model.name,
            eps=eps,
            min_samples=min_samples,
            scale=scale,
        )
        return RedirectResponse(
            f"/f{collection.path}/analysis/cluster?success=Reload+soon",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    async with Session() as session:
        base_vars = await get_base_template_vars(
            request, current_user, collection, session
        )
        collection_ob: Collection = base_vars["collection"]
        model = model or collection_ob.embed_model()
        if model.name not in embed_db_model_by_name:
            raise BadRequest("Invalid model")
        Embedding = embed_db_model_by_name[model.name]
        q = select(Embedding.fragment_id, Embedding.embedding, VisibleClaim.text).join(
            VisibleClaim, Embedding.fragment
        )
        q = q.filter(poly_type_clause(VisibleClaim))
        if scale:
            q = q.filter(VisibleClaim.scale == scale)
        else:
            q = q.filter(VisibleClaim.scale != fragment_type.standalone_category)
        q = q.join(Collection, VisibleClaim.collections).filter_by(id=collection_ob.id)
        data = await session.execute(q)
    data = list(zip(*data))
    clusters = defaultdict(list)
    if data:
        (fids, embeds, texts) = data
        embeds = np.array(embeds)
        scan = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
        db = await run_sync(scan.fit)(embeds)
        for i, c in enumerate(db.labels_):
            if c == -1:
                continue
            clusters[c].append((fids[i], texts[i]))
        missing = len(db.labels_) - sum(len(c) for c in clusters.values())
    else:
        missing = 0
    return templates.TemplateResponse(
        request,
        "claim_clusters_new.html",
        dict(
            clusters=clusters.values(),
            missing=missing,
            eps=eps,
            models=collection_ob.embed_models_names(),
            min_samples=min_samples,
            model=model,
            algorithm=algorithm,
            **base_vars,
        ),
    )


@app_router.get("/c/{collection}/analysis/cluster/new")
async def claim_cluster_get(
    request: Request,
    current_user: user_with_coll_permission_c_dep("access"),
    collection: str,
    eps: float = 0.1,
    min_samples: int = 5,
    model: Optional[embedding_model] = None,
    scale: Optional[fragment_type] = None,
    algorithm: Optional[str] = "dbscan",
    save: Optional[bool] = None,
    recalc: Optional[bool] = None,
):
    return await claim_cluster(
        request,
        current_user,
        collection,
        eps,
        min_samples,
        model,
        scale,
        algorithm,
        save,
        recalc,
    )


@app_router.post("/c/{collection}/analysis/cluster/new")
async def claim_cluster_post(
    request: Request,
    current_user: user_with_coll_permission_c_dep("access"),
    collection: str,
    model: Annotated[Optional[embedding_model], Form()] = None,
    scale: Annotated[Optional[fragment_type], Form()] = None,
    eps: Annotated[float, Form()] = 0.1,
    min_samples: Annotated[int, Form()] = 5,
    algorithm: Annotated[Optional[str], Form()] = "dbscan",
    save: Annotated[bool, Form()] = False,
    recalc: Annotated[bool, Form()] = False,
):
    form = await request.form()
    return await claim_cluster(
        request,
        current_user,
        collection,
        eps,
        min_samples,
        model,
        scale,
        algorithm,
        "save" in form,
        "recalc" in form,
    )


@app_router.get("/c/{collection}/analysis/cluster")
async def list_claim_clusters(
    request: Request,
    current_user: user_with_coll_permission_c_dep("access"),
    collection: str,
):
    cluster_analyzer_id = await ClusterTask.get_analyzer_id()
    async with Session() as session:
        base_vars = await get_base_template_vars(
            request, current_user, collection, session
        )
        collection_ob: Collection = base_vars["collection"]
        cluster_analyses = await session.execute(
            select(Analysis).filter_by(
                analyzer_id=cluster_analyzer_id, collection_id=collection_ob.id
            )
        )
        cluster_analyses = [a for (a,) in cluster_analyses]
        num_clusters = await session.execute(
            select(
                ClusterData.analysis_id,
                func.count(ClusterData.id.distinct()),
                func.count(VisibleClaim.id),
            )
            .join(VisibleClaim, ClusterData.fragments)
            .filter(poly_type_clause(VisibleClaim))
            .join(
                Analysis,
                ClusterData.analysis.and_(Analysis.collection_id == collection_ob.id),
            )
            .group_by(ClusterData.analysis_id)
        )
        num_clusters = {ana.id: (0, 0) for ana in cluster_analyses} | {
            ana_id: (cl_count, fr_count)
            for (ana_id, cl_count, fr_count) in num_clusters
        }

        total_num_clusters = sum(
            cl_count for (cl_count, fr_count) in num_clusters.values()
        )
        num_attached = sum(fr_count for (cl_count, fr_count) in num_clusters.values())
        total_num_claims = await session.scalar(
            select(func.count(VisibleClaim.id))
            .filter(poly_type_clause(VisibleClaim))
            .join(Collection, VisibleClaim.collections)
            .filter_by(name=collection_ob.name)
        )
        if total_num_claims and not num_clusters and not cluster_analyses:
            return RedirectResponse(f"/f{collection_ob.path}/analysis/cluster/new")
        creators_set = {a.creator_id for a in cluster_analyses}
        creators_set.discard(None)
        creators = dict()
        if creators_set:
            creators_q = await session.execute(
                select(User.id, User.handle).filter(User.id.in_(creators_set))
            )
            creators = dict(list(creators_q))
    return templates.TemplateResponse(
        request,
        "list_claim_clusters.html",
        dict(
            num_clusters=num_clusters,
            total_num_claims=total_num_claims,
            total_num_clusters=total_num_clusters,
            num_outlier_claims=total_num_claims - num_attached,
            cluster_analyses=cluster_analyses,
            creators=creators,
            **base_vars,
        ),
    )


@app_router.get("/c/{collection}/analysis/cluster/all")
async def claim_all_clusters(
    request: Request,
    current_user: user_with_coll_permission_c_dep("access"),
    collection: str,
):
    cluster_analyzer_id = await ClusterTask.get_analyzer_id()
    async with Session() as session:
        base_vars = await get_base_template_vars(
            request, current_user, collection, session
        )
        collection_ob: Collection = base_vars["collection"]
        clusters = await session.execute(
            select(ClusterData)
            .join(Analysis)
            .filter_by(collection_id=collection_ob.id, analyzer_id=cluster_analyzer_id)
            .order_by(ClusterData.analysis_id, ClusterData.cluster_size.desc())
            .options(
                joinedload(ClusterData.distinguished_claim),
                subqueryload(ClusterData.analysis),
                subqueryload(ClusterData.has_cluster_rels).subqueryload(
                    InClusterData.fragment
                ),
            )
        )
        clusters = [cluster for (cluster,) in clusters]
        grouped_clusters = groupby(clusters, lambda c: c.analysis)

        user_ids = {cluster.analysis.creator_id for cluster in clusters}
        for cluster in clusters:
            user_ids.update(icd.confirmed_by_id for icd in cluster.has_cluster_rels)
        user_ids.discard(None)
        usernames = await session.execute(
            select(User.id, User.handle).filter(User.id.in_(list(user_ids)))
        )
        usernames = dict(list(usernames))

    return templates.TemplateResponse(
        request,
        "claim_cluster_analysis.html",
        dict(
            grouped_clusters=grouped_clusters,
            usernames=usernames,
            analysis=None,
            **base_vars,
        ),
    )


@app_router.get("/c/{collection}/analysis/cluster/{analysis_id}/{cluster_id}")
async def get_claim_cluster_cluster_details(
    request: Request,
    current_user: user_with_coll_permission_c_dep("access"),
    collection: str,
    analysis_id: int,
    cluster_id: int,
):
    cluster_analyzer_id = await ClusterTask.get_analyzer_id()
    async with Session() as session:
        # TODO: Factor out redundant code
        base_vars = await get_base_template_vars(
            request, current_user, collection, session
        )
        collection_ob: Collection = base_vars["collection"]
        analysis = await session.get(Analysis, analysis_id)
        cluster = await session.get(ClusterData, cluster_id)
        model_name = (
            analysis.params.get("model", None) or collection_ob.embed_model().name
        )
        Embedding = embed_db_model_by_name[model_name]
        target = aliased(Embedding, name="target")
        neighbour_embedding = aliased(Embedding, name="neighbour_embedding")
        neighbour = aliased(VisibleClaim, name="neighbour")
        subq = (
            select(target.embedding)
            .filter_by(fragment_id=cluster.distinguished_claim_id)
            .cte()
        )
        distance = neighbour_embedding.distance()(subq.c.embedding).label("rank")
        in_cluster = await session.execute(
            select(neighbour, distance, InClusterData)
            .join(neighbour_embedding, neighbour_embedding.fragment_id == neighbour.id)
            .join(InClusterData, neighbour.in_cluster_rels)
            .join(ClusterData, InClusterData.cluster)
            .filter_by(id=cluster_id)
            .order_by(distance)
        )
        in_cluster = list(in_cluster)
        boundary = max(in_cluster[-1][1], cluster.auto_include_diameter or 0)
        limit = max(2 * boundary, 0.15)
        distances = {f.id: d for (f, d, i) in in_cluster}
        joinrows = {f.id: i for (f, d, i) in in_cluster}
        in_cluster = [f for (f, d, i) in in_cluster]
        q = (
            select(neighbour, distance)
            .join(Collection, neighbour.collections)
            .filter_by(id=collection_ob.id)
            .join(neighbour_embedding, neighbour_embedding.fragment_id == neighbour.id)
        )
        if scale := analysis.params.get("scale", None):
            q = q.filter(neighbour.scale == scale)
        else:
            q = q.filter(poly_type_clause(neighbour))
        q = q.options(
            subqueryload(neighbour.in_cluster_rels).subqueryload(InClusterData.cluster)
        )
        q = q.filter(distance < limit).order_by(distance).limit(50 + len(in_cluster))
        neighbours_r = await session.execute(q)
        neighbours_r = list(neighbours_r)
        near_neighbours = [
            f for (f, d) in neighbours_r if f.id not in distances and d <= boundary
        ]
        far_neighbours = [
            f for (f, d) in neighbours_r if f.id not in distances and d > boundary
        ]
        distances |= {f.id: d for (f, d) in neighbours_r}

        has_confirmed_fragment = any(icd.confirmed_by_id for icd in joinrows.values())

        user_ids = {analysis.creator_id, cluster.relevance_checker_id}
        user_ids.update(jr.confirmed_by_id for jr in joinrows.values())
        user_ids.discard(None)
        usernames = await session.execute(
            select(User.id, User.handle).filter(User.id.in_(list(user_ids)))
        )
        usernames = dict(list(usernames))

    return templates.TemplateResponse(
        request,
        "claim_cluster_cluster_details.html",
        dict(
            cluster=cluster,
            analysis=analysis,
            usernames=usernames,
            boundary=boundary,
            near_neighbours=near_neighbours,
            far_neighbours=far_neighbours,
            has_confirmed_fragment=has_confirmed_fragment,
            distances=distances,
            in_cluster=in_cluster,
            joinrows=joinrows,
            **base_vars,
        ),
    )


@app_router.post("/c/{collection}/analysis/cluster/{analysis_id}/{cluster_id}")
async def claim_cluster_cluster_details(
    request: Request,
    current_user: user_with_coll_permission_c_dep("confirm_claim"),
    collection: str,
    analysis_id: int,
    cluster_id: int,
    target_action: Annotated[str, Form()],
    target_id: Annotated[Union[int, List[int]], Form()],
    relevance: Annotated[Optional[str], Form()] = None,
    auto_include_diameter: Annotated[float, Form()] = 0,
    delete_cluster: Annotated[bool, Form()] = False,
):
    cluster_analyzer_id = await ClusterTask.get_analyzer_id()
    async with Session() as session:
        base_vars = await get_base_template_vars(
            request, current_user, collection, session
        )
        collection_ob: Collection = base_vars["collection"]
        analysis = await session.get(Analysis, analysis_id)
        cluster = await session.get(ClusterData, cluster_id)
        model_name = (
            analysis.params.get("model", None) or collection_ob.embed_model().name
        )
        Embedding = embed_db_model_by_name[model_name]
        if target_action:
            assert target_id
        if target_action == "add":
            fragment = await session.get(Statement, target_id)
            assert fragment
            session.add(
                InClusterData(
                    fragment=fragment,
                    cluster=cluster,
                    manual=True,
                    confirmed_by_id=current_user.id,
                )
            )
            cluster.cluster_size += 1
            # Should we recalculate the centroid?
        elif target_action == "add_up_to":
            target_id = [target_id] if isinstance(target_id, int) else target_id
            fragments = await session.execute(
                select(VisibleClaim).filter(VisibleClaim.id.in_(target_id))
            )
            assert fragments
            for (fragment,) in fragments:
                session.add(
                    InClusterData(
                        fragment=fragment,
                        cluster=cluster,
                        manual=True,
                        confirmed_by_id=current_user.id,
                    )
                )
            cluster.cluster_size += len(fragments)
            # Should we recalculate the centroid?
        elif target_action == "remove":
            if target_id == cluster.distinguished_claim_id:
                raise BadRequest("Cannot delete the last cluster element")
            icd = await session.scalar(
                select(InClusterData).filter_by(
                    cluster_id=cluster_id, fragment_id=target_id
                )
            )
            await session.delete(icd)
            cluster.cluster_size -= 1
            # Should we recalculate the centroid?
        elif target_action == "set_center":
            cluster.distinguished_claim_id = target_id
        elif delete_cluster:
            icds = await session.execute(
                select(InClusterData).filter_by(cluster_id=cluster_id)
            )
            has_confirmed_fragment = any(icd.confirmed_by_id for (icd,) in icds)
            if has_confirmed_fragment:
                raise BadRequest("Do not delete a cluster with confirmed fragments.")
                # Note: What if the last fragment is confirmed? then add some random claim and delete the other one.
            # What to do if last cluster of analysis?
            await session.delete(cluster)
            # Will not adjust num_clusters, because that is historical.
            await session.commit()
            return RedirectResponse(
                f"/f{collection_ob.path}/analysis/cluster/{analysis.id}",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        elif target_action == "confirm":
            icd = await session.scalar(
                select(InClusterData).filter_by(
                    cluster_id=cluster_id, fragment_id=target_id
                )
            )
            icd.confirmed_by_id = current_user.id
        elif target_action == "add_cluster":
            fragment = await session.get(Statement, target_id)
            in_rel = await session.scalar(
                select(InClusterData).filter_by(
                    fragment_id=target_id, cluster_id=cluster_id
                )
            )
            if in_rel:
                session.delete(in_rel)
            params = dict(model=model_name)
            analysis = await session.scalar(
                select(Analysis).filter_by(
                    analyzer_id=cluster_analyzer_id,
                    collection_id=collection_ob.id,
                    params=params,
                )
            )
            if not analysis:
                analysis = Analysis(
                    analyzer_id=cluster_analyzer_id,
                    collection_id=collection_ob.id,
                    params=params,
                    results={},
                    creator_id=current_user.id,
                )
                session.add(analysis)
            cluster = ClusterData(
                analysis=analysis, distinguished_claim=fragment, fragments=[fragment]
            )
            session.add(cluster)
            await session.commit()
            return RedirectResponse(
                f"/f{collection_ob.path}/analysis/cluster/{analysis.id}/{cluster.id}",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        elif not target_action:
            # no target_action, just save
            relevance = cluster.relevant if relevance is None else relevance
            if cluster.relevant != relevance:
                cluster.relevant = relevance
                cluster.relevance_checker_id = current_user.id
                # Should we apply to all the claims now?
            cluster.auto_include_diameter = auto_include_diameter
        else:
            raise BadRequest(f"Unknown target_action: {target_action}")

        await session.commit()

        target = aliased(Embedding, name="target")
        neighbour_embedding = aliased(Embedding, name="neighbour_embedding")
        neighbour = aliased(VisibleClaim, name="neighbour")
        subq = (
            select(target.embedding)
            .filter_by(fragment_id=cluster.distinguished_claim_id)
            .cte()
        )
        distance = neighbour_embedding.distance()(subq.c.embedding).label("rank")
        in_cluster = await session.execute(
            select(neighbour, distance, InClusterData)
            .join(neighbour_embedding, neighbour_embedding.fragment_id == neighbour.id)
            .join(InClusterData, neighbour.in_cluster_rels)
            .join(ClusterData, InClusterData.cluster)
            .filter_by(id=cluster_id)
            .order_by(distance)
        )
        in_cluster = list(in_cluster)
        boundary = max(in_cluster[-1][1], cluster.auto_include_diameter or 0)
        limit = max(2 * boundary, 0.15)
        distances = {f.id: d for (f, d, i) in in_cluster}
        joinrows = {f.id: i for (f, d, i) in in_cluster}
        in_cluster = [f for (f, d, i) in in_cluster]
        q = (
            select(neighbour, distance)
            .join(Collection, neighbour.collections)
            .filter_by(id=collection_ob.id)
            .join(neighbour_embedding, neighbour_embedding.fragment_id == neighbour.id)
        )
        if scale := analysis.params.get("scale", None):
            q = q.filter(neighbour.scale == scale)
        else:
            q = q.filter(poly_type_clause(neighbour))
        q = q.options(
            subqueryload(neighbour.in_cluster_rels).subqueryload(InClusterData.cluster)
        )
        q = q.filter(distance < limit).order_by(distance).limit(50 + len(in_cluster))
        neighbours_r = await session.execute(q)
        neighbours_r = list(neighbours_r)
        near_neighbours = [
            f for (f, d) in neighbours_r if f.id not in distances and d <= boundary
        ]
        far_neighbours = [
            f for (f, d) in neighbours_r if f.id not in distances and d > boundary
        ]
        distances |= {f.id: d for (f, d) in neighbours_r}

        has_confirmed_fragment = any(icd.confirmed_by_id for icd in joinrows.values())

        user_ids = {analysis.creator_id, cluster.relevance_checker_id}
        user_ids.update(jr.confirmed_by_id for jr in joinrows.values())
        user_ids.discard(None)
        usernames = await session.execute(
            select(User.id, User.handle).filter(User.id.in_(list(user_ids)))
        )
        usernames = dict(list(usernames))

    return templates.TemplateResponse(
        request,
        "claim_cluster_cluster_details.html",
        dict(
            cluster=cluster,
            analysis=analysis,
            usernames=usernames,
            boundary=boundary,
            near_neighbours=near_neighbours,
            far_neighbours=far_neighbours,
            has_confirmed_fragment=has_confirmed_fragment,
            distances=distances,
            in_cluster=in_cluster,
            joinrows=joinrows,
            **base_vars,
        ),
    )


@app_router.get("/c/{collection}/analysis/cluster/{analysis_id}/old")
async def claim_cluster_analysis(
    request: Request,
    current_user: user_with_coll_permission_c_dep("access"),
    collection: str,
    analysis_id: int,
):
    # Might be ready for deletion
    cluster_analyzer_id = await ClusterTask.get_analyzer_id()
    async with Session() as session:
        base_vars = await get_base_template_vars(
            request, current_user, collection, session
        )
        analysis = await session.get(Analysis, analysis_id)
        # TODO: Order by distance from distinguished here also?
        clusters = await session.execute(
            select(ClusterData)
            .join(Analysis)
            .filter(Analysis.id == analysis_id)
            .order_by(ClusterData.cluster_size.desc())
            .options(
                joinedload(ClusterData.distinguished_claim),
                subqueryload(ClusterData.has_cluster_rels).subqueryload(
                    InClusterData.fragment
                ),
            )
        )
        clusters = [cluster for (cluster,) in clusters]

        user_ids = {analysis.creator_id}
        for cluster in clusters:
            user_ids.update(icd.confirmed_by_id for icd in cluster.has_cluster_rels)
        user_ids.discard(None)
        usernames = await session.execute(
            select(User.id, User.handle).filter(User.id.in_(list(user_ids)))
        )
        usernames = dict(list(usernames))

    return templates.TemplateResponse(
        request,
        "claim_cluster_analysis.html",
        dict(
            grouped_clusters=((analysis, clusters),),
            analysis=analysis,
            usernames=usernames,
            **base_vars,
        ),
    )
