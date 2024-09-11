from collections import defaultdict
from itertools import groupby
from typing import Optional
from logging import getLogger

import numpy as np
from numpy.linalg import norm
from sqlalchemy.sql import func
from sqlalchemy.orm import subqueryload, joinedload
from sklearn.cluster import DBSCAN

from .. import Session, select, run_sync
from ..models import (
    Statement,
    VisibleClaim,
    Collection,
    embed_db_model_by_name,
    ClusterData,
    InClusterData,
    Analysis,
    poly_type_clause,
    aliased,
)
from ..pyd_models import fragment_type
from .tasks import ClusterTask, ClusterAnalysisModel

logger = getLogger(__name__)


def choose_representative(embeddings):
    # naively choose the one closest to the centroid
    embeds = np.array(embeddings)
    mean = embeds.sum(axis=0)
    mean = mean / norm(mean)
    similarity = embeds.dot(mean)
    return np.argmax(similarity)


async def do_cluster_claims(analysis_id):
    analyzer_id = await ClusterTask.get_analyzer_id()
    async with Session() as session:
        analysis = await session.get(
            Analysis, analysis_id, options=[joinedload(Analysis.collection)]
        )
        analysis_model: ClusterAnalysisModel = analysis.as_model(session)
        collection_ob = analysis.collection
        collection = collection_ob.name
        model = analysis_model.model or collection_ob.embed_model()
        scale = analysis_model.scale
        Embedding = embed_db_model_by_name[model.name]
        q = select(Embedding.embedding, VisibleClaim).join(VisibleClaim)
        q = q.filter(poly_type_clause(VisibleClaim))
        if scale:
            q = q.filter_by(scale=fragment_type[scale])
        else:
            q = q.filter(VisibleClaim.scale != fragment_type.standalone_category)
        q = q.join(Collection, VisibleClaim.collections).filter(
            Collection.name == collection
        )
        q2 = q.with_only_columns(
            func.count(Embedding.fragment_id), func.max(Embedding.fragment_id)
        )
        r = await session.execute(q2)
        count_ids, max_id = r.one()
        if not count_ids:
            logger.error("No claims!")
            return None
        # Check for existing analysis. Should happen upstream.
        params = dict(
            model=model.name,
            eps=analysis_model.eps,
            min_samples=analysis_model.min_samples,
            scale=scale.name if scale else None,
            collection=collection,
        )
        existing = await session.scalar(
            select(Analysis).filter_by(
                analyzer_id=analyzer_id, params=params, theme_id=None
            )
        )
        if existing and existing.id != analysis_id:
            if (
                existing.results["max_id"] != max_id
                or existing.results[count_ids] != count_ids
            ):
                # Old analysis with different set of claims
                await session.delete(existing)
            else:
                logger.warning("Analysis exists")
                return existing
        data = await session.execute(q)
        clusters = defaultdict(list)
        cluster_embeddings = defaultdict(list)
        data = list(zip(*data))
        (embeds, fragments) = data
        embeds = np.array(embeds)
        assert analysis_model.algorithm == "dbscan"
        # TODO: Implement more algorithms
        logger.debug("scanning %d", len(fragments))
        scan = DBSCAN(
            eps=analysis_model.eps,
            min_samples=analysis_model.min_samples,
            metric="cosine",
        )
        db = await run_sync(scan.fit)(embeds)
        for i, c in enumerate(db.labels_):
            if c == -1:
                continue
            clusters[c].append(fragments[i])
            cluster_embeddings[c].append(embeds[i])
        results = dict(
            max_id=max_id,
            count_ids=count_ids,
            num_clusters=len(clusters),
            num_outliers=count_ids - sum(len(v) for v in clusters.values()),
        )
        # TODO: Use hierarchical analysis
        for label, fragments in clusters.items():
            representative_pos = choose_representative(cluster_embeddings[label])
            cluster_data = ClusterData(
                analysis=analysis,
                cluster_size=len(fragments),
                fragments=fragments,
                distinguished_claim=fragments[representative_pos],
            )
            logger.debug("%s", cluster_data.__dict__)
            session.add(cluster_data)
        await session.commit()
        return analysis


async def do_autoclassify(claim_id):
    async with Session() as session:
        claim = await session.get(Statement, claim_id)
        if not claim:
            return False
        candidates = await session.execute(
            select(ClusterData)
            .filter(ClusterData.auto_include_diameter > 0)
            .join(ClusterData.analysis)
            .order_by(
                Analysis.params["model"],
                Analysis.collection_id,
                ClusterData.auto_include_diameter,
            )
            .options(
                subqueryload(ClusterData.analysis).subqueryload(Analysis.collection)
            )
        )
        candidates = (c for (c,) in candidates)
        for model, c_by_model in groupby(
            candidates, key=lambda x: x.analysis.params["model"]
        ):
            c_by_model = list(c_by_model)
            Embedding = embed_db_model_by_name.get(model)
            if not Embedding:
                logger.error(f"Invalid model in analysis {c_by_model[0].analysis_id}")
                continue
            target = aliased(Embedding, name="target")
            neighbour_embedding = aliased(Embedding, name="neighbour_embedding")
            subq = select(target.embedding).filter_by(fragment_id=claim_id).cte()
            distance = neighbour_embedding.distance()(subq.c.embedding).label("rank")
            for collection, c_by_coll in groupby(
                c_by_model, lambda x: x.analysis.collection
            ):
                c_by_coll = [
                    c
                    for c in c_by_coll
                    if (not c.analysis.params.get("scale"))
                    or (c.analysis.params["scale"] == claim.scale.name)
                ]
                c_by_claim = {c.distinguished_claim_id: c for c in c_by_coll}
                embeddings = await session.execute(
                    select(neighbour_embedding.fragment_id, distance)
                    .filter(
                        neighbour_embedding.fragment_id.in_(list(c_by_claim.keys()))
                    )
                    .order_by(distance)
                )
                included = []
                for fragment_id, distance in embeddings:
                    candidate = c_by_claim[fragment_id]
                    if distance <= candidate.auto_include_diameter:
                        included.append((candidate, distance))
                if included:
                    # Choose the one with the most restrictive inclusion criterion, maybe not closest
                    included.sort(key=lambda x: x[0].auto_include_diameter)
                    session.add(
                        InClusterData(cluster=included[0][0], fragment_id=claim_id)
                    )
                    # Should there be a way to mark that this comes from auto-inclusion and not original clustering algorithm?
        await session.commit()


async def do_autoclassify_all(collection=None):
    async with Session() as session:
        q = (
            select(ClusterData)
            .filter(ClusterData.auto_include_diameter > 0)
            .join(ClusterData.analysis)
        )
        if collection:
            q = q.join(Analysis.collection).filter_by(name=collection)
        q = q.order_by(
            Analysis.params["model"],
            Analysis.collection_id,
            ClusterData.auto_include_diameter,
        ).options(subqueryload(ClusterData.analysis).subqueryload(Analysis.collection))
        candidates = await session.execute(q)
        candidates = (c for (c,) in candidates)
        for model, c_by_model in groupby(
            candidates, key=lambda x: x.analysis.params["model"]
        ):
            c_by_model = list(c_by_model)
            Embedding = embed_db_model_by_name.get(model)
            if not Embedding:
                logger.error(f"Invalid model in analysis {c_by_model[0].analysis_id}")
                continue
            target = aliased(Embedding, name="target")
            neighbour_embedding = aliased(Embedding, name="neighbour_embedding")
            for collection, c_by_coll in groupby(
                c_by_model, lambda x: x.analysis.collection
            ):
                for cluster in c_by_coll:
                    subq = (
                        select(target.embedding)
                        .filter_by(fragment_id=cluster.distinguished_claim_id)
                        .cte()
                    )
                    distance = neighbour_embedding.distance()(subq.c.embedding).label(
                        "rank"
                    )
                    q = (
                        select(neighbour_embedding.fragment_id)
                        .join(VisibleClaim, neighbour_embedding.fragment)
                        .join(VisibleClaim.collections)
                        .filter_by(id=collection.id)
                        .outerjoin(VisibleClaim.in_cluster_rels)
                        .filter(
                            InClusterData.cluster_id.is_(None),
                            distance <= cluster.auto_include_diameter,
                        )
                    )
                    q = q.filter(poly_type_clause(VisibleClaim))
                    if scale := cluster.analysis.params.get("scale"):
                        q = q.filter(VisibleClaim.scale == fragment_type[scale])
                    candidates = await session.execute(q)
                    for (fragment_id,) in candidates:
                        session.add(
                            InClusterData(cluster=cluster, fragment_id=fragment_id)
                        )
                        # Should there be a way to mark that this comes from auto-inclusion and not original clustering algorithm?
                    await (
                        session.flush()
                    )  # The next query should take those new clusters in account.
                    # Note we've been ordering clusters by their include diiameter
        await session.commit()
