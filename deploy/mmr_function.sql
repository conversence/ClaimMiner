-- Deploy mmr_function
-- admin
-- idempotent
-- requires: embedding
-- copyright Conversence 2023-2024
-- license: Apache 2.0

drop function IF EXISTS mmr(vector, varchar, varchar[], integer, double precision, integer);
drop function IF EXISTS mmr(vector, integer, varchar, varchar[], integer, double precision, integer);

CREATE OR REPLACE FUNCTION mmr(query_embedding vector, query_fragment bigint, embedding_table varchar, scales varchar[], k integer, lam float, qlimit integer) RETURNS SETOF id_score_type AS $$
    import numpy as np
    pos_by_id = {}

    def to_array(vs, normalize=True):
        array = np.fromstring(vs.strip('[').strip(']'), dtype=float, sep=',')
        if normalize:
            array /= np.linalg.norm(array)
        return array
    scales_clause = f"= '{scales[0]}'" if len(scales) == 1 else f"""IN ('{"','".join(scales)}')"""
    if query_embedding:
        plan = plpy.prepare(f"""
            SELECT fragment_id AS doc_id, embedding <=> $1 AS distance, embedding
            FROM {embedding_table} AS emb
            JOIN fragment ON emb.fragment_id = fragment.id
            WHERE fragment.scale {scales_clause}
            ORDER BY distance LIMIT {qlimit}""",
            ["vector"])
        results = plpy.execute(plan, [query_embedding])
    elif query_fragment:
        plan = plpy.prepare(f"""
            SELECT fragment_id AS doc_id, emb.embedding <=> (SELECT embedding FROM {embedding_table} WHERE fragment_id = $1) AS distance, emb.embedding
            FROM {embedding_table} AS emb
            JOIN fragment ON emb.fragment_id = fragment.id
            WHERE fragment.scale {scales_clause} AND fragment.id != $1
            ORDER BY distance LIMIT {qlimit}""",
            ["bigint"])
        results = plpy.execute(plan, [query_fragment])
    else:
        return []
    doc_ids = np.array([r["doc_id"] for r in results], dtype=int)
    scores = 1.0 - np.array([r["distance"] for r in results], dtype=float)
    embeddings = np.array([to_array(r["embedding"]) for r in results])
    pos = 0
    while scores[pos] > 0.999:
        pos += 1
    if pos > 0:
        doc_ids = doc_ids[pos:]
        scores = scores[pos:]
        embeddings = embeddings[pos:]
        pos = 0
    pos_results = []
    result_scores = []
    scores_max = np.zeros(doc_ids.size) - 1
    new_scores = scores
    while len(pos_results) < k:
        pos_results.append(pos)
        result_scores.append(new_scores[pos])
        scores_max = np.maximum(scores_max, embeddings.dot(embeddings[pos]))
        new_scores = scores * lam - (scores_max * (1-lam))
        for x in pos_results:
            new_scores[x] = -1
        pos = np.argmax(new_scores)
    return [dict(id=doc_ids[p], score=result_scores[i]) for i, p in enumerate(pos_results)]
$$ LANGUAGE plpython3u;
