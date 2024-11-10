-- Deploy structured_idea_functions
-- requires: structured_idea
-- requires: claim_link
-- idempotent
-- copyright Conversence 2023
-- license: Apache 2.0

BEGIN;


CREATE OR REPLACE FUNCTION public.in_structures_rec(target_id BIGINT) RETURNS TABLE(id BIGINT) LANGUAGE sql STABLE AS $$
WITH RECURSIVE t(x) AS (
  values (target_id) UNION ALL
  SELECT id FROM public.structured_idea, t
  WHERE ARRAY[t.x] && refs
  )
  SELECT DISTINCT x AS id FROM t;
$$;

CREATE OR REPLACE FUNCTION public.in_structures_rec_filtered_ids(target_id BIGINT, structure_types BIGINT[]) RETURNS TABLE(id BIGINT) LANGUAGE sql STABLE AS $$
WITH RECURSIVE t(x) AS (
  values (target_id) UNION ALL
  SELECT si.id FROM t, public.structured_idea AS si
  JOIN public.topic USING(id)
  WHERE ARRAY[t.x] && si.refs AND schema_in_type_ids(topic.schema_def_id, structure_types))
  SELECT DISTINCT x AS id FROM t;
$$;

CREATE OR REPLACE FUNCTION public.in_structures_rec_filtered_terms(target_id BIGINT, structure_terms varchar[]) RETURNS TABLE(id BIGINT) LANGUAGE sql STABLE AS $$
WITH RECURSIVE t(x) AS (
  values (target_id) UNION ALL
  SELECT si.id FROM t, public.structured_idea AS si
  JOIN public.topic USING(id)
  WHERE ARRAY[t.x] && si.refs AND schema_in_type_terms(topic.schema_def_id, structure_terms))
  SELECT DISTINCT x AS id FROM t;
$$;

CREATE OR REPLACE FUNCTION public.sub_structures_rec(target_id BIGINT) RETURNS TABLE(id BIGINT) LANGUAGE sql STABLE AS $$
WITH RECURSIVE t(x) AS (
  VALUES (target_id) UNION ALL
  SELECT UNNEST(refs) FROM public.structured_idea, t
  WHERE id=t.x)
  SELECT DISTINCT x AS id FROM t;
$$;

CREATE OR REPLACE FUNCTION public.structure_neighbourhood(target_id BIGINT) RETURNS TABLE(id BIGINT) LANGUAGE sql STABLE AS $$
WITH super_structs AS (SELECT in_structures_rec(target_id) AS id),
     all_topics_rep AS (SELECT sub_structures_rec(id) AS id FROM super_structs)
SELECT DISTINCT id FROM all_topics_rep
$$;

CREATE OR REPLACE FUNCTION public.structure_neighbourhood_filtered_ids(target_id BIGINT, structure_types BIGINT[]) RETURNS TABLE(id BIGINT) LANGUAGE sql STABLE AS $$
WITH super_structs AS (SELECT in_structures_rec_filtered_ids(target_id, structure_types) AS id),
     all_topics_rep AS (SELECT sub_structures_rec(id) AS id FROM super_structs)
SELECT DISTINCT id FROM all_topics_rep
$$;

CREATE OR REPLACE FUNCTION public.structure_neighbourhood_filtered_terms(target_id BIGINT, structure_terms varchar[]) RETURNS TABLE(id BIGINT) LANGUAGE sql STABLE AS $$
WITH super_structs AS (SELECT in_structures_rec_filtered_terms(target_id, structure_terms) AS id),
     all_topics_rep AS (SELECT sub_structures_rec(id) AS id FROM super_structs)
SELECT DISTINCT id FROM all_topics_rep
$$;


CREATE OR REPLACE FUNCTION public.in_structures_cl(target_id BIGINT) RETURNS TABLE(id BIGINT) LANGUAGE sql STABLE AS $$
  SELECT id FROM public.structured_idea AS si
  WHERE ARRAY[target_id] && refs
  UNION ALL SELECT id FROM public.claim_link AS cl
  WHERE target_id = cl.source OR target_id = cl.target;
$$;

CREATE OR REPLACE FUNCTION public.in_structures_cl_rec(target_id BIGINT) RETURNS TABLE(id BIGINT) LANGUAGE sql STABLE AS $$
WITH RECURSIVE t(x) AS (
  values (target_id) UNION ALL
  SELECT id FROM t, LATERAL (SELECT in_structures_cl(t.x) AS id)
  )
  SELECT DISTINCT x AS id FROM t;
$$;

CREATE OR REPLACE FUNCTION public.in_structures_cl_filtered_ids(target_id BIGINT, structure_types BIGINT[]) RETURNS TABLE(id BIGINT) LANGUAGE sql STABLE AS $$
  SELECT id FROM public.structured_idea AS si
  JOIN public.topic USING(id)
  WHERE ARRAY[target_id] && refs AND schema_in_type_ids(topic.schema_def_id, structure_types)
  UNION ALL SELECT id FROM public.claim_link AS cl
  JOIN public.topic USING(id)
  WHERE (target_id = cl.source OR target_id = cl.target) AND schema_in_type_ids(topic.schema_def_id, structure_types);
$$;

CREATE OR REPLACE FUNCTION public.in_structures_cl_filtered_terms(target_id BIGINT, structure_terms varchar[]) RETURNS TABLE(id BIGINT) LANGUAGE sql STABLE AS $$
  SELECT id FROM public.structured_idea AS si
  JOIN public.topic USING(id)
  WHERE ARRAY[target_id] && refs AND schema_in_type_terms(topic.schema_def_id, structure_terms)
  UNION ALL SELECT id FROM public.claim_link AS cl
  JOIN public.topic USING(id)
  WHERE (target_id = cl.source OR target_id = cl.target) AND schema_in_type_terms(topic.schema_def_id, structure_terms);
$$;


CREATE OR REPLACE FUNCTION public.in_structures_cl_rec_filtered_ids(target_id BIGINT, structure_types BIGINT[]) RETURNS TABLE(id BIGINT) LANGUAGE sql STABLE AS $$
WITH RECURSIVE t(x) AS (
  values (target_id) UNION ALL
  SELECT id FROM t, LATERAL (SELECT in_structures_cl_filtered_ids(t.x, structure_types) AS id)
  )
  SELECT DISTINCT x AS id FROM t;
$$;


CREATE OR REPLACE FUNCTION public.in_structures_cl_rec_filtered_terms(target_id BIGINT, structure_terms varchar[]) RETURNS TABLE(id BIGINT) LANGUAGE sql STABLE AS $$
WITH RECURSIVE t(x) AS (
  values (target_id) UNION ALL
  SELECT id FROM t, LATERAL (SELECT in_structures_cl_filtered_terms(t.x, structure_terms) AS id)
  )
  SELECT DISTINCT x AS id FROM t;
$$;

CREATE OR REPLACE FUNCTION public.sub_structures(target_id BIGINT) RETURNS TABLE(id BIGINT) LANGUAGE sql STABLE AS $$
  SELECT UNNEST(refs) AS id FROM public.structured_idea WHERE id = target_id
  UNION SELECT source AS id FROM public.claim_link WHERE id = target_id
  UNION SELECT target AS id FROM public.claim_link WHERE id = target_id
$$;

CREATE OR REPLACE FUNCTION public.sub_structures_cl_rec(target_id BIGINT) RETURNS TABLE(id BIGINT) LANGUAGE sql STABLE AS $$
WITH RECURSIVE t(x) AS (
  VALUES (target_id) UNION ALL
  SELECT id FROM t, LATERAL (SELECT id FROM sub_structures(t.x) AS id))
  SELECT DISTINCT x AS id FROM t;
$$;

CREATE OR REPLACE FUNCTION public.structure_neighbourhood_cl(target_id BIGINT) RETURNS TABLE(id BIGINT) LANGUAGE sql STABLE AS $$
WITH super_structs AS (SELECT in_structures_cl_rec(target_id) AS id),
     all_topics_rep AS (SELECT sub_structures_cl_rec(id) AS id FROM super_structs)
SELECT DISTINCT id FROM all_topics_rep
$$;

CREATE OR REPLACE FUNCTION public.structure_neighbourhood_cl_filtered_ids(target_id BIGINT, structure_types BIGINT[]) RETURNS TABLE(id BIGINT) LANGUAGE sql STABLE AS $$
WITH super_structs AS (SELECT in_structures_cl_rec_filtered_ids(target_id, structure_types) AS id),
     all_topics_rep AS (SELECT sub_structures_cl_rec(id) AS id FROM super_structs)
SELECT DISTINCT id FROM all_topics_rep
$$;

CREATE OR REPLACE FUNCTION public.structure_neighbourhood_cl_filtered_terms(target_id BIGINT, structure_terms varchar[]) RETURNS TABLE(id BIGINT) LANGUAGE sql STABLE AS $$
WITH super_structs AS (SELECT in_structures_cl_rec_filtered_terms(target_id, structure_terms) AS id),
     all_topics_rep AS (SELECT sub_structures_cl_rec(id) AS id FROM super_structs)
SELECT DISTINCT id FROM all_topics_rep
$$;

CREATE OR REPLACE FUNCTION public.before_update_structured_idea() RETURNS trigger LANGUAGE plpgsql
  AS $$
BEGIN
  NEW.modified_at = now();
  return NEW;
END;
$$;

DROP TRIGGER IF EXISTS before_update_structured_idea ON structured_idea;
CREATE TRIGGER before_update_structured_idea BEFORE UPDATE ON public.structured_idea FOR EACH ROW EXECUTE FUNCTION public.before_update_structured_idea();


COMMIT;

-- todo: Sanity check on topic category when inserting/updating id in a depenent table
