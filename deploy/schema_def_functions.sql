-- Deploy schema_term_functions
-- requires: schema_def
-- idempotent
-- copyright Conversence 2023
-- license: Apache 2.0

BEGIN;

CREATE OR REPLACE FUNCTION public.calc_schema_term_ancestors(parent_id_ BIGINT) RETURNS BIGINT[] LANGUAGE sql STABLE AS $$
  SELECT CASE WHEN parent_id_ IS NULL THEN '{}'::bigint[] ELSE (SELECT ancestors_id FROM public.schema_term WHERE id = parent_id_) END CASE;
$$;

CREATE OR REPLACE PROCEDURE update_schema_term_ancestors(schema_id BIGINT) LANGUAGE sql AS $$
  UPDATE public.schema_term SET ancestors_id = array_append(calc_schema_term_ancestors(parent_id), id) WHERE id=schema_id;
$$;

CREATE OR REPLACE FUNCTION public.before_update_schema_term() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  IF OLD.term IS NULL OR coalesce(NEW.parent_id, -1) != coalesce(OLD.parent_id, -1) THEN
    NEW.ancestors_id := array_append(calc_schema_term_ancestors(NEW.parent_id), NEW.id);
  END IF;
  RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS before_create_schema_term ON public.schema_term;
CREATE TRIGGER before_create_schema_term BEFORE INSERT ON public.schema_term FOR EACH ROW EXECUTE FUNCTION public.before_update_schema_term();

DROP TRIGGER IF EXISTS before_update_schema_term ON public.schema_term;
CREATE TRIGGER before_update_schema_term BEFORE UPDATE ON public.schema_term FOR EACH ROW EXECUTE FUNCTION public.before_update_schema_term();

CREATE OR REPLACE FUNCTION public.after_update_schema_term() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE child_id BIGINT;
BEGIN
  IF NEW.ancestors_id != OLD.ancestors_id THEN
    FOR child_id IN
      SELECT id FROM schema_term WHERE parent_id = NEW.id
    LOOP
      CALL update_schema_term_ancestors(child_id);
    END LOOP;
  END IF;
  RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS after_update_schema_term ON public.schema_term;
CREATE TRIGGER after_update_schema_term AFTER UPDATE ON public.schema_term FOR EACH ROW EXECUTE FUNCTION public.after_update_schema_term();


CREATE OR REPLACE FUNCTION public.intersect_p(a BIGINT[], b BIGINT[])
  RETURNS bool
  language sql IMMUTABLE
AS $$
    WITH A AS (SELECT UNNEST(a) = any(b) AS t1) SELECT bool_or(A.t1) FROM A;
$$;

CREATE OR REPLACE FUNCTION public.schema_in_type_ids(schema_term_id BIGINT, type_list BIGINT[])
  RETURNS bool
  language sql STABLE
AS $$
  SELECT intersect_p(ancestors_id, type_list) FROM schema_term WHERE id=schema_term_id;
$$;

CREATE OR REPLACE FUNCTION public.schema_in_type_terms(schema_term_id BIGINT, term_list varchar[])
  RETURNS bool
  language sql STABLE
AS $$
  WITH type_ids AS (SELECT array_agg(id) AS type_list FROM schema_term WHERE term = any(term_list))
  SELECT intersect_p(ancestors_id, type_ids.type_list) FROM schema_term, type_ids WHERE id=schema_term_id;
$$;

COMMIT;
