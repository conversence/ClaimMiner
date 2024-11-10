-- Deploy schema_def_functions


BEGIN;

DROP TRIGGER IF EXISTS before_create_schema_def ON schema_def;
DROP TRIGGER IF EXISTS before_update_schema_def ON schema_def;
DROP TRIGGER IF EXISTS after_update_schema_def ON schema_def;

DROP FUNCTION IF EXISTS public.schema_in_type_terms(schema_def_id BIGINT, term_list varchar[]);
DROP FUNCTION IF EXISTS public.schema_in_type_ids(schema_def_id BIGINT, type_list BIGINT[]);
DROP FUNCTION IF EXISTS public.intersect_p(a BIGINT[], b BIGINT[]);
DROP FUNCTION IF EXISTS public.calc_schema_def_ancestors(parent_id_ BIGINT);
DROP PROCEDURE IF EXISTS update_schema_def_ancestors(schema_id BIGINT);
DROP FUNCTION IF EXISTS public.before_update_schema_def();
DROP FUNCTION IF EXISTS public.after_update_schema_def();

COMMIT;
