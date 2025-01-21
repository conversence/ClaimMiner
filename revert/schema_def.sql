-- Deploy schema_def


BEGIN;

ALTER TABLE public.topic DROP CONSTRAINT topic_schema_term_fkey;

DROP TABLE IF EXISTS public.schema_term;
DROP TABLE IF EXISTS public.ontology;
DROP TABLE IF EXISTS public.namespace;

COMMIT;
