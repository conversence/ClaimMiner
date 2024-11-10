-- Deploy schema


BEGIN;

ALTER TABLE public.topic DROP CONSTRAINT topic_schema_def_fkey;

DROP TABLE IF EXISTS public.schema_def;
DROP TABLE IF EXISTS public.namespace;

COMMIT;
