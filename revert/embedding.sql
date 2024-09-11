-- Deploy embedding
-- Copyright Society Library and Conversence 2022-2024

BEGIN;

DO $$DECLARE
  name varchar;
BEGIN
  FOR name IN (SELECT typname FROM pg_type JOIN pg_namespace ON (typnamespace=pg_namespace.oid)
    WHERE typname LIKE 'embedding_%' AND typtype='c' AND nspname='public') LOOP
    EXECUTE format('DROP TABLE IF EXISTS public.%I', name);
  END LOOP;
END;$$;

DROP TYPE IF EXISTS public.embedding_model;

COMMIT;
