-- Deploy structured_idea


BEGIN;

DROP TABLE IF EXISTS public.structured_idea;
DROP FUNCTION IF EXISTS public.extract_references(JSONB);

COMMIT;
