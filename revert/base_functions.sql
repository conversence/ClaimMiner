-- Deploy base_functions


BEGIN;

DROP FUNCTION IF EXISTS public.encode_uuid(id UUID);
DROP FUNCTION IF EXISTS public.decode_uuid(id text);

COMMIT;
