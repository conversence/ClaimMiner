-- Deploy base_functions
-- requires: base
-- idempotent

BEGIN;

CREATE OR REPLACE FUNCTION public.encode_uuid(id UUID) RETURNS varchar(22) LANGUAGE SQL IMMUTABLE AS $$
        SELECT replace(replace(
        trim(trailing '=' FROM encode(decode(replace(id::varchar, '-', ''), 'hex'), 'base64'))
        , '+', '-'), '/', '_');
$$;

CREATE OR REPLACE FUNCTION public.decode_uuid(id text) RETURNS UUID LANGUAGE SQL IMMUTABLE AS $$
        SELECT encode(decode(
                replace(replace(id, '_', '/'), '-', '+') || substr('==', 1, (33-length(id)) % 3), 'base64'), 'hex')::uuid;
$$;


COMMIT;
