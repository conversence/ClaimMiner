-- Deploy base
-- Copyright Society Library and Conversence 2022-2024

BEGIN;


DROP TYPE IF EXISTS public.permission;
DROP TYPE IF EXISTS public.process_status;
DROP TYPE IF EXISTS public.fragment_size;
DROP TYPE IF EXISTS public.fragment_type;
DROP TYPE IF EXISTS public.link_type;
DROP TYPE if EXISTS public.id_score_type;
DROP TYPE if EXISTS public.relevance;
DROP TYPE if EXISTS public.topic_type;

DROP SEQUENCE IF EXISTS public.topic_id_seq;

COMMIT;
