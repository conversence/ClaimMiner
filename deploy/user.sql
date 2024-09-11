-- Deploy user
-- requires: topic
-- Copyright Society Library and Conversence 2022-2024

BEGIN;

CREATE TABLE IF NOT EXISTS public.user (
    id bigint NOT NULL,
    email character varying(255) NOT NULL,
    handle character varying(255) NOT NULL,
    passwd character varying(255) NOT NULL,
    confirmed boolean DEFAULT false,
    created timestamp without time zone NOT NULL default now(),
    permissions public.permission[] DEFAULT ARRAY[]::public.permission[],
    external_id character varying(255),
    picture_url character varying(255),
    name character varying(255),
    CONSTRAINT user_pkey PRIMARY KEY (id),
    CONSTRAINT user_id_fkey FOREIGN KEY (id)
      REFERENCES public.topic(id) ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT user_handle_key UNIQUE (handle),
    CONSTRAINT user_email_key UNIQUE (email)
);

CREATE UNIQUE INDEX IF NOT EXISTS user_external_id_idx ON public.user (external_id);

COMMIT;
