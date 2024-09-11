-- Deploy analyzer
-- requires: topic
-- Copyright Society Library and Conversence 2022-2024

BEGIN;


CREATE TABLE IF NOT EXISTS public.analyzer (
    id bigint PRIMARY KEY,
    name character varying(100) NOT NULL,
    version smallint NOT NULL,
    CONSTRAINT analyzer_id_fkey FOREIGN KEY (id)
      REFERENCES public.topic(id) ON DELETE CASCADE ON UPDATE CASCADE
);

CREATE UNIQUE INDEX IF NOT EXISTS "analyzer_name_version_idx" ON public.analyzer USING btree (name, version);

COMMIT;
