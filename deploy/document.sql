-- Deploy document
-- requires: user
-- requires: collection
-- requires: analyzer
-- requires: uri_equiv
-- Copyright Society Library and Conversence 2022-2024

BEGIN;

CREATE TABLE IF NOT EXISTS public.document (
    id bigint NOT NULL,
    uri_id bigint NOT NULL,
    is_archive boolean NOT NULL DEFAULT false,
    requested timestamp with time zone NOT NULL default now(),
    return_code smallint,
    retrieved timestamp without time zone,
    created timestamp without time zone,
    modified timestamp without time zone,
    mimetype character varying(255),
    language character varying(16),
    text_analyzer_id bigint,
    etag varchar(64),
    file_identity char(64),
    file_size integer,
    text_identity char(64),
    text_size integer,
    title text,
    process_params JSONB,
    metadata JSONB default '{}',
    public_contents boolean NOT NULL default true,
    CONSTRAINT document_pkey PRIMARY KEY (id),
    CONSTRAINT document_id_fkey  FOREIGN KEY (id)
      REFERENCES public.topic (id) ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT document_url_id_key FOREIGN KEY (uri_id)
      REFERENCES uri_equiv (id),
    CONSTRAINT document_text_analyzer_id_key FOREIGN KEY (text_analyzer_id)
      REFERENCES public.analyzer (id) ON DELETE SET NULL ON UPDATE CASCADE
);

CREATE UNIQUE INDEX IF NOT EXISTS document_uri_id_idx ON public.document (uri_id) WHERE (not is_archive);

COMMIT;
