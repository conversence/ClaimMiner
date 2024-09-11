-- Deploy uri_equiv
-- requires: base

BEGIN;

CREATE TYPE public.uri_status AS ENUM (
  'canonical',
  'urn',
  'snapshot',
  'alt',
  'unknown'
);

CREATE TABLE IF NOT EXISTS public.uri_equiv (
    id bigint NOT NULL DEFAULT nextval('public.topic_id_seq'::regclass),
    status uri_status NOT NULL DEFAULT 'unknown',
    canonical_id bigint,
    uri character varying(2048) NOT NULL,
    CONSTRAINT uri_pkey PRIMARY KEY (id),
    CONSTRAINT uri_uri_key UNIQUE (uri),
    CONSTRAINT uri_canonical_key FOREIGN KEY (canonical_id)
      REFERENCES public.uri_equiv (id) ON DELETE SET NULL ON UPDATE CASCADE
);

COMMIT;
