-- Deploy schema_def
-- requires: topic
-- copyright Conversence 2023-2024
-- license: Apache 2.0

BEGIN;

CREATE TYPE public.ontology_status AS ENUM (
    'draft',
    'published',
    'deprecated',
    'obsolete'
);

CREATE TABLE IF NOT EXISTS public.namespace (
    prefix varchar(10) NOT NULL PRIMARY KEY,
    uri varchar NOT NULL,
    is_base bool DEFAULT false
);

CREATE UNIQUE INDEX namespace_uri_idx ON public.namespace (uri);

CREATE TABLE IF NOT EXISTS public.ontology (
  prefix varchar(10) PRIMARY KEY NOT NULL,
  status ontology_status DEFAULT 'draft',
  ontology_language varchar DEFAULT 'linkml',
  data JSONB NOT NULL DEFAULT '{}',
  CONSTRAINT ontology_prefix_fkey  FOREIGN KEY (prefix)
      REFERENCES public.namespace (prefix) ON DELETE CASCADE ON UPDATE CASCADE
);

CREATE INDEX ontology_data_idx on ontology USING gin (data jsonb_path_ops);

CREATE TABLE IF NOT EXISTS public.schema_term (
  id BIGINT PRIMARY KEY NOT NULL,
  ontology_prefix VARCHAR(10) NOT NULL,
  term varchar NOT NULL,
  public_term varchar,
  parent_id BIGINT,
  ancestors_id BIGINT[],
  CONSTRAINT schema_term_id_fkey  FOREIGN KEY (id)
      REFERENCES public.topic (id) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT schema_term_ontology_prefix_fkey  FOREIGN KEY (ontology_prefix)
      REFERENCES ontology (prefix) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT schema_term_parent_fkey FOREIGN KEY (parent_id)
      REFERENCES public.schema_term (id) ON DELETE SET NULL ON UPDATE CASCADE
);

CREATE UNIQUE INDEX schema_prefix_term_idx on schema_term (ontology_prefix, term);
CREATE UNIQUE INDEX schema_term_public_term_idx on schema_term (public_term);
CREATE INDEX schema_term_parent_idx on schema_term (parent_id);
CREATE INDEX schema_term_ancestors_idx on schema_term USING gin (ancestors_id);

ALTER TABLE public.topic ADD CONSTRAINT topic_schema_term_fkey FOREIGN KEY (schema_term_id)
    REFERENCES public.schema_term (id) ON DELETE SET NULL ON UPDATE CASCADE;

COMMIT;
