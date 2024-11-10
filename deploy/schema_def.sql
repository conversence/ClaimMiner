-- Deploy schema_def
-- requires: topic
-- copyright Conversence 2023
-- license: Apache 2.0

BEGIN;

CREATE TABLE IF NOT EXISTS public.namespace (
    prefix varchar(10) NOT NULL PRIMARY KEY,
    uri varchar NOT NULL,
    is_base bool DEFAULT false
);

CREATE UNIQUE INDEX namespace_uri_idx ON public.namespace (uri);

CREATE TABLE IF NOT EXISTS public.schema_def (
  id BIGINT PRIMARY KEY NOT NULL,
  prefix varchar(10) NOT NULL,
  term varchar NOT NULL,
  parent_id BIGINT,
  ancestors_id BIGINT[],
  data JSONB NOT NULL DEFAULT '{}',
  CONSTRAINT schema_def_id_fkey  FOREIGN KEY (id)
      REFERENCES public.topic (id) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT namespace_prefix_fkey  FOREIGN KEY (prefix)
      REFERENCES public.namespace (prefix) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT schema_def_parent_fkey FOREIGN KEY (parent_id)
      REFERENCES public.schema_def (id) ON DELETE SET NULL ON UPDATE CASCADE
);

CREATE UNIQUE INDEX schema_def_term_idx on schema_def (prefix, term);
CREATE INDEX schema_def_parent_idx on schema_def (parent_id);
CREATE INDEX schema_def_ancestors_idx on schema_def USING gin (ancestors_id);
CREATE INDEX schema_def_data_idx on schema_def USING gin (data jsonb_path_ops);

ALTER TABLE public.topic ADD CONSTRAINT topic_schema_def_fkey FOREIGN KEY (schema_def_id)
    REFERENCES public.schema_def (id) ON DELETE SET NULL ON UPDATE CASCADE;

COMMIT;
