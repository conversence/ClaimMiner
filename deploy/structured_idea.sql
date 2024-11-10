-- Deploy structured_idea
-- requires: schema_def
-- copyright Conversence 2023
-- license: Apache 2.0

BEGIN;

CREATE OR REPLACE FUNCTION extract_references(refs JSONB) RETURNS BIGINT[] LANGUAGE SQL IMMUTABLE AS $$
  SELECT array_agg(i::bigint) FROM (SELECT jsonb_array_elements(jsonb_path_query_array(refs, '$.*[*]')) i);
$$;


CREATE TABLE IF NOT EXISTS public.structured_idea (
  id BIGINT PRIMARY KEY NOT NULL,
  ref_structure JSONB NOT NULL DEFAULT '{}',
  literal_structure JSONB NOT NULL DEFAULT '{}',
  modified_at timestamp without time zone NOT NULL DEFAULT now(),
  refs BIGINT[] GENERATED ALWAYS AS (extract_references(ref_structure)) STORED,
  CONSTRAINT structured_idea_id_fkey  FOREIGN KEY (id)
      REFERENCES public.topic (id) ON DELETE CASCADE ON UPDATE CASCADE
);

CREATE INDEX structured_idea_refs_idx on structured_idea USING gin (refs);
CREATE INDEX structured_idea_data_idx on structured_idea USING gin ((literal_structure || ref_structure) jsonb_path_ops);

COMMIT;
