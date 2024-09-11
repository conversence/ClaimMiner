-- Deploy fragment
-- requires: document
-- requires: topic
-- requires: base_functions
-- Copyright Society Library and Conversence 2022-2024

BEGIN;

CREATE TABLE IF NOT EXISTS public.fragment (
    id bigint NOT NULL,
    doc_id bigint,
    part_of bigint,
    position integer,
    char_position integer,
    scale public.fragment_type NOT NULL,
    language character varying(16) NOT NULL,
    text text NOT NULL,
    created_by bigint,
    generation_data JSONB,
    confirmed boolean NOT NULL default true,

    CONSTRAINT fragment_pkey PRIMARY KEY (id),
    CONSTRAINT fragment_id_fkey  FOREIGN KEY (id)
      REFERENCES public.topic (id) ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT fragment_doc_id_key FOREIGN KEY (doc_id)
      REFERENCES public.document (id) ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT fragment_part_of_key FOREIGN KEY (part_of)
      REFERENCES public.fragment (id) ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT fragment_created_by_key FOREIGN KEY (created_by)
      REFERENCES public.user (id) ON DELETE SET NULL ON UPDATE CASCADE
);

CREATE INDEX fragment_doc_id_idx on fragment (doc_id);
CREATE INDEX fragment_part_of_idx on fragment (part_of);
CREATE INDEX fragment_text_hash_idx on fragment using hash (text);  -- for identity
CREATE INDEX IF NOT EXISTS fragment_text_prefix_idx on fragment (substr(text, 0, 64)) WHERE scale >= 'standalone';  -- for alphabetical sort of claims
-- CREATE UNIQUE INDEX fragment_text_hash_u_idx on fragment (text, coalesce(external_id, '')) WHERE doc_id IS NULL;
-- TODO: NULLS NOT DISTINCT
-- CREATE INDEX fragment_text_idx on fragment (to_tsvector(text)) using gin;
CREATE INDEX fragment_text_en_idx on fragment using gin (to_tsvector('english', text)) WHERE starts_with(language, 'en');

COMMIT;
