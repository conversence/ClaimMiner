-- Deploy document_link
-- requires: document

BEGIN;

CREATE TABLE IF NOT EXISTS public.document_link (
  source_id BIGINT NOT NULL,
  target_id BIGINT NOT NULL,
  analyzer_id BIGINT,
  CONSTRAINT document_link_pkey PRIMARY KEY (source_id, target_id),
  CONSTRAINT document_link_source_fkey FOREIGN KEY (source_id)
    REFERENCES public.document(id) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT document_link_target_fkey FOREIGN KEY (target_id)
    REFERENCES public.uri_equiv(id) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT document_link_analyzer_fkey FOREIGN KEY (analyzer_id)
    REFERENCES public.analyzer(id) ON DELETE SET NULL ON UPDATE CASCADE
);

CREATE INDEX document_link_target_idx on document_link (target_id);

COMMIT;
