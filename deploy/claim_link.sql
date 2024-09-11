-- Deploy claim_link
-- requires: fragment
-- Copyright Society Library and Conversence 2022-2024

BEGIN;

CREATE TABLE IF NOT EXISTS public.claim_link (
  id BIGINT PRIMARY KEY NOT NULL,
  source BIGINT NOT NULL,
  target BIGINT NOT NULL,
  link_type public.link_type NOT NULL,
  analyzer BIGINT,
  created_by BIGINT,
  score REAL,
  CONSTRAINT claim_link_id_fkey  FOREIGN KEY (id)
      REFERENCES public.topic (id) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT claim_link_source_fkey FOREIGN KEY (source)
    REFERENCES public.topic(id) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT claim_link_dest_fkey FOREIGN KEY (target)
    REFERENCES public.topic(id) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT claim_link_analyzer_fkey FOREIGN KEY (analyzer)
    REFERENCES public.analyzer(id) ON DELETE SET NULL ON UPDATE CASCADE,
  CONSTRAINT claim_link_created_by_fkey FOREIGN KEY (created_by)
    REFERENCES public.user(id) ON DELETE SET NULL ON UPDATE CASCADE
);

CREATE INDEX claim_link_source_idx on claim_link (source);
CREATE INDEX claim_link_dest_idx on claim_link (target);

COMMIT;
