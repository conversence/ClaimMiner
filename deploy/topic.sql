-- Deploy topic
-- requires: base

BEGIN;

CREATE TABLE IF NOT EXISTS public.topic (
  id BIGINT NOT NULL PRIMARY KEY DEFAULT nextval('public.topic_id_seq'::regclass),
  "type" public.topic_type NOT NULL,
  created_by BIGINT,
  CONSTRAINT topic_created_by_fkey FOREIGN KEY (created_by)
    REFERENCES public.topic (id) ON DELETE SET NULL ON UPDATE CASCADE
);

COMMIT;
