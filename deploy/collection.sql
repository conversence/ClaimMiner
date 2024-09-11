-- Deploy collection
-- requires: user
-- requires: topic
-- Copyright Society Library and Conversence 2022-2024

BEGIN;


CREATE TABLE IF NOT EXISTS public.collection (
    id bigint NOT NULL DEFAULT nextval('public.topic_id_seq'::regclass) PRIMARY KEY,
    name varchar NOT NULL,
    params JSONB NOT NULL DEFAULT '{}'::JSONB
);

CREATE UNIQUE INDEX collection_name_idx on collection (name);

CREATE TABLE IF NOT EXISTS public.collection_permissions (
  user_id bigint NOT NULL,
  collection_id bigint NOT NULL,
  permissions public.permission[] DEFAULT ARRAY[]::public.permission[],
  CONSTRAINT pcollection_pkey PRIMARY KEY (user_id, collection_id),
  CONSTRAINT collection_permissions_user_id_key FOREIGN KEY (user_id)
    REFERENCES public.user (id) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT collection_permissions_collection_id_key FOREIGN KEY (collection_id)
    REFERENCES public.collection (id) ON DELETE CASCADE ON UPDATE CASCADE
);



CREATE TABLE IF NOT EXISTS public.topic_collection (
  topic_id bigint NOT NULL,
  collection_id bigint NOT NULL,
  CONSTRAINT topic_collection_pkey PRIMARY KEY (topic_id, collection_id),
  CONSTRAINT topic_collection_topic_id FOREIGN KEY (topic_id)
    REFERENCES topic (id) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT topic_collection_collection_id FOREIGN KEY (collection_id)
    REFERENCES collection (id) ON DELETE CASCADE ON UPDATE CASCADE
);

CREATE INDEX IF NOT EXISTS topic_collection_inv_idx ON public.topic_collection (collection_id, topic_id);

COMMIT;
