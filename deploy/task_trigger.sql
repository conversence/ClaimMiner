-- Deploy task_trigger
-- requires: task_template
-- requires: collection

BEGIN;

-- CREATE OR REPLACE FUNCTION topic_type_id(ttval topic_type) RETURNS BIGINT IMMUTABLE LANGUAGE sql AS $$
--   SELECT CAST(pg_enum.oid AS BIGINT) AS id from pg_enum join pg_type on enumtypid=pg_type.oid where pg_type.typname='topic_type' and enumlabel = CAST(ttval AS varchar);
-- $$;

CREATE TABLE IF NOT EXISTS public.task_trigger (
  id INTEGER PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
  target_analyzer_id BIGINT NOT NULL,
  task_template_id BIGINT,
  collection_id BIGINT,
  analyzer_trigger_id BIGINT,
  creation_trigger_id topic_type,
  automatic boolean DEFAULT false,
  conditions JSONB DEFAULT '{}',
  params JSONB DEFAULT '{}',
  creator_id BIGINT,
  -- collection_idk BIGINT GENERATED ALWAYS AS (COALESCE(collection_id, 0)) STORED,
  -- trigger_idk BIGINT GENERATED ALWAYS AS (COALESCE(analyzer_trigger_id, topic_type_id(creation_trigger_id))) STORED,

  CONSTRAINT task_trigger_analyzer_fkey FOREIGN KEY (target_analyzer_id)
    REFERENCES public.analyzer (id) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT task_trigger_trigger_fkey FOREIGN KEY (analyzer_trigger_id)
    REFERENCES public.analyzer (id) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT task_trigger_task_template_fkey FOREIGN KEY (task_template_id)
    REFERENCES public.task_template (id) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT task_trigger_collection_fkey FOREIGN KEY (collection_id)
    REFERENCES public.collection (id) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT task_trigger_creator_fkey FOREIGN KEY (creator_id)
    REFERENCES public.user (id) ON DELETE SET NULL ON UPDATE CASCADE
);

CREATE UNIQUE INDEX task_trigger_unique_analyzer_idx ON task_trigger (analyzer_trigger_id, target_analyzer_id, task_template_id, collection_id, conditions) WHERE analyzer_trigger_id IS NOT NULL;
CREATE UNIQUE INDEX task_trigger_unique_creation_idx ON task_trigger (creation_trigger_id, target_analyzer_id, task_template_id, collection_id, conditions) WHERE creation_trigger_id IS NOT NULL;

-- TODO: Constraint so exactly one of analyzer_trigger_id or creation_trigger_id is null.

COMMIT;
