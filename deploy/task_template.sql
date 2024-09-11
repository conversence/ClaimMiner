-- Deploy task_template
-- requires: analyzer
-- requires: collection

BEGIN;

CREATE TABLE IF NOT EXISTS public.task_template (
    id bigint NOT NULL DEFAULT nextval('public.topic_id_seq'::regclass) PRIMARY KEY,
    analyzer_id bigint NOT NULL,
    nickname character varying(100),
    params JSONB DEFAULT '{}',
    draft BOOLEAN NOT NULL DEFAULT false,
    collection_id BIGINT,
    CONSTRAINT task_template_analyzer_id_fkey FOREIGN KEY (analyzer_id)
        REFERENCES public.analyzer (id) ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT task_template_collection_id_fkey FOREIGN KEY (collection_id)
        REFERENCES public.collection (id) ON DELETE CASCADE ON UPDATE CASCADE
);

CREATE UNIQUE INDEX IF NOT EXISTS task_template_unique_params_idx ON public.task_template (analyzer_id, params);
CREATE UNIQUE INDEX IF NOT EXISTS task_template_unique_name_idx ON public.task_template (nickname);

COMMIT;
