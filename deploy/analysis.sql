-- Deploy analysis
-- requires: analyzer
-- requires: fragment
-- requires: task_template
-- Copyright Society Library and Conversence 2022-2024

BEGIN;

CREATE TABLE IF NOT EXISTS public.analysis (
    id bigint NOT NULL DEFAULT nextval('public.topic_id_seq'::regclass),
    collection_id bigint,
    analyzer_id bigint NOT NULL,
    task_template_id bigint,
    part_of_id bigint,
    target_id bigint,
    theme_id bigint,
    params JSONB DEFAULT '{}',
    created timestamp without time zone NOT NULL DEFAULT now(),
    completed timestamp without time zone,
    results JSONB NOT NULL,
    creator_id BIGINT,
    triggered_by_analysis_id BIGINT,
    status public.process_status DEFAULT 'complete',
    CONSTRAINT analysis_pkey PRIMARY KEY (id),
    CONSTRAINT analysis_analyzer_id_fkey FOREIGN KEY (analyzer_id)
      REFERENCES public.analyzer (id) ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT analysis_task_template_id_fkey FOREIGN KEY (task_template_id)
      REFERENCES public.task_template (id) ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT analysis_collection_id_fkey FOREIGN KEY (collection_id)
      REFERENCES public.collection (id) ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT analysis_part_of_id_fkey FOREIGN KEY (part_of_id)
      REFERENCES public.analysis (id) ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT analysis_triggered_by_analysis_fkey FOREIGN KEY (triggered_by_analysis_id)
      REFERENCES public.analysis(id) ON DELETE SET NULL ON UPDATE CASCADE,
    CONSTRAINT analysis_target_id_fkey FOREIGN KEY (target_id)
      REFERENCES public.topic (id) ON DELETE SET NULL ON UPDATE CASCADE,
    CONSTRAINT analysis_theme_id_fkey FOREIGN KEY (theme_id)
      REFERENCES public.fragment (id) ON DELETE SET NULL ON UPDATE CASCADE,
    CONSTRAINT analysis_creator_id_fkey FOREIGN KEY (creator_id)
      REFERENCES public.user (id) ON DELETE SET NULL ON UPDATE CASCADE
);

CREATE INDEX IF NOT EXISTS analysis_theme_idx ON analysis (theme_id);
CREATE INDEX IF NOT EXISTS analysis_target_idx ON analysis (target_id);
CREATE UNIQUE INDEX IF NOT EXISTS analysis_unique_idx ON analysis (analyzer_id, target_id, collection_id, theme_id, part_of_id, task_template_id, params);

CREATE TABLE IF NOT EXISTS public.analysis_context (
  analysis_id bigint NOT NULL,
  fragment_id bigint NOT NULL,

  CONSTRAINT analysis_context_pkey PRIMARY KEY (fragment_id, analysis_id),
  CONSTRAINT analysis_context_analysis_id_fkey FOREIGN KEY (analysis_id)
    REFERENCES public.analysis (id) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT analysis_context_fragment_id_fkey FOREIGN KEY (fragment_id)
    REFERENCES public.fragment (id) ON DELETE CASCADE ON UPDATE CASCADE
);


CREATE INDEX IF NOT EXISTS analysis_context_analysis_idx ON analysis_context (analysis_id);

CREATE TABLE IF NOT EXISTS public.analysis_output (
  topic_id bigint NOT NULL,
  analysis_id bigint NOT NULL,

  CONSTRAINT analysis_output_pkey PRIMARY KEY (topic_id, analysis_id),
  CONSTRAINT analysis_output_analysis_id_fkey FOREIGN KEY (analysis_id)
    REFERENCES public.analysis (id) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT analysis_output_topic_id_fkey FOREIGN KEY (topic_id)
    REFERENCES public.topic (id) ON DELETE CASCADE ON UPDATE CASCADE
);

CREATE INDEX IF NOT EXISTS analysis_output_analysis_idx ON analysis_output (analysis_id);

COMMIT;
