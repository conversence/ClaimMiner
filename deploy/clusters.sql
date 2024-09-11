-- Deploy clusters
-- requires: fragment
-- requires: analysis

BEGIN;

CREATE TABLE cluster_data (
  id BIGINT NOT NULL PRIMARY KEY DEFAULT nextval('public.topic_id_seq'::regclass),
  analysis_id BIGINT NOT NULL,
  cluster_size SMALLINT NOT NULL DEFAULT 1,
  distinguished_claim_id BIGINT,
  auto_include_diameter float,
  relevant public.relevance DEFAULT 'unknown',
  relevance_checker_id BIGINT,
  CONSTRAINT cluster_data_distinguished_fkey FOREIGN KEY (distinguished_claim_id)
    REFERENCES public.fragment(id) ON DELETE SET NULL ON UPDATE CASCADE,
  CONSTRAINT cluster_data_relevance_checker_fkey FOREIGN KEY (relevance_checker_id)
    REFERENCES public.user(id) ON DELETE SET NULL ON UPDATE CASCADE,
  CONSTRAINT cluster_data_analysis_id_fkey FOREIGN KEY (analysis_id)
    REFERENCES public.analysis(id) ON DELETE CASCADE ON UPDATE CASCADE
);

CREATE TABLE in_cluster_data (
  cluster_id BIGINT NOT NULL,
  fragment_id BIGINT NOT NULL,
  confirmed_by_id BIGINT,
  manual boolean DEFAULT false,

  CONSTRAINT in_cluster_data_pkey PRIMARY KEY (cluster_id, fragment_id),
  CONSTRAINT in_cluster_data_cluster_fkey FOREIGN KEY (cluster_id)
    REFERENCES public.cluster_data(id) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT in_cluster_data_fragment_fkey FOREIGN KEY (fragment_id)
    REFERENCES public.fragment(id) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT in_cluster_data_confirmed_by_fkey FOREIGN KEY (confirmed_by_id)
    REFERENCES public.user(id) ON DELETE SET NULL ON UPDATE CASCADE
);

COMMIT;
