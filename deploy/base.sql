-- Deploy base
-- requires: admin_base
-- Copyright Society Library and Conversence 2022-2024

BEGIN;

CREATE SEQUENCE IF NOT EXISTS public.topic_id_seq
    AS bigint
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;

CREATE TYPE public.topic_type AS ENUM (
    'fragment',
    'standalone',
    'link',
    'hyperedge',
    'document',
    'analyzer',
    'analysis',
    'collection',
    'agent',
    'cluster',
    'ontology',
    'schema_term',
    'structured_idea'
);


CREATE TYPE public.permission AS ENUM (
    'admin',
    'access',
    'add_document',
    'add_claim',
    'claim_score_query',
    'bigdata_query',
    'openai_query',
    'confirm_claim',
    'edit_prompts'
);

CREATE TYPE public.link_type AS ENUM (
    'freeform',
    'key_point',
    'supported_by',
    'opposed_by',
    'implied',
    'implicit',
    'derived',
    'has_premise',
    'answers_question',
    'irrelevant',
    'relevant',
    'subcategory',
    'subclaim',
    'subquestion',
    'quote'
);

CREATE TYPE public.process_status AS ENUM (
    'inapplicable',
    'not_ready',
    'not_requested',
    'pending',
    'ongoing',
    'complete',
    'error'
);


CREATE TYPE public.fragment_type AS ENUM (
  'document',
  'paragraph',
  'sentence',
  'phrase',
  'quote',
  'summary',
  'reified_arg_link',
  'standalone',
  'generated',
  'standalone_root',
  'standalone_category',
  'standalone_question',
  'standalone_claim',
  'standalone_argument'
);

CREATE TYPE public.relevance AS ENUM (
    'irrelevant',
    'unknown',
    'relevant'
);


CREATE TYPE public.id_score_type AS (
    id bigint, score float
);

COMMIT;
