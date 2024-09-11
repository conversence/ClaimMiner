-- Deploy embedding
-- requires: fragment
-- Copyright Society Library and Conversence 2022-2024

BEGIN;

CREATE TYPE public.embedding_model AS ENUM (
  'universal_sentence_encoder_4'
);

COMMIT;
