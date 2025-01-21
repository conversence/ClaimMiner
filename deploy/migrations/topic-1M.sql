-- Deploy topic


BEGIN;

ALTER TABLE topic ADD COLUMN schema_term_id BIGINT;

CREATE INDEX IF NOT EXISTS topic_schema_term_idx ON topic (schema_term_id);

COMMIT;
