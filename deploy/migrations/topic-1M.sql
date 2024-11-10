-- Deploy topic


BEGIN;

ALTER TABLE topic ADD COLUMN schema_def_id BIGINT;

CREATE INDEX IF NOT EXISTS topic_schema_def_idx ON topic (schema_def_id);

COMMIT;
