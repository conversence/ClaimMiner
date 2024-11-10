-- Deploy base


BEGIN;

ALTER TYPE topic_type ADD VALUE 'schema_def';
ALTER TYPE topic_type ADD VALUE 'structured_idea';

COMMIT;
