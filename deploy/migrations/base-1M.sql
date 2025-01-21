-- Deploy base


BEGIN;

ALTER TYPE topic_type ADD VALUE 'ontology';
ALTER TYPE topic_type ADD VALUE 'schema_term';
ALTER TYPE topic_type ADD VALUE 'structured_idea';

COMMIT;
