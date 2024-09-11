-- Deploy claim_link
-- Copyright Society Library and Conversence 2022-2024

BEGIN;

DROP INDEX IF EXISTS claim_link_source_idx;
DROP INDEX IF EXISTS claim_link_dest_idx;
DROP INDEX IF EXISTS claim_link_external_id_idx;
DROP TABLE IF EXISTS claim_link;

COMMIT;
