-- Deploy claim_link_functions


BEGIN;

DROP TRIGGER IF EXISTS before_update_claim_link ON claim_link;
DROP TRIGGER IF EXISTS after_delete_claim_link ON claim_link;

DROP FUNCTION IF EXISTS public.before_update_claim_link();
DROP FUNCTION IF EXISTS public.after_delete_claim_link();

COMMIT;
