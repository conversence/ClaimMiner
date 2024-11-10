-- Deploy claim_link_functions
-- requires: claim_link
-- idempotent

BEGIN;

CREATE OR REPLACE FUNCTION public.before_update_claim_link() RETURNS trigger LANGUAGE plpgsql
  AS $$
BEGIN
  NEW.modified_at = now();
  return NEW;
END;
$$;

DROP TRIGGER IF EXISTS before_update_claim_link ON claim_link;
CREATE TRIGGER before_update_claim_link BEFORE UPDATE ON public.claim_link FOR EACH ROW EXECUTE FUNCTION public.before_update_claim_link();

CREATE OR REPLACE FUNCTION public.after_delete_claim_link() RETURNS trigger LANGUAGE plpgsql
  AS $$
BEGIN
  DELETE FROM public.topic AS t WHERE t.id=OLD.id AND "type" = 'link';
  return NEW;
END;
$$;

DROP TRIGGER IF EXISTS after_delete_claim_link ON claim_link;
CREATE TRIGGER after_delete_claim_link AFTER DELETE ON public.claim_link FOR EACH ROW EXECUTE FUNCTION public.after_delete_claim_link();

COMMIT;
