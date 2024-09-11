-- Deploy user_functions
-- requires: user

BEGIN;

CREATE OR REPLACE FUNCTION public.before_create_user() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
    DECLARE num_mem integer;
    BEGIN
      SELECT count(id) INTO STRICT num_mem FROM public.user;
      IF num_mem <= 1 THEN
        -- give admin to first registered user (after system)
        NEW.permissions = ARRAY['admin'::permission];
        NEW.confirmed = true;
      END IF;
      RETURN NEW;
    END;
    $$;

DROP TRIGGER IF EXISTS before_create_user ON public.user;
CREATE TRIGGER before_create_user BEFORE INSERT ON public.user FOR EACH ROW EXECUTE FUNCTION public.before_create_user();

COMMIT;
