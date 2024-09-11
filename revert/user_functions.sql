-- Deploy user_functions


BEGIN;

DROP TRIGGER IF EXISTS before_create_user ON public.user;
DROP FUNCTION IF EXISTS public.before_create_user();

COMMIT;
