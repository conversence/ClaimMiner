-- Deploy analysis_functions
-- requires analysis

BEGIN;

CREATE OR REPLACE FUNCTION public.after_create_update_analysis() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
    BEGIN
      IF NEW.status != OLD.status THEN
        PERFORM pg_notify(current_database(), concat('analysis ' , NEW.id, ' ', NEW.analyzer_id, ' ', NEW.status, ' ', NEW.target_id, ' ', NEW.theme_id));
      END IF;
      RETURN NEW;
    END;
    $$;

DROP TRIGGER IF EXISTS after_update_analysis ON public.analysis;
CREATE TRIGGER after_update_analysis after UPDATE ON public.analysis FOR EACH ROW EXECUTE FUNCTION public.after_create_update_analysis();
DROP TRIGGER IF EXISTS after_create_analysis ON public.analysis;
CREATE TRIGGER after_create_analysis after INSERT ON public.analysis FOR EACH ROW EXECUTE FUNCTION public.after_create_update_analysis();

CREATE OR REPLACE FUNCTION public.after_create_analysis_output() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
    BEGIN
      PERFORM pg_notify(current_database(), concat('analysis_output ' , NEW.analysis_id, ' ', NEW.topic_id));
      RETURN NEW;
    END;
    $$;

DROP TRIGGER IF EXISTS after_create_analysis_output ON public.analysis;
CREATE TRIGGER after_create_analysis_output after INSERT ON public.analysis_output FOR EACH ROW EXECUTE FUNCTION public.after_create_analysis_output();

COMMIT;
