-- Deploy structured_idea_functions


BEGIN;

DROP FUNCTION IF EXISTS public.structure_neighbourhood_cl_filtered_terms(target_id BIGINT, structure_terms varchar[]);
DROP FUNCTION IF EXISTS public.structure_neighbourhood_cl_filtered_ids(target_id BIGINT, structure_types BIGINT[]);
DROP FUNCTION IF EXISTS public.structure_neighbourhood_cl(target_id BIGINT);
DROP FUNCTION IF EXISTS public.sub_structures_cl_rec(target_id BIGINT);
DROP FUNCTION IF EXISTS public.sub_structures(target_id BIGINT);
DROP FUNCTION IF EXISTS public.in_structures_cl_rec_filtered_terms(target_id BIGINT, structure_terms varchar[]);
DROP FUNCTION IF EXISTS public.in_structures_cl_rec_filtered_ids(target_id BIGINT, structure_types BIGINT[]);
DROP FUNCTION IF EXISTS public.in_structures_cl_filtered_terms(target_id BIGINT, structure_terms varchar[]);
DROP FUNCTION IF EXISTS public.in_structures_cl_filtered_ids(target_id BIGINT, structure_types BIGINT[]);
DROP FUNCTION IF EXISTS public.in_structures_cl_rec(target_id BIGINT);
DROP FUNCTION IF EXISTS public.in_structures_cl(target_id BIGINT);
DROP FUNCTION IF EXISTS public.structure_neighbourhood_filtered_terms(target_id BIGINT, structure_terms varchar[]);
DROP FUNCTION IF EXISTS public.structure_neighbourhood_filtered_ids(target_id BIGINT, structure_types BIGINT[]);
DROP FUNCTION IF EXISTS public.structure_neighbourhood(target_id BIGINT);
DROP FUNCTION IF EXISTS public.sub_structures_rec(target_id BIGINT);
DROP FUNCTION IF EXISTS public.in_structures_rec_filtered_terms(target_id BIGINT, structure_terms varchar[]);
DROP FUNCTION IF EXISTS public.in_structures_rec_filtered_ids(target_id BIGINT, structure_types BIGINT[]);
DROP FUNCTION IF EXISTS public.in_structures_rec(target_id BIGINT);

COMMIT;
