-- Deploy collection
-- Copyright Society Library and Conversence 2022-2024

BEGIN;

drop function IF EXISTS mmr(vector, varchar, varchar[], integer, double precision, integer);
drop function IF EXISTS mmr(vector, integer, varchar, varchar[], integer, double precision, integer);
drop function IF EXISTS mmr(vector, bigint, varchar, varchar[], integer, double precision, integer);

COMMIT;
