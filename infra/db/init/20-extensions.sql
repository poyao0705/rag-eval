-- The official ParadeDB image creates vector and pg_search in its bootstrap
-- script. Keep project-specific extensions here, after that bootstrap runs.
CREATE EXTENSION IF NOT EXISTS unaccent;
