-- OpenClaw Memory Kernel v0.1 — PostgreSQL Schema
-- Requires: pgvector, Apache AGE
-- Run: psql $DATABASE_URL -f schema.sql

-- Extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Apache AGE (graph layer — requires AGE installed in PG)
-- See: https://age.apache.org/age-manual/master/intro/setup.html
CREATE EXTENSION IF NOT EXISTS age;
LOAD 'age';
SET search_path = ag_catalog, "$user", public;

-- ---------------------------------------------------------------------------
-- TEMPORAL MEMORY EVENTS
-- Every fact observed by or communicated to the agent is a memory_event.
-- valid_from / valid_until implement Graphiti-style temporal validity windows.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS memory_events (
  id            uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
  actor         text        NOT NULL,                -- "user" | "assistant" | "tool:<name>"
  source        text        NOT NULL,                -- "chat" | "agent_turn" | "skill_output" | "belief_revision"
  content       text        NOT NULL,
  event_time    timestamptz NOT NULL DEFAULT now(),
  valid_from    timestamptz,                         -- NULL = always valid from creation
  valid_until   timestamptz,                         -- NULL = no expiry
  provenance    jsonb       NOT NULL DEFAULT '{}',   -- {session_id, skill_id, belief_ids, ...}
  confidence    numeric     NOT NULL DEFAULT 0.7     CHECK (confidence BETWEEN 0 AND 1),
  created_at    timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_memory_events_event_time ON memory_events (event_time DESC);
CREATE INDEX IF NOT EXISTS idx_memory_events_actor      ON memory_events (actor);
CREATE INDEX IF NOT EXISTS idx_memory_events_valid      ON memory_events (valid_until) WHERE valid_until IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_memory_events_provenance ON memory_events USING gin (provenance);

-- ---------------------------------------------------------------------------
-- SEMANTIC EMBEDDINGS (pgvector)
-- Separate table so the heavy vector column stays out of the main event scan.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS memory_embeddings (
  id               uuid    PRIMARY KEY DEFAULT gen_random_uuid(),
  memory_event_id  uuid    NOT NULL REFERENCES memory_events(id) ON DELETE CASCADE,
  embedding        vector(1536) NOT NULL,            -- text-embedding-3-small
  created_at       timestamptz NOT NULL DEFAULT now()
);

-- HNSW index for fast approximate nearest-neighbour search
CREATE INDEX IF NOT EXISTS idx_memory_embeddings_hnsw
  ON memory_embeddings USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);

-- ---------------------------------------------------------------------------
-- SKILLS (Voyager-style capability library)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS skills (
  id                  uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
  name                text        UNIQUE NOT NULL,
  description         text        NOT NULL,
  input_schema        jsonb       NOT NULL DEFAULT '{}',
  output_schema       jsonb       NOT NULL DEFAULT '{}',
  implementation_ref  text        NOT NULL,          -- module path or function identifier
  health_status       text        NOT NULL DEFAULT 'unknown'
                                  CHECK (health_status IN ('healthy','degraded','unhealthy','unknown')),
  last_health_check   timestamptz,
  health_detail       text,
  success_count       int         NOT NULL DEFAULT 0,
  failure_count       int         NOT NULL DEFAULT 0,
  avg_latency_ms      numeric,
  tags                text[]      NOT NULL DEFAULT '{}',
  created_at          timestamptz NOT NULL DEFAULT now(),
  updated_at          timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_skills_health   ON skills (health_status);
CREATE INDEX IF NOT EXISTS idx_skills_tags     ON skills USING gin (tags);

-- ---------------------------------------------------------------------------
-- BELIEFS
-- Each belief is a claim with confidence, temporal scope, and status.
-- Subject / predicate / object allow structured triple extraction.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS beliefs (
  id          uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
  claim       text        NOT NULL,
  subject     text,                                  -- entity the claim is about
  predicate   text,                                  -- relationship type
  object      text,                                  -- entity or value
  confidence  numeric     NOT NULL DEFAULT 0.5       CHECK (confidence BETWEEN 0 AND 1),
  status      text        NOT NULL DEFAULT 'active'
              CHECK (status IN ('active','deprecated','contested','archived')),
  valid_from  timestamptz,
  valid_until timestamptz,
  created_at  timestamptz NOT NULL DEFAULT now(),
  updated_at  timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_beliefs_status          ON beliefs (status);
CREATE INDEX IF NOT EXISTS idx_beliefs_subject_pred    ON beliefs (subject, predicate);
CREATE INDEX IF NOT EXISTS idx_beliefs_valid           ON beliefs (valid_until) WHERE valid_until IS NOT NULL;

-- ---------------------------------------------------------------------------
-- JUSTIFICATIONS
-- Links beliefs to the memory events that support them.
-- Multiple justifications can support one belief (evidence accumulation).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS justifications (
  id               uuid    PRIMARY KEY DEFAULT gen_random_uuid(),
  belief_id        uuid    NOT NULL REFERENCES beliefs(id) ON DELETE CASCADE,
  memory_event_id  uuid    REFERENCES memory_events(id) ON DELETE SET NULL,
  support_type     text    NOT NULL                   -- "direct_statement" | "inference" | "correction" | "contradiction"
                           CHECK (support_type IN ('direct_statement','inference','correction','contradiction','reinforcement')),
  weight           numeric NOT NULL DEFAULT 1.0       CHECK (weight > 0),
  explanation      text,
  created_at       timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_justifications_belief  ON justifications (belief_id);
CREATE INDEX IF NOT EXISTS idx_justifications_event   ON justifications (memory_event_id);

-- ---------------------------------------------------------------------------
-- AGE GRAPH SETUP
-- The graph mirrors key nodes and relationships from the relational schema
-- for multi-hop traversal queries.
-- ---------------------------------------------------------------------------
SELECT * FROM ag_catalog.create_graph('openclaw') WHERE NOT EXISTS (
  SELECT 1 FROM ag_catalog.ag_graph WHERE name = 'openclaw'
);

-- ---------------------------------------------------------------------------
-- HELPER FUNCTIONS
-- ---------------------------------------------------------------------------

-- Auto-update updated_at
CREATE OR REPLACE FUNCTION touch_updated_at()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN NEW.updated_at = now(); RETURN NEW; END;
$$;

CREATE OR REPLACE TRIGGER beliefs_updated_at
  BEFORE UPDATE ON beliefs
  FOR EACH ROW EXECUTE FUNCTION touch_updated_at();

CREATE OR REPLACE TRIGGER skills_updated_at
  BEFORE UPDATE ON skills
  FOR EACH ROW EXECUTE FUNCTION touch_updated_at();
