import { pool } from '../db/client.js';
import { embed } from '../db/embed.js';

export interface Skill {
  id: string;
  name: string;
  description: string;
  input_schema: Record<string, unknown>;
  output_schema: Record<string, unknown>;
  implementation_ref: string;
  health_status: 'healthy' | 'degraded' | 'unhealthy' | 'unknown';
  success_count: number;
  failure_count: number;
  avg_latency_ms: number | null;
  tags: string[];
  created_at: Date;
  updated_at: Date;
}

export interface RegisterSkillParams {
  name: string;
  description: string;
  input_schema?: Record<string, unknown>;
  output_schema?: Record<string, unknown>;
  implementation_ref: string;
  tags?: string[];
}

// ---------------------------------------------------------------------------
// Register a new skill or update an existing one (upsert by name).
// ---------------------------------------------------------------------------
export async function registerSkill(params: RegisterSkillParams): Promise<Skill> {
  const { rows: [skill] } = await pool.query<Skill>(`
    INSERT INTO skills (name, description, input_schema, output_schema, implementation_ref, tags)
    VALUES ($1, $2, $3, $4, $5, $6)
    ON CONFLICT (name) DO UPDATE SET
      description = EXCLUDED.description,
      input_schema = EXCLUDED.input_schema,
      output_schema = EXCLUDED.output_schema,
      implementation_ref = EXCLUDED.implementation_ref,
      tags = EXCLUDED.tags,
      updated_at = now()
    RETURNING *
  `, [
    params.name,
    params.description,
    JSON.stringify(params.input_schema ?? {}),
    JSON.stringify(params.output_schema ?? {}),
    params.implementation_ref,
    params.tags ?? [],
  ]);
  return skill;
}

// ---------------------------------------------------------------------------
// Find skills by semantic similarity to a query.
// Falls back to keyword search if embedding is unavailable.
// ---------------------------------------------------------------------------
export async function findRelevantSkills(
  query: string,
  opts: { limit?: number; minStatus?: Skill['health_status'] } = {},
): Promise<(Skill & { similarity?: number })[]> {
  const limit = opts.limit ?? 5;

  // Exclude unhealthy skills unless caller explicitly opts in
  const statusFilter = `health_status != 'unhealthy'`;

  try {
    const embedding = await embed(query);
    // Use pgvector on the description embedding stored inline (simple approach:
    // compute similarity in-process since descriptions are short)
    const { rows: skills } = await pool.query<Skill>(`
      SELECT * FROM skills
      WHERE ${statusFilter}
      ORDER BY updated_at DESC
      LIMIT ${limit * 4}
    `);

    // Compute similarity against descriptions via embed (cached in embed module)
    const scored = await Promise.all(
      skills.map(async (s) => {
        try {
          const descEmb = await embed(s.description);
          const sim = cosineSim(embedding, descEmb);
          return { ...s, similarity: sim };
        } catch {
          return { ...s, similarity: 0 };
        }
      }),
    );
    scored.sort((a, b) => (b.similarity ?? 0) - (a.similarity ?? 0));
    return scored.slice(0, limit);
  } catch {
    // Fallback: keyword overlap
    const { rows } = await pool.query<Skill>(`
      SELECT * FROM skills
      WHERE ${statusFilter}
        AND (description ILIKE $1 OR name ILIKE $1 OR $2 = ANY(tags))
      LIMIT $3
    `, [`%${query}%`, query.toLowerCase(), limit]);
    return rows;
  }
}

function cosineSim(a: number[], b: number[]): number {
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return dot;
}

// ---------------------------------------------------------------------------
// Retrieve a skill by name or id.
// ---------------------------------------------------------------------------
export async function getSkill(nameOrId: string): Promise<Skill | null> {
  const { rows } = await pool.query<Skill>(`
    SELECT * FROM skills WHERE name = $1 OR id::text = $1 LIMIT 1
  `, [nameOrId]);
  return rows[0] ?? null;
}

// ---------------------------------------------------------------------------
// List all skills with optional tag filter.
// ---------------------------------------------------------------------------
export async function listSkills(opts: {
  tags?: string[];
  status?: Skill['health_status'];
  limit?: number;
} = {}): Promise<Skill[]> {
  const conditions: string[] = [];
  const values: unknown[] = [];
  let idx = 1;

  if (opts.status) { conditions.push(`health_status = $${idx++}`); values.push(opts.status); }
  if (opts.tags?.length) { conditions.push(`tags && $${idx++}::text[]`); values.push(opts.tags); }

  values.push(opts.limit ?? 50);
  const where = conditions.length ? `WHERE ${conditions.join(' AND ')}` : '';
  const { rows } = await pool.query<Skill>(`
    SELECT * FROM skills ${where} ORDER BY name LIMIT $${idx}
  `, values);
  return rows;
}

// ---------------------------------------------------------------------------
// Record an execution outcome for a skill (updates stats).
// ---------------------------------------------------------------------------
export async function recordSkillExecution(
  skillId: string,
  opts: { success: boolean; latencyMs: number },
): Promise<void> {
  if (opts.success) {
    await pool.query(`
      UPDATE skills SET
        success_count = success_count + 1,
        avg_latency_ms = CASE
          WHEN avg_latency_ms IS NULL THEN $1
          ELSE (avg_latency_ms * success_count + $1) / (success_count + 1)
        END,
        updated_at = now()
      WHERE id = $2
    `, [opts.latencyMs, skillId]);
  } else {
    await pool.query(`
      UPDATE skills SET
        failure_count = failure_count + 1,
        updated_at = now()
      WHERE id = $2
    `, [skillId]);
  }
}
