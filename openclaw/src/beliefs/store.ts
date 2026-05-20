import { pool } from '../db/client.js';

export interface Belief {
  id: string;
  claim: string;
  subject: string | null;
  predicate: string | null;
  object: string | null;
  confidence: number;
  status: 'active' | 'deprecated' | 'contested' | 'archived';
  valid_from: Date | null;
  valid_until: Date | null;
  created_at: Date;
  updated_at: Date;
}

export interface Justification {
  id: string;
  belief_id: string;
  memory_event_id: string | null;
  support_type: string;
  weight: number;
  explanation: string | null;
  created_at: Date;
}

export interface CreateBeliefParams {
  claim: string;
  subject?: string;
  predicate?: string;
  object?: string;
  confidence?: number;
  valid_from?: Date;
  valid_until?: Date;
}

// ---------------------------------------------------------------------------
// Create a new belief (claim + optional triple decomposition).
// ---------------------------------------------------------------------------
export async function createBelief(
  params: CreateBeliefParams,
): Promise<Belief> {
  const { rows: [belief] } = await pool.query<Belief>(`
    INSERT INTO beliefs (claim, subject, predicate, object, confidence, valid_from, valid_until)
    VALUES ($1, $2, $3, $4, $5, $6, $7)
    RETURNING *
  `, [
    params.claim,
    params.subject ?? null,
    params.predicate ?? null,
    params.object ?? null,
    params.confidence ?? 0.5,
    params.valid_from ?? null,
    params.valid_until ?? null,
  ]);
  return belief;
}

// ---------------------------------------------------------------------------
// Attach evidence (a memory event) to an existing belief.
// ---------------------------------------------------------------------------
export async function addJustification(params: {
  belief_id: string;
  memory_event_id?: string;
  support_type: Justification['support_type'];
  weight?: number;
  explanation?: string;
}): Promise<Justification> {
  const { rows: [j] } = await pool.query<Justification>(`
    INSERT INTO justifications (belief_id, memory_event_id, support_type, weight, explanation)
    VALUES ($1, $2, $3, $4, $5)
    RETURNING *
  `, [
    params.belief_id,
    params.memory_event_id ?? null,
    params.support_type,
    params.weight ?? 1.0,
    params.explanation ?? null,
  ]);
  return j;
}

// ---------------------------------------------------------------------------
// Update confidence (clamped to [0,1]).
// Returns the updated belief.
// ---------------------------------------------------------------------------
export async function adjustConfidence(
  beliefId: string,
  delta: number,
): Promise<Belief> {
  const { rows: [belief] } = await pool.query<Belief>(`
    UPDATE beliefs
    SET confidence = GREATEST(0, LEAST(1, confidence + $1)),
        status = CASE
          WHEN GREATEST(0, LEAST(1, confidence + $1)) < 0.2 THEN 'contested'
          WHEN GREATEST(0, LEAST(1, confidence + $1)) > 0.9 THEN 'active'
          ELSE status
        END
    WHERE id = $2
    RETURNING *
  `, [delta, beliefId]);
  return belief;
}

// ---------------------------------------------------------------------------
// Deprecate a belief (e.g. a correction supersedes it).
// The old belief is preserved for provenance; a new one replaces it.
// ---------------------------------------------------------------------------
export async function deprecateBelief(
  beliefId: string,
  reason?: string,
): Promise<void> {
  await pool.query(`
    UPDATE beliefs
    SET status = 'deprecated', valid_until = now(),
        updated_at = now()
    WHERE id = $1
  `, [beliefId]);

  if (reason) {
    await pool.query(`
      INSERT INTO justifications (belief_id, support_type, weight, explanation)
      VALUES ($1, 'correction', 1.0, $2)
    `, [beliefId, reason]);
  }
}

// ---------------------------------------------------------------------------
// Find active beliefs about a subject/predicate pair.
// ---------------------------------------------------------------------------
export async function findBeliefs(opts: {
  subject?: string;
  predicate?: string;
  status?: Belief['status'];
  minConfidence?: number;
  limit?: number;
}): Promise<Belief[]> {
  const conditions = ['status = $1'];
  const values: unknown[] = [opts.status ?? 'active'];
  let idx = 2;

  if (opts.subject)       { conditions.push(`subject = $${idx++}`);   values.push(opts.subject); }
  if (opts.predicate)     { conditions.push(`predicate = $${idx++}`); values.push(opts.predicate); }
  if (opts.minConfidence) { conditions.push(`confidence >= $${idx++}`); values.push(opts.minConfidence); }

  values.push(opts.limit ?? 20);
  const { rows } = await pool.query<Belief>(`
    SELECT * FROM beliefs
    WHERE ${conditions.join(' AND ')}
      AND (valid_until IS NULL OR valid_until > now())
    ORDER BY confidence DESC
    LIMIT $${idx}
  `, values);
  return rows;
}

// ---------------------------------------------------------------------------
// Get all justifications for a belief with their linked event summaries.
// ---------------------------------------------------------------------------
export async function getJustifications(
  beliefId: string,
): Promise<(Justification & { event_content?: string })[]> {
  const { rows } = await pool.query(`
    SELECT j.*, me.content AS event_content
    FROM justifications j
    LEFT JOIN memory_events me ON j.memory_event_id = me.id
    WHERE j.belief_id = $1
    ORDER BY j.weight DESC, j.created_at DESC
  `, [beliefId]);
  return rows;
}
