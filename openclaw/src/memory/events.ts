import { pool, withClient } from '../db/client.js';
import { embed } from '../db/embed.js';

export interface MemoryEvent {
  id: string;
  actor: string;
  source: string;
  content: string;
  event_time: Date;
  valid_from: Date | null;
  valid_until: Date | null;
  provenance: Record<string, unknown>;
  confidence: number;
  created_at: Date;
}

export interface WriteMemoryEventParams {
  actor: string;
  source: string;
  content: string;
  confidence?: number;
  valid_from?: Date;
  valid_until?: Date;
  provenance?: Record<string, unknown>;
  event_time?: Date;
}

// ---------------------------------------------------------------------------
// Write a memory event and its embedding in a single transaction.
// Returns the persisted event.
// ---------------------------------------------------------------------------
export async function writeMemoryEvent(
  params: WriteMemoryEventParams,
): Promise<MemoryEvent> {
  return withClient(async (client) => {
    await client.query('BEGIN');
    try {
      const { rows: [event] } = await client.query<MemoryEvent>(`
        INSERT INTO memory_events
          (actor, source, content, confidence, valid_from, valid_until, provenance, event_time)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        RETURNING *
      `, [
        params.actor,
        params.source,
        params.content,
        params.confidence ?? 0.7,
        params.valid_from ?? null,
        params.valid_until ?? null,
        JSON.stringify(params.provenance ?? {}),
        params.event_time ?? new Date(),
      ]);

      const embedding = await embed(params.content);
      await client.query(`
        INSERT INTO memory_embeddings (memory_event_id, embedding)
        VALUES ($1, $2)
      `, [event.id, JSON.stringify(embedding)]);

      await client.query('COMMIT');
      return event;
    } catch (err) {
      await client.query('ROLLBACK');
      throw err;
    }
  });
}

// ---------------------------------------------------------------------------
// Read a single event by id.
// ---------------------------------------------------------------------------
export async function getMemoryEvent(id: string): Promise<MemoryEvent | null> {
  const { rows } = await pool.query<MemoryEvent>(
    'SELECT * FROM memory_events WHERE id = $1',
    [id],
  );
  return rows[0] ?? null;
}

// ---------------------------------------------------------------------------
// Mark an event as expired (sets valid_until = now() if currently open).
// Used when a fact is superseded.
// ---------------------------------------------------------------------------
export async function expireMemoryEvent(id: string): Promise<void> {
  await pool.query(`
    UPDATE memory_events
    SET valid_until = now()
    WHERE id = $1 AND (valid_until IS NULL OR valid_until > now())
  `, [id]);
}

// ---------------------------------------------------------------------------
// Fetch the N most recent events for an actor within an optional time window.
// ---------------------------------------------------------------------------
export async function recentEvents(opts: {
  actor?: string;
  source?: string;
  limit?: number;
  afterHours?: number;
}): Promise<MemoryEvent[]> {
  const conditions: string[] = ['(valid_until IS NULL OR valid_until > now())'];
  const values: unknown[] = [];
  let idx = 1;

  if (opts.actor) { conditions.push(`actor = $${idx++}`); values.push(opts.actor); }
  if (opts.source) { conditions.push(`source = $${idx++}`); values.push(opts.source); }
  if (opts.afterHours) {
    conditions.push(`event_time > now() - ($${idx++} || ' hours')::interval`);
    values.push(opts.afterHours);
  }

  values.push(opts.limit ?? 20);
  const { rows } = await pool.query<MemoryEvent>(`
    SELECT * FROM memory_events
    WHERE ${conditions.join(' AND ')}
    ORDER BY event_time DESC
    LIMIT $${idx}
  `, values);
  return rows;
}
