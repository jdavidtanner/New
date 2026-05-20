import { pool } from '../db/client.js';
import type { MemoryEvent } from './events.js';

const DEFAULT_WINDOW_HOURS = parseInt(
  process.env.TEMPORAL_WINDOW_HOURS ?? '168',
);

export interface TemporalQuery {
  // Time window (defaults to TEMPORAL_WINDOW_HOURS env)
  windowHours?: number;
  // Optionally pin to a specific moment in time (what was valid then?)
  atTime?: Date;
  limit?: number;
  actors?: string[];
}

// ---------------------------------------------------------------------------
// Retrieve events that are currently valid within the time window.
// "Currently valid" means: valid_from <= now AND (valid_until IS NULL OR valid_until > now)
// ---------------------------------------------------------------------------
export async function retrieveTemporal(
  opts: TemporalQuery = {},
): Promise<MemoryEvent[]> {
  const windowHours = opts.windowHours ?? DEFAULT_WINDOW_HOURS;
  const atTime = opts.atTime ?? new Date();
  const limit = opts.limit ?? 20;

  const actorFilter =
    opts.actors && opts.actors.length > 0
      ? `AND actor = ANY($3)`
      : '';

  const params: unknown[] = [atTime.toISOString(), windowHours];
  if (opts.actors && opts.actors.length > 0) params.push(opts.actors);

  const { rows } = await pool.query<MemoryEvent>(`
    SELECT * FROM memory_events
    WHERE event_time > $1::timestamptz - ($2 || ' hours')::interval
      AND event_time <= $1::timestamptz
      AND (valid_from IS NULL OR valid_from <= $1::timestamptz)
      AND (valid_until IS NULL OR valid_until > $1::timestamptz)
      ${actorFilter}
    ORDER BY event_time DESC
    LIMIT ${limit}
  `, params);

  return rows;
}

// ---------------------------------------------------------------------------
// What did the system believe was valid at a specific past time?
// The Graphiti-style temporal reconstruction.
// ---------------------------------------------------------------------------
export async function snapshotAt(
  atTime: Date,
  limit = 50,
): Promise<MemoryEvent[]> {
  const { rows } = await pool.query<MemoryEvent>(`
    SELECT * FROM memory_events
    WHERE (valid_from IS NULL OR valid_from <= $1)
      AND (valid_until IS NULL OR valid_until > $1)
      AND event_time <= $1
    ORDER BY confidence DESC, event_time DESC
    LIMIT $2
  `, [atTime.toISOString(), limit]);
  return rows;
}

// ---------------------------------------------------------------------------
// Find events in a validity window that overlaps with [rangeStart, rangeEnd].
// ---------------------------------------------------------------------------
export async function eventsInRange(
  rangeStart: Date,
  rangeEnd: Date,
  limit = 30,
): Promise<MemoryEvent[]> {
  const { rows } = await pool.query<MemoryEvent>(`
    SELECT * FROM memory_events
    WHERE (valid_from IS NULL OR valid_from <= $2)
      AND (valid_until IS NULL OR valid_until >= $1)
      AND event_time BETWEEN $1 AND $2
    ORDER BY event_time DESC
    LIMIT $3
  `, [rangeStart.toISOString(), rangeEnd.toISOString(), limit]);
  return rows;
}
