import { pool } from '../db/client.js';
import { embed } from '../db/embed.js';

export interface SemanticMemory {
  id: string;
  content: string;
  actor: string;
  source: string;
  confidence: number;
  event_time: Date;
  similarity: number;
  provenance: Record<string, unknown>;
}

const DEFAULT_LIMIT = parseInt(process.env.SEMANTIC_RETRIEVAL_LIMIT ?? '12');

// ---------------------------------------------------------------------------
// ANN search over memory_embeddings using pgvector cosine distance.
// Only returns currently-valid events (valid_until IS NULL or in the future).
// ---------------------------------------------------------------------------
export async function retrieveSemantic(
  query: string,
  opts: {
    limit?: number;
    minSimilarity?: number;
    actor?: string;
  } = {},
): Promise<SemanticMemory[]> {
  const embedding = await embed(query);
  const limit = opts.limit ?? DEFAULT_LIMIT;
  const minSim = opts.minSimilarity ?? 0.3;

  const actorClause = opts.actor
    ? `AND me.actor = '${opts.actor.replace(/'/g, "''")}'`
    : '';

  const { rows } = await pool.query<SemanticMemory>(`
    SELECT
      me.id,
      me.content,
      me.actor,
      me.source,
      me.confidence,
      me.event_time,
      me.provenance,
      (1 - (emb.embedding <=> $1::vector)) AS similarity
    FROM memory_embeddings emb
    JOIN memory_events me ON emb.memory_event_id = me.id
    WHERE (me.valid_until IS NULL OR me.valid_until > now())
      ${actorClause}
    ORDER BY emb.embedding <=> $1::vector
    LIMIT $2
  `, [JSON.stringify(embedding), limit]);

  return rows.filter((r) => r.similarity >= minSim);
}

// ---------------------------------------------------------------------------
// Hybrid search: ANN + recency boost.
// Scores = cosine_sim * relevance_weight + recency_weight
// ---------------------------------------------------------------------------
export async function retrieveHybrid(
  query: string,
  opts: {
    limit?: number;
    recencyHalfLifeHours?: number;
    semanticWeight?: number;
  } = {},
): Promise<SemanticMemory[]> {
  const semantic = await retrieveSemantic(query, { limit: (opts.limit ?? 12) * 3 });
  const halfLife = (opts.recencyHalfLifeHours ?? 72) * 3_600_000;
  const semW = opts.semanticWeight ?? 0.7;
  const recW = 1 - semW;
  const now = Date.now();

  const scored = semantic.map((m) => {
    const ageMsec = now - new Date(m.event_time).getTime();
    const recencyScore = Math.exp(-ageMsec / halfLife);
    const blended = semW * m.similarity + recW * recencyScore;
    return { ...m, similarity: blended };
  });

  scored.sort((a, b) => b.similarity - a.similarity);
  return scored.slice(0, opts.limit ?? 12);
}
