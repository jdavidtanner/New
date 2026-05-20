import { cypher, withClient, ageSession } from '../db/client.js';
import type { MemoryEvent } from './events.js';

export interface GraphMemory {
  id: string;
  content: string;
  actor: string;
  event_time: string;
  hops: number;
  edge_types: string[];
}

// ---------------------------------------------------------------------------
// Write a memory event as a vertex in the AGE graph.
// Also creates edges to recent related vertices (co-session proximity).
// ---------------------------------------------------------------------------
export async function writeEventToGraph(event: MemoryEvent): Promise<void> {
  const safeContent = event.content
    .replace(/\\/g, '\\\\')
    .replace(/'/g, "\\'")
    .replace(/\n/g, ' ')
    .slice(0, 500);

  try {
    await cypher(
      `CREATE (:MemoryEvent {
         id: '${event.id}',
         actor: '${event.actor}',
         content: '${safeContent}',
         source: '${event.source}',
         event_time: '${event.event_time.toISOString()}',
         confidence: ${event.confidence}
       }) RETURN 1`,
      'result',
    );
  } catch (err) {
    // AGE not installed or graph not initialised — log and continue
    console.warn('[graph] AGE write skipped:', (err as Error).message);
  }
}

// ---------------------------------------------------------------------------
// Create a typed directed edge between two memory event vertices.
// ---------------------------------------------------------------------------
export async function addGraphEdge(
  fromId: string,
  toId: string,
  edgeType: string,
  weight = 1.0,
): Promise<void> {
  const safeType = edgeType.replace(/[^A-Z_]/gi, '_').toUpperCase();
  try {
    await cypher(
      `MATCH (a:MemoryEvent {id: '${fromId}'}), (b:MemoryEvent {id: '${toId}'})
       CREATE (a)-[:${safeType} {weight: ${weight}}]->(b)
       RETURN 1`,
      'result',
    );
  } catch (err) {
    console.warn('[graph] AGE edge skipped:', (err as Error).message);
  }
}

// ---------------------------------------------------------------------------
// Multi-hop graph traversal from a set of seed event ids.
// Returns the reachable memory events within hop_limit hops.
// ---------------------------------------------------------------------------
export async function traverseFromSeeds(
  seedIds: string[],
  hopLimit = parseInt(process.env.GRAPH_HOP_LIMIT ?? '2'),
): Promise<GraphMemory[]> {
  if (seedIds.length === 0) return [];

  const idList = seedIds.map((id) => `'${id}'`).join(', ');

  try {
    const result = await cypher(
      `MATCH path = (seed:MemoryEvent)-[*1..${hopLimit}]-(neighbor:MemoryEvent)
       WHERE seed.id IN [${idList}]
         AND NOT neighbor.id IN [${idList}]
       RETURN neighbor.id AS id,
              neighbor.content AS content,
              neighbor.actor AS actor,
              neighbor.event_time AS event_time,
              length(path) AS hops,
              [r IN relationships(path) | type(r)] AS edge_types
       ORDER BY hops ASC
       LIMIT 20`,
      'id agtype, content agtype, actor agtype, event_time agtype, hops agtype, edge_types agtype',
    );

    return result.rows.map((row) => ({
      id: JSON.parse(row.id as unknown as string),
      content: JSON.parse(row.content as unknown as string),
      actor: JSON.parse(row.actor as unknown as string),
      event_time: JSON.parse(row.event_time as unknown as string),
      hops: Number(JSON.parse(row.hops as unknown as string)),
      edge_types: JSON.parse(row.edge_types as unknown as string) as string[],
    }));
  } catch (err) {
    console.warn('[graph] AGE traversal skipped:', (err as Error).message);
    return [];
  }
}

// ---------------------------------------------------------------------------
// Link belief vertices to memory event vertices.
// Called after belief creation to wire the justification graph.
// ---------------------------------------------------------------------------
export async function linkBeliefToEvent(
  beliefId: string,
  eventId: string,
  supportType: string,
): Promise<void> {
  try {
    // Upsert belief vertex if it doesn't exist
    await cypher(
      `MERGE (b:Belief {id: '${beliefId}'}) RETURN b`,
      'b',
    );
    await cypher(
      `MATCH (b:Belief {id: '${beliefId}'}), (e:MemoryEvent {id: '${eventId}'})
       CREATE (b)-[:JUSTIFIED_BY {support_type: '${supportType}'}]->(e)
       RETURN 1`,
      'result',
    );
  } catch (err) {
    console.warn('[graph] AGE belief link skipped:', (err as Error).message);
  }
}
