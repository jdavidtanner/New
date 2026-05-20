import pg from 'pg';

const { Pool } = pg;

if (!process.env.DATABASE_URL) {
  throw new Error('DATABASE_URL environment variable is required');
}

export const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
  max: 10,
  idleTimeoutMillis: 30_000,
  connectionTimeoutMillis: 5_000,
});

pool.on('error', (err) => {
  console.error('[db] Unexpected pool error', err);
});

// ---------------------------------------------------------------------------
// AGE session initializer
// Must be called on any client that will run Cypher queries.
// ---------------------------------------------------------------------------
export async function ageSession(client: pg.PoolClient): Promise<void> {
  await client.query("LOAD 'age'");
  await client.query(`SET search_path = ag_catalog, "$user", public`);
}

// ---------------------------------------------------------------------------
// Run a Cypher query against the openclaw graph.
// Returns raw rows. Caller is responsible for typing the agtype output.
// ---------------------------------------------------------------------------
export async function cypher(
  query: string,
  returnAlias: string,
): Promise<pg.QueryResult> {
  const client = await pool.connect();
  try {
    await ageSession(client);
    return await client.query(
      `SELECT * FROM cypher('openclaw', $q$${query}$q$) AS (${returnAlias} agtype)`,
    );
  } finally {
    client.release();
  }
}

export async function withClient<T>(
  fn: (client: pg.PoolClient) => Promise<T>,
): Promise<T> {
  const client = await pool.connect();
  try {
    return await fn(client);
  } finally {
    client.release();
  }
}
