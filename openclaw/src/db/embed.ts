import OpenAI from 'openai';

const EMBEDDING_MODEL = 'text-embedding-3-small';
const EMBEDDING_DIMS = 1536;

if (!process.env.OPENAI_API_KEY) {
  throw new Error('OPENAI_API_KEY environment variable is required');
}

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

let _requestCount = 0;

export async function embed(text: string): Promise<number[]> {
  const trimmed = text.trim().slice(0, 8000);
  const res = await openai.embeddings.create({
    model: EMBEDDING_MODEL,
    input: trimmed,
    dimensions: EMBEDDING_DIMS,
  });
  _requestCount++;
  return res.data[0].embedding;
}

export async function embedBatch(texts: string[]): Promise<number[][]> {
  if (texts.length === 0) return [];
  const trimmed = texts.map((t) => t.trim().slice(0, 8000));
  const res = await openai.embeddings.create({
    model: EMBEDDING_MODEL,
    input: trimmed,
    dimensions: EMBEDDING_DIMS,
  });
  _requestCount++;
  return res.data.map((d) => d.embedding);
}

// Cosine similarity between two embeddings (both already unit-normalized by OpenAI)
export function cosineSim(a: number[], b: number[]): number {
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return dot;
}

export function embeddingStats() {
  return { requestCount: _requestCount, model: EMBEDDING_MODEL, dims: EMBEDDING_DIMS };
}
