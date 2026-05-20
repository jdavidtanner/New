import Anthropic from '@anthropic-ai/sdk';
import {
  findBeliefs,
  createBelief,
  addJustification,
  adjustConfidence,
  deprecateBelief,
  type Belief,
} from './store.js';

const client = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });
const MODEL = process.env.ANTHROPIC_MODEL ?? 'claude-opus-4-5';

export interface RevisionInput {
  userQuery: string;
  agentResponse: string;
  outcome: string;       // "success" | "failure" | "partial"
  outcomeDetail: string; // Natural-language description of what happened
  memoryEventId: string;
}

export interface RevisionResult {
  newBeliefs: Belief[];
  updatedBeliefs: { id: string; delta: number }[];
  deprecatedBeliefs: string[];
  contradictionsDetected: string[];
}

// ---------------------------------------------------------------------------
// After each agent turn, extract claims from the exchange and reconcile them
// against the existing belief graph.
//
// Strategy:
// 1. Ask the LLM to extract atomic claims from the interaction.
// 2. For each claim, find existing beliefs about the same subject+predicate.
// 3. If the claim reinforces an existing belief → boost confidence.
// 4. If it contradicts → mark contested, reduce confidence.
// 5. If novel → create new belief with moderate confidence.
// ---------------------------------------------------------------------------
export async function reviseBeliefs(
  input: RevisionInput,
): Promise<RevisionResult> {
  const extracted = await extractClaims(input);

  const result: RevisionResult = {
    newBeliefs: [],
    updatedBeliefs: [],
    deprecatedBeliefs: [],
    contradictionsDetected: [],
  };

  for (const claim of extracted) {
    const existing = await findBeliefs({
      subject: claim.subject,
      predicate: claim.predicate,
      limit: 5,
    });

    if (existing.length === 0) {
      // Novel claim — create a new belief
      const newBelief = await createBelief({
        claim: claim.claim,
        subject: claim.subject,
        predicate: claim.predicate,
        object: claim.object,
        confidence: outcomeConfidence(input.outcome, claim.confidence),
      });
      await addJustification({
        belief_id: newBelief.id,
        memory_event_id: input.memoryEventId,
        support_type: 'direct_statement',
        weight: 1.0,
        explanation: `Extracted from: ${input.outcomeDetail.slice(0, 200)}`,
      });
      result.newBeliefs.push(newBelief);
    } else {
      // Check alignment with each existing belief
      for (const existingBelief of existing) {
        const alignment = await checkAlignment(
          claim.claim,
          existingBelief.claim,
        );

        if (alignment === 'supports') {
          const delta = input.outcome === 'success' ? +0.05 : -0.02;
          await adjustConfidence(existingBelief.id, delta);
          await addJustification({
            belief_id: existingBelief.id,
            memory_event_id: input.memoryEventId,
            support_type: 'reinforcement',
            weight: 0.8,
          });
          result.updatedBeliefs.push({ id: existingBelief.id, delta });
        } else if (alignment === 'contradicts') {
          await adjustConfidence(existingBelief.id, -0.15);
          await addJustification({
            belief_id: existingBelief.id,
            memory_event_id: input.memoryEventId,
            support_type: 'contradiction',
            weight: 1.0,
            explanation: `New claim: ${claim.claim}`,
          });
          result.contradictionsDetected.push(existingBelief.id);
          result.updatedBeliefs.push({ id: existingBelief.id, delta: -0.15 });

          // If confidence drops below threshold, deprecate and replace
          if (existingBelief.confidence - 0.15 < 0.2) {
            await deprecateBelief(
              existingBelief.id,
              `Superseded by: ${claim.claim}`,
            );
            result.deprecatedBeliefs.push(existingBelief.id);
            const replacement = await createBelief({
              claim: claim.claim,
              subject: claim.subject,
              predicate: claim.predicate,
              object: claim.object,
              confidence: 0.6,
            });
            await addJustification({
              belief_id: replacement.id,
              memory_event_id: input.memoryEventId,
              support_type: 'correction',
              weight: 1.2,
            });
            result.newBeliefs.push(replacement);
          }
        }
        // "unrelated" — skip
      }
    }
  }

  return result;
}

// ---------------------------------------------------------------------------
// Extract atomic, triple-structured claims from an agent interaction.
// ---------------------------------------------------------------------------
interface ExtractedClaim {
  claim: string;
  subject: string;
  predicate: string;
  object: string;
  confidence: number;
}

async function extractClaims(
  input: RevisionInput,
): Promise<ExtractedClaim[]> {
  const prompt = `You are a belief extractor. Given a user-agent interaction and its outcome, extract atomic factual claims that should be stored as beliefs.

User query: ${input.userQuery}
Agent response: ${input.agentResponse.slice(0, 1000)}
Outcome: ${input.outcome} — ${input.outcomeDetail.slice(0, 300)}

Extract 0-5 specific, atomic claims. Ignore conversational pleasantries.
Return JSON array (can be empty):
[{
  "claim": "complete claim sentence",
  "subject": "the thing this is about",
  "predicate": "the relationship or attribute",
  "object": "the value or target",
  "confidence": 0.0-1.0
}]

Return only the JSON array, no other text.`;

  try {
    const msg = await client.messages.create({
      model: MODEL,
      max_tokens: 800,
      messages: [{ role: 'user', content: prompt }],
    });
    const text = (msg.content[0] as { text: string }).text.trim();
    const parsed = JSON.parse(text);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

// ---------------------------------------------------------------------------
// Check whether two claims support, contradict, or are unrelated.
// ---------------------------------------------------------------------------
async function checkAlignment(
  newClaim: string,
  existingClaim: string,
): Promise<'supports' | 'contradicts' | 'unrelated'> {
  const prompt = `Do these two claims support each other, contradict each other, or are they unrelated?

Claim A: ${newClaim}
Claim B: ${existingClaim}

Reply with exactly one word: supports, contradicts, or unrelated.`;

  try {
    const msg = await client.messages.create({
      model: MODEL,
      max_tokens: 10,
      messages: [{ role: 'user', content: prompt }],
    });
    const text = (msg.content[0] as { text: string }).text.trim().toLowerCase();
    if (text.includes('support'))     return 'supports';
    if (text.includes('contradict'))  return 'contradicts';
    return 'unrelated';
  } catch {
    return 'unrelated';
  }
}

function outcomeConfidence(
  outcome: string,
  baseClaim: number,
): number {
  const multiplier =
    outcome === 'success' ? 1.1 :
    outcome === 'failure' ? 0.7 :
    0.9;
  return Math.min(0.9, baseClaim * multiplier);
}
