import Anthropic from '@anthropic-ai/sdk';
import { HumanMessage, AIMessage } from '@langchain/core/messages';
import { retrieveSemantic } from '../memory/semantic.js';
import { traverseFromSeeds, writeEventToGraph, linkBeliefToEvent } from '../memory/graph.js';
import { retrieveTemporal } from '../memory/temporal.js';
import { findBeliefs } from '../beliefs/store.js';
import { findRelevantSkills, recordSkillExecution } from '../skills/registry.js';
import { runHealthChecks } from '../skills/health.js';
import { writeMemoryEvent } from '../memory/events.js';
import { reviseBeliefs } from '../beliefs/revision.js';
import {
  allocateContextBudget,
  packMemoryContext,
  packSkillContext,
} from '../context/budget.js';
import type { AgentStateType } from './state.js';

const client = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });
const MODEL = process.env.ANTHROPIC_MODEL ?? 'claude-opus-4-5';
const MAX_TOKENS = parseInt(process.env.MAX_CONTEXT_TOKENS ?? '100000');

// ---------------------------------------------------------------------------
// Node 1 — Ingest user input, add to message history.
// ---------------------------------------------------------------------------
export async function ingestInput(
  state: AgentStateType,
): Promise<Partial<AgentStateType>> {
  return {
    messages: [new HumanMessage(state.userQuery)],
    error: null,
  };
}

// ---------------------------------------------------------------------------
// Node 2 — Classify intent for downstream routing.
// ---------------------------------------------------------------------------
export async function classifyIntent(
  state: AgentStateType,
): Promise<Partial<AgentStateType>> {
  const msg = await client.messages.create({
    model: MODEL,
    max_tokens: 50,
    messages: [{
      role: 'user',
      content: `Classify this request in 1-3 words (e.g. "code generation", "factual lookup", "task planning", "conversation"): ${state.userQuery}`,
    }],
  });
  const intent = (msg.content[0] as { text: string }).text.trim().toLowerCase();
  return { intent };
}

// ---------------------------------------------------------------------------
// Node 3a — Semantic memory retrieval.
// ---------------------------------------------------------------------------
export async function retrieveSemanticMemory(
  state: AgentStateType,
): Promise<Partial<AgentStateType>> {
  const memories = await retrieveSemantic(state.userQuery, { limit: 15 });
  return { semanticMemories: memories };
}

// ---------------------------------------------------------------------------
// Node 3b — Graph memory retrieval (multi-hop from semantic seed ids).
// ---------------------------------------------------------------------------
export async function retrieveGraphMemory(
  state: AgentStateType,
): Promise<Partial<AgentStateType>> {
  const seedIds = state.semanticMemories.slice(0, 5).map((m) => m.id);
  const graphMems = await traverseFromSeeds(seedIds);
  return { graphMemories: graphMems };
}

// ---------------------------------------------------------------------------
// Node 3c — Temporal memory retrieval.
// ---------------------------------------------------------------------------
export async function retrieveTemporalMemory(
  state: AgentStateType,
): Promise<Partial<AgentStateType>> {
  const temporal = await retrieveTemporal({ limit: 10 });
  return { temporalMemories: temporal };
}

// ---------------------------------------------------------------------------
// Node 3d — Retrieve relevant beliefs.
// ---------------------------------------------------------------------------
export async function retrieveRelevantBeliefs(
  state: AgentStateType,
): Promise<Partial<AgentStateType>> {
  // Pull active high-confidence beliefs (broad — planner will filter)
  const beliefs = await findBeliefs({ minConfidence: 0.4, limit: 15 });
  return { activeBeliefs: beliefs };
}

// ---------------------------------------------------------------------------
// Node 4 — Find skills relevant to the current intent.
// ---------------------------------------------------------------------------
export async function retrieveRelevantSkills(
  state: AgentStateType,
): Promise<Partial<AgentStateType>> {
  const skills = await findRelevantSkills(state.userQuery, { limit: 8 });
  return { relevantSkills: skills };
}

// ---------------------------------------------------------------------------
// Node 5 — Enforce context budget: pack memories and skills.
// ---------------------------------------------------------------------------
export async function enforceContextBudget(
  state: AgentStateType,
): Promise<Partial<AgentStateType>> {
  const budget = allocateContextBudget(MAX_TOKENS);
  const packedMemory = packMemoryContext(
    state.semanticMemories,
    state.graphMemories,
    state.temporalMemories,
    state.activeBeliefs,
    budget,
  );
  const packedSkills = packSkillContext(state.relevantSkills, budget);
  return { contextBudget: budget, packedMemory, packedSkills };
}

// ---------------------------------------------------------------------------
// Node 6 — Planner: produce a short action plan.
// ---------------------------------------------------------------------------
export async function planNextAction(
  state: AgentStateType,
): Promise<Partial<AgentStateType>> {
  const memCtx = state.packedMemory?.text ?? '';
  const skillCtx = state.packedSkills?.text ?? '';

  const systemPrompt = `You are a planning agent. Given the user request, available memories, and skills, output a concise action plan (3-5 bullet points). End with a line: SKILL: <skill_name> or SKILL: none.

${memCtx}
${skillCtx}`;

  const msg = await client.messages.create({
    model: MODEL,
    max_tokens: 500,
    system: systemPrompt,
    messages: [{ role: 'user', content: state.userQuery }],
  });
  const plan = (msg.content[0] as { text: string }).text.trim();
  const skillMatch = plan.match(/SKILL:\s*(\S+)/i);
  const selectedSkill = skillMatch ? skillMatch[1].toLowerCase() : 'none';

  return { plan, selectedSkill };
}

// ---------------------------------------------------------------------------
// Node 7 — Run health checks on skills scheduled for this turn.
// ---------------------------------------------------------------------------
export async function runSkillHealthChecks(
  state: AgentStateType,
): Promise<Partial<AgentStateType>> {
  // Fire-and-forget background check; don't block the turn on it.
  runHealthChecks(30).catch((e) =>
    console.warn('[health] background check failed:', (e as Error).message),
  );
  return {};
}

// ---------------------------------------------------------------------------
// Node 8 — Execute selected skill.
// ---------------------------------------------------------------------------
export async function executeSkill(
  state: AgentStateType,
): Promise<Partial<AgentStateType>> {
  if (!state.selectedSkill || state.selectedSkill === 'none') {
    // No skill — generate a direct response from the LLM
    const memCtx = state.packedMemory?.text ?? '';
    const msg = await client.messages.create({
      model: MODEL,
      max_tokens: state.contextBudget?.reservedResponseTokens ?? 2000,
      system: `You are a helpful agent with access to the following context:\n\n${memCtx}`,
      messages: [{ role: 'user', content: state.userQuery }],
    });
    const response = (msg.content[0] as { text: string }).text;
    return {
      skillOutput: response,
      agentResponse: response,
      outcome: 'success',
      outcomeDetail: 'Direct LLM response generated.',
    };
  }

  const skill = state.relevantSkills.find(
    (s) => s.name.toLowerCase() === state.selectedSkill,
  );
  if (!skill) {
    return {
      outcome: 'failure',
      outcomeDetail: `Skill "${state.selectedSkill}" not found in registry.`,
      agentResponse: `I couldn't find the skill needed to complete this task.`,
    };
  }

  const start = Date.now();
  try {
    const mod = await import(skill.implementation_ref);
    const output = await mod.execute(state.skillInput ?? { query: state.userQuery });
    const latencyMs = Date.now() - start;
    await recordSkillExecution(skill.id, { success: true, latencyMs });

    const response = typeof output === 'string' ? output : JSON.stringify(output, null, 2);
    return {
      skillOutput: output,
      agentResponse: response,
      outcome: 'success',
      outcomeDetail: `Skill "${skill.name}" executed in ${latencyMs}ms.`,
    };
  } catch (err) {
    const latencyMs = Date.now() - start;
    await recordSkillExecution(skill.id, { success: false, latencyMs });
    const detail = `Skill "${skill.name}" failed: ${(err as Error).message}`;
    return {
      outcome: 'failure',
      outcomeDetail: detail,
      agentResponse: `I encountered an error executing the requested skill.`,
      error: detail,
    };
  }
}

// ---------------------------------------------------------------------------
// Node 9 — Evaluate outcome quality (supplement with LLM if needed).
// ---------------------------------------------------------------------------
export async function evaluateOutcome(
  state: AgentStateType,
): Promise<Partial<AgentStateType>> {
  // Outcome was set by executeSkill; we just surface it here for legibility.
  // Could add LLM-based quality scoring in v0.2.
  return {
    outcomeDetail: state.outcomeDetail || `Turn completed with outcome: ${state.outcome}`,
  };
}

// ---------------------------------------------------------------------------
// Node 10 — Write memory event for this turn.
// ---------------------------------------------------------------------------
export async function writeMemory(
  state: AgentStateType,
): Promise<Partial<AgentStateType>> {
  const content = [
    `Q: ${state.userQuery}`,
    `A: ${state.agentResponse?.slice(0, 800)}`,
    `Outcome: ${state.outcome}`,
  ].join('\n');

  const event = await writeMemoryEvent({
    actor: 'agent',
    source: 'turn',
    content,
    confidence: state.outcome === 'success' ? 0.85 : 0.5,
    provenance: {
      intent: state.intent,
      skill: state.selectedSkill,
      memoryBudgetUsed: state.packedMemory?.tokenCount,
    },
  });

  await writeEventToGraph(event);

  // Wire beliefs to event in graph
  for (const belief of state.activeBeliefs) {
    await linkBeliefToEvent(belief.id, event.id, 'REFERENCED_IN');
  }

  return { memoryEventId: event.id, messages: [new AIMessage(state.agentResponse ?? '')] };
}

// ---------------------------------------------------------------------------
// Node 11 — Revise beliefs based on the completed turn.
// ---------------------------------------------------------------------------
export async function reviseBeliefNode(
  state: AgentStateType,
): Promise<Partial<AgentStateType>> {
  if (!state.memoryEventId) return {};

  const result = await reviseBeliefs({
    userQuery: state.userQuery,
    agentResponse: state.agentResponse ?? '',
    outcome: state.outcome,
    outcomeDetail: state.outcomeDetail,
    memoryEventId: state.memoryEventId,
  });

  return { revisionResult: result };
}
