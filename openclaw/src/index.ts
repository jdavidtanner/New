import 'dotenv/config';
import { v4 as uuid } from 'uuid';
import { getAgent } from './agent/graph.js';
import type { AgentStateType } from './agent/state.js';

export interface AgentTurnOptions {
  threadId?: string;    // Persistent conversation thread (checkpointed)
  userId?: string;
}

export interface AgentTurnResult {
  response: string;
  threadId: string;
  outcome: string;
  memoryEventId: string;
  revisionSummary: {
    newBeliefs: number;
    updatedBeliefs: number;
    deprecatedBeliefs: number;
    contradictions: number;
  };
  budgetReport?: {
    memoryTokensUsed: number;
    skillTokensUsed: number;
    totalBudget: number;
    utilizationPct: number;
  };
}

// ---------------------------------------------------------------------------
// Primary entry point for a single agent turn.
// ---------------------------------------------------------------------------
export async function agentTurn(
  userQuery: string,
  opts: AgentTurnOptions = {},
): Promise<AgentTurnResult> {
  const agent = await getAgent();
  const threadId = opts.threadId ?? uuid();

  const initialState: Partial<AgentStateType> = {
    userQuery,
    messages: [],
    semanticMemories: [],
    graphMemories: [],
    temporalMemories: [],
    activeBeliefs: [],
    relevantSkills: [],
    contextBudget: null,
    packedMemory: null,
    packedSkills: null,
    revisionResult: null,
    error: null,
  };

  const config = {
    configurable: { thread_id: threadId },
  };

  let finalState: AgentStateType | null = null;

  for await (const step of await agent.stream(initialState, config)) {
    // Each step is a map of node_name → partial state
    for (const [, partial] of Object.entries(step)) {
      finalState = { ...(finalState ?? {}), ...(partial as Partial<AgentStateType>) } as AgentStateType;
    }
  }

  if (!finalState) {
    throw new Error('Agent produced no output state');
  }

  const r = finalState.revisionResult;
  const budget = finalState.contextBudget;
  const packed = finalState.packedMemory;
  const skills = finalState.packedSkills;

  return {
    response: finalState.agentResponse ?? '',
    threadId,
    outcome: finalState.outcome ?? 'success',
    memoryEventId: finalState.memoryEventId ?? '',
    revisionSummary: {
      newBeliefs:       r?.newBeliefs.length ?? 0,
      updatedBeliefs:   r?.updatedBeliefs.length ?? 0,
      deprecatedBeliefs: r?.deprecatedBeliefs.length ?? 0,
      contradictions:   r?.contradictionsDetected.length ?? 0,
    },
    budgetReport: budget
      ? {
          memoryTokensUsed: packed?.tokenCount ?? 0,
          skillTokensUsed:  skills?.tokenCount ?? 0,
          totalBudget:      budget.maxTokens,
          utilizationPct:   Math.round(
            ((packed?.tokenCount ?? 0) + (skills?.tokenCount ?? 0)) /
            budget.maxTokens * 100,
          ),
        }
      : undefined,
  };
}

// ---------------------------------------------------------------------------
// Re-export public surface for library consumers.
// ---------------------------------------------------------------------------
export { writeMemoryEvent, getMemoryEvent } from './memory/events.js';
export { retrieveSemantic, retrieveHybrid } from './memory/semantic.js';
export { retrieveTemporal, snapshotAt } from './memory/temporal.js';
export { createBelief, findBeliefs, adjustConfidence } from './beliefs/store.js';
export { reviseBeliefs } from './beliefs/revision.js';
export { registerSkill, listSkills } from './skills/registry.js';
export { allocateContextBudget } from './context/budget.js';
