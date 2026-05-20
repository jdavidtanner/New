import { Annotation, messagesStateReducer } from '@langchain/langgraph';
import type { BaseMessage } from '@langchain/core/messages';
import type { SemanticMemory } from '../memory/semantic.js';
import type { GraphMemory } from '../memory/graph.js';
import type { MemoryEvent } from '../memory/events.js';
import type { Skill } from '../skills/registry.js';
import type { Belief } from '../beliefs/store.js';
import type { ContextBudget, PackedMemory, PackedSkills } from '../context/budget.js';
import type { RevisionResult } from '../beliefs/revision.js';

// ---------------------------------------------------------------------------
// Top-level LangGraph agent state.
// Each field uses its own reducer; messages uses the built-in append reducer.
// ---------------------------------------------------------------------------
export const AgentState = Annotation.Root({
  // Conversation messages (append-only via built-in reducer)
  messages: Annotation<BaseMessage[]>({
    reducer: messagesStateReducer,
    default: () => [],
  }),

  // Raw input for the current turn
  userQuery: Annotation<string>({ reducer: (_, v) => v, default: () => '' }),

  // Intent classification result
  intent: Annotation<string>({ reducer: (_, v) => v, default: () => '' }),

  // Retrieved memories
  semanticMemories: Annotation<SemanticMemory[]>({
    reducer: (_, v) => v,
    default: () => [],
  }),
  graphMemories: Annotation<GraphMemory[]>({
    reducer: (_, v) => v,
    default: () => [],
  }),
  temporalMemories: Annotation<MemoryEvent[]>({
    reducer: (_, v) => v,
    default: () => [],
  }),

  // Active beliefs retrieved for this turn
  activeBeliefs: Annotation<Belief[]>({
    reducer: (_, v) => v,
    default: () => [],
  }),

  // Skills selected for this turn
  relevantSkills: Annotation<Skill[]>({
    reducer: (_, v) => v,
    default: () => [],
  }),

  // Context budget and assembled prompt sections
  contextBudget: Annotation<ContextBudget | null>({
    reducer: (_, v) => v,
    default: () => null,
  }),
  packedMemory: Annotation<PackedMemory | null>({
    reducer: (_, v) => v,
    default: () => null,
  }),
  packedSkills: Annotation<PackedSkills | null>({
    reducer: (_, v) => v,
    default: () => null,
  }),

  // Planning output
  plan: Annotation<string>({ reducer: (_, v) => v, default: () => '' }),

  // Skill execution
  selectedSkill: Annotation<string>({ reducer: (_, v) => v, default: () => '' }),
  skillInput: Annotation<Record<string, unknown>>({
    reducer: (_, v) => v,
    default: () => ({}),
  }),
  skillOutput: Annotation<unknown>({ reducer: (_, v) => v, default: () => null }),

  // Outcome evaluation
  outcome: Annotation<'success' | 'failure' | 'partial'>({
    reducer: (_, v) => v,
    default: () => 'success',
  }),
  outcomeDetail: Annotation<string>({ reducer: (_, v) => v, default: () => '' }),

  // Final agent response text
  agentResponse: Annotation<string>({ reducer: (_, v) => v, default: () => '' }),

  // Memory write result
  memoryEventId: Annotation<string>({ reducer: (_, v) => v, default: () => '' }),

  // Belief revision result
  revisionResult: Annotation<RevisionResult | null>({
    reducer: (_, v) => v,
    default: () => null,
  }),

  // Error propagation
  error: Annotation<string | null>({ reducer: (_, v) => v, default: () => null }),
});

export type AgentStateType = typeof AgentState.State;
