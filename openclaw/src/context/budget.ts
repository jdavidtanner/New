import type { SemanticMemory } from '../memory/semantic.js';
import type { GraphMemory } from '../memory/graph.js';
import type { MemoryEvent } from '../memory/events.js';
import type { Skill } from '../skills/registry.js';
import type { Belief } from '../beliefs/store.js';

// ---------------------------------------------------------------------------
// Context budget — every token slot is explicitly claimed before assembly.
// Allocations must sum to <= maxTokens.
// ---------------------------------------------------------------------------
export interface ContextBudget {
  maxTokens: number;
  systemTokens: number;         // 12% — system prompt + agent instructions
  userInputTokens: number;      // 10% — current user message
  reservedResponseTokens: number; // 15% — space for model output
  memoryBudget: number;         // 35% — semantic + graph + temporal memories
  skillBudget: number;          // 15% — skill descriptions injected into prompt
  reasoningBudget: number;      // 13% — chain-of-thought / scratchpad
}

export function allocateContextBudget(maxTokens: number): ContextBudget {
  return {
    maxTokens,
    systemTokens:           Math.floor(maxTokens * 0.12),
    userInputTokens:        Math.floor(maxTokens * 0.10),
    reservedResponseTokens: Math.floor(maxTokens * 0.15),
    memoryBudget:           Math.floor(maxTokens * 0.35),
    skillBudget:            Math.floor(maxTokens * 0.15),
    reasoningBudget:        Math.floor(maxTokens * 0.13),
  };
}

// ---------------------------------------------------------------------------
// Rough token estimator: ~4 chars per token (GPT-style approximation).
// Good enough for budget enforcement without a tokeniser dependency.
// ---------------------------------------------------------------------------
export function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4);
}

// ---------------------------------------------------------------------------
// Assemble memory context within budget.
// Priority: semantic > graph > temporal (semantic is highest signal).
// ---------------------------------------------------------------------------
export interface PackedMemory {
  text: string;
  tokenCount: number;
  included: {
    semantic: number;
    graph: number;
    temporal: number;
    beliefs: number;
  };
  truncated: boolean;
}

export function packMemoryContext(
  semantic: SemanticMemory[],
  graph: GraphMemory[],
  temporal: MemoryEvent[],
  beliefs: Belief[],
  budget: ContextBudget,
): PackedMemory {
  const sections: string[] = [];
  let used = 0;
  const remaining = () => budget.memoryBudget - used;
  const included = { semantic: 0, graph: 0, temporal: 0, beliefs: 0 };
  let truncated = false;

  // -- Beliefs (highest provenance value) --
  if (beliefs.length > 0) {
    const header = '## Active Beliefs\n';
    used += estimateTokens(header);
    sections.push(header);
    for (const b of beliefs) {
      const line = `- [${(b.confidence * 100).toFixed(0)}%] ${b.claim}\n`;
      const t = estimateTokens(line);
      if (t > remaining()) { truncated = true; break; }
      sections.push(line);
      used += t;
      included.beliefs++;
    }
  }

  // -- Semantic memories --
  if (semantic.length > 0) {
    const header = '\n## Relevant Memories\n';
    used += estimateTokens(header);
    sections.push(header);
    for (const m of semantic) {
      const line = `- [${m.actor}, sim=${m.similarity.toFixed(2)}] ${m.content.slice(0, 300)}\n`;
      const t = estimateTokens(line);
      if (t > remaining()) { truncated = true; break; }
      sections.push(line);
      used += t;
      included.semantic++;
    }
  }

  // -- Graph memories --
  if (graph.length > 0) {
    const header = '\n## Graph Context\n';
    used += estimateTokens(header);
    sections.push(header);
    for (const g of graph) {
      const line = `- [${g.hops} hop(s), via ${g.edge_types.join('>')}] ${g.content.slice(0, 200)}\n`;
      const t = estimateTokens(line);
      if (t > remaining()) { truncated = true; break; }
      sections.push(line);
      used += t;
      included.graph++;
    }
  }

  // -- Temporal memories --
  if (temporal.length > 0) {
    const header = '\n## Recent Events\n';
    used += estimateTokens(header);
    sections.push(header);
    for (const e of temporal) {
      const line = `- [${e.event_time.toISOString().slice(0, 16)}, ${e.actor}] ${e.content.slice(0, 200)}\n`;
      const t = estimateTokens(line);
      if (t > remaining()) { truncated = true; break; }
      sections.push(line);
      used += t;
      included.temporal++;
    }
  }

  return {
    text: sections.join(''),
    tokenCount: used,
    included,
    truncated,
  };
}

// ---------------------------------------------------------------------------
// Assemble skill context within skillBudget.
// ---------------------------------------------------------------------------
export interface PackedSkills {
  text: string;
  tokenCount: number;
  includedCount: number;
}

export function packSkillContext(
  skills: Skill[],
  budget: ContextBudget,
): PackedSkills {
  let used = 0;
  let count = 0;
  const lines: string[] = [];
  const header = '## Available Skills\n';
  used += estimateTokens(header);
  lines.push(header);

  for (const s of skills) {
    const line = `- **${s.name}** [${s.health_status}]: ${s.description}\n`;
    const t = estimateTokens(line);
    if (used + t > budget.skillBudget) break;
    lines.push(line);
    used += t;
    count++;
  }

  return { text: lines.join(''), tokenCount: used, includedCount: count };
}

// ---------------------------------------------------------------------------
// Budget utilisation report (for the debug UI / logging).
// ---------------------------------------------------------------------------
export interface BudgetReport {
  budget: ContextBudget;
  memoryUsed: number;
  skillsUsed: number;
  utilizationPct: number;
}

export function buildBudgetReport(
  budget: ContextBudget,
  packed: PackedMemory,
  skills: PackedSkills,
): BudgetReport {
  const memoryUsed = packed.tokenCount;
  const skillsUsed = skills.tokenCount;
  const totalUsed =
    budget.systemTokens +
    budget.userInputTokens +
    memoryUsed +
    skillsUsed +
    budget.reasoningBudget +
    budget.reservedResponseTokens;
  return {
    budget,
    memoryUsed,
    skillsUsed,
    utilizationPct: Math.round((totalUsed / budget.maxTokens) * 100),
  };
}
