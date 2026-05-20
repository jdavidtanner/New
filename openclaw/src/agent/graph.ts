import { StateGraph, END, START } from '@langchain/langgraph';
import { PostgresSaver } from '@langchain/langgraph-checkpoint-postgres';
import { AgentState } from './state.js';
import {
  ingestInput,
  classifyIntent,
  retrieveSemanticMemory,
  retrieveGraphMemory,
  retrieveTemporalMemory,
  retrieveRelevantBeliefs,
  retrieveRelevantSkills,
  enforceContextBudget,
  planNextAction,
  runSkillHealthChecks,
  executeSkill,
  evaluateOutcome,
  writeMemory,
  reviseBeliefNode,
} from './nodes.js';

// ---------------------------------------------------------------------------
// Build and return the compiled LangGraph agent.
// The retrieval nodes (semantic, graph, temporal, beliefs, skills) run in
// parallel via independent edges from a fan-out node.
// ---------------------------------------------------------------------------
export async function buildAgent() {
  const graph = new StateGraph(AgentState)
    // -- Linear prefix --
    .addNode('ingestInput',      ingestInput)
    .addNode('classifyIntent',   classifyIntent)

    // -- Fan-out retrieval (all parallel) --
    .addNode('retrieveSemantic', retrieveSemanticMemory)
    .addNode('retrieveGraph',    retrieveGraphMemory)
    .addNode('retrieveTemporal', retrieveTemporalMemory)
    .addNode('retrieveBeliefs',  retrieveRelevantBeliefs)
    .addNode('retrieveSkills',   retrieveRelevantSkills)

    // -- Budget enforcement + planning --
    .addNode('enforceContextBudget', enforceContextBudget)
    .addNode('planNextAction',       planNextAction)

    // -- Execution --
    .addNode('runHealthChecks', runSkillHealthChecks)
    .addNode('executeSkill',    executeSkill)
    .addNode('evaluateOutcome', evaluateOutcome)

    // -- Memory + belief write-back --
    .addNode('writeMemory',     writeMemory)
    .addNode('reviseBeliefs',   reviseBeliefNode)

    // -- Edge wiring --
    .addEdge(START, 'ingestInput')
    .addEdge('ingestInput', 'classifyIntent')

    // Fan-out from classify → all retrieval nodes simultaneously
    .addEdge('classifyIntent', 'retrieveSemantic')
    .addEdge('classifyIntent', 'retrieveTemporal')
    .addEdge('classifyIntent', 'retrieveBeliefs')
    .addEdge('classifyIntent', 'retrieveSkills')

    // Graph retrieval depends on semantic seeds (sequential)
    .addEdge('retrieveSemantic', 'retrieveGraph')

    // Fan-in: all retrieval must complete before budget enforcement
    .addEdge('retrieveGraph',    'enforceContextBudget')
    .addEdge('retrieveTemporal', 'enforceContextBudget')
    .addEdge('retrieveBeliefs',  'enforceContextBudget')
    .addEdge('retrieveSkills',   'enforceContextBudget')

    .addEdge('enforceContextBudget', 'planNextAction')
    .addEdge('planNextAction',       'runHealthChecks')
    .addEdge('runHealthChecks',      'executeSkill')
    .addEdge('executeSkill',         'evaluateOutcome')
    .addEdge('evaluateOutcome',      'writeMemory')
    .addEdge('writeMemory',          'reviseBeliefs')
    .addEdge('reviseBeliefs',        END);

  const checkpointer = PostgresSaver.fromConnString(process.env.DATABASE_URL!);
  await checkpointer.setup();

  return graph.compile({ checkpointer });
}

// Singleton — built once per process lifetime
let _agent: Awaited<ReturnType<typeof buildAgent>> | null = null;

export async function getAgent() {
  if (!_agent) _agent = await buildAgent();
  return _agent;
}
