import { pool } from '../db/client.js';
import type { Skill } from './registry.js';

const TIMEOUT_MS = parseInt(process.env.SKILL_HEALTH_TIMEOUT_MS ?? '5000');

// Minimum success rate to remain "healthy" (vs "degraded")
const DEGRADED_THRESHOLD = 0.7;
const UNHEALTHY_THRESHOLD = 0.4;

export interface HealthCheckResult {
  skillId: string;
  skillName: string;
  status: Skill['health_status'];
  latencyMs: number | null;
  error?: string;
}

// ---------------------------------------------------------------------------
// Run a health-check ping for a single skill.
// The check calls the skill's implementation_ref as a module export named
// `healthCheck`. If no such export exists, we use a latency-based heuristic
// derived from historical stats.
// ---------------------------------------------------------------------------
export async function checkSkillHealth(
  skill: Skill,
): Promise<HealthCheckResult> {
  const start = Date.now();

  try {
    // Attempt to dynamically import and call a `healthCheck` export
    const mod = await import(skill.implementation_ref).catch(() => null);

    if (mod && typeof mod.healthCheck === 'function') {
      await Promise.race([
        mod.healthCheck(),
        new Promise<never>((_, reject) =>
          setTimeout(() => reject(new Error('timeout')), TIMEOUT_MS),
        ),
      ]);
      const latencyMs = Date.now() - start;
      const status = deriveStatus(skill, true);
      await updateHealthStatus(skill.id, status);
      return { skillId: skill.id, skillName: skill.name, status, latencyMs };
    } else {
      // No dedicated check — use historical success-rate heuristic
      const status = deriveStatus(skill, null);
      await updateHealthStatus(skill.id, status);
      return { skillId: skill.id, skillName: skill.name, status, latencyMs: null };
    }
  } catch (err) {
    const latencyMs = Date.now() - start;
    const status: Skill['health_status'] = 'degraded';
    await updateHealthStatus(skill.id, status);
    return {
      skillId: skill.id,
      skillName: skill.name,
      status,
      latencyMs,
      error: (err as Error).message,
    };
  }
}

// ---------------------------------------------------------------------------
// Run health checks for all skills that haven't been checked recently.
// ---------------------------------------------------------------------------
export async function runHealthChecks(
  staleSinceMinutes = 30,
): Promise<HealthCheckResult[]> {
  const { rows: skills } = await pool.query<Skill>(`
    SELECT * FROM skills
    WHERE health_status = 'unknown'
       OR updated_at < now() - ($1 || ' minutes')::interval
    ORDER BY updated_at ASC
    LIMIT 20
  `, [staleSinceMinutes]);

  const results = await Promise.allSettled(
    skills.map((s) => checkSkillHealth(s)),
  );

  return results.map((r, i) =>
    r.status === 'fulfilled'
      ? r.value
      : {
          skillId: skills[i].id,
          skillName: skills[i].name,
          status: 'unknown' as const,
          latencyMs: null,
          error: (r.reason as Error).message,
        },
  );
}

// ---------------------------------------------------------------------------
// Derive health status from a skill's historical execution record.
// `pingSuccess`: true if a live ping just succeeded, false if it failed, null
// if no live ping was performed (heuristic-only).
// ---------------------------------------------------------------------------
function deriveStatus(
  skill: Skill,
  pingSuccess: boolean | null,
): Skill['health_status'] {
  const total = skill.success_count + skill.failure_count;

  if (pingSuccess === false) return 'degraded';
  if (total === 0) return pingSuccess === true ? 'healthy' : 'unknown';

  const rate = skill.success_count / total;
  if (rate >= DEGRADED_THRESHOLD) return 'healthy';
  if (rate >= UNHEALTHY_THRESHOLD) return 'degraded';
  return 'unhealthy';
}

async function updateHealthStatus(
  skillId: string,
  status: Skill['health_status'],
): Promise<void> {
  await pool.query(
    `UPDATE skills SET health_status = $1, updated_at = now() WHERE id = $2`,
    [status, skillId],
  );
}
