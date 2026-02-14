#!/usr/bin/env node
/**
 * Derive math problem solver for MathArena.
 * Reads a math problem from stdin, runs all three agents (claude, codex, gemini)
 * in parallel, combines responses, runs critique + refine cycles,
 * and outputs the result to stdout as JSON with full per-round detail.
 *
 * Usage: echo "problem text" | node solve.mjs
 * Output: JSON with response, per-round scores, agent outputs, timing
 */

import { spawn } from 'child_process';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'fs';
import { join } from 'path';
import { homedir } from 'os';

const CLAUDE_MODEL = process.env.DERIVE_CLAUDE_MODEL ?? 'claude-opus-4-6';
const CODEX_MODEL = process.env.DERIVE_CODEX_MODEL ?? 'gpt-5.3-codex';
const GEMINI_MODEL = process.env.DERIVE_GEMINI_MODEL ?? 'gemini-3-pro-preview';
const COMBINER_MODEL = process.env.DERIVE_ORCHESTRATOR_MODEL_FAST ?? 'claude-sonnet-4-5';
const MAX_CRITIQUE_ROUNDS = parseInt(process.env.DERIVE_MAX_CRITIQUE_ROUNDS ?? '7', 10);
const CONVERGENCE_SCORE = 95;
const AGENT_TIMEOUT_MS = parseInt(process.env.DERIVE_AGENT_TIMEOUT_MS ?? '300000', 10); /* 5 min per agent call */

/** Fallback models for rate-limit recovery. */
const FALLBACKS = {
  'claude-opus-4-6': 'claude-sonnet-4-5',
  'claude-sonnet-4-5': 'claude-haiku-4-5',
  'gpt-5.3-codex': 'o3',
  'gemini-3-pro-preview': 'gemini-3-flash-preview',
};

/** Check whether agent output indicates a rate-limit / quota error. */
function isQuotaError(stderr) {
  const lower = (stderr || '').toLowerCase();
  return lower.includes('rate limit') || lower.includes('429')
    || lower.includes('quota') || lower.includes('too many requests')
    || lower.includes('usage limit') || lower.includes('capacity')
    || lower.includes('terminalquotaerror') || lower.includes('exhausted');
}

/** Check whether a failure looks transient (network glitch, server error). */
function isTransientError(stderr, timedOut) {
  if (timedOut) return true;
  const lower = (stderr || '').toLowerCase();
  return lower.includes('econnreset') || lower.includes('econnrefused')
    || lower.includes('etimedout') || lower.includes('epipe')
    || lower.includes('socket hang up') || lower.includes('network')
    || lower.includes('stream disconnected')
    || lower.includes('500') || lower.includes('502')
    || lower.includes('503') || lower.includes('529')
    || lower.includes('overloaded') || lower.includes('internal server error')
    || lower.includes('service unavailable') || lower.includes('bad gateway');
}

/** Sleep for a random duration between 60 s and 90 s. */
function transientBackoff() {
  const ms = 60_000 + Math.floor(Math.random() * 30_000);
  log(`[derive] Waiting ${(ms / 1000).toFixed(0)}s before retry...`);
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Default output-format instructions for MathArena math competition problems.
 * Can be overridden via the DERIVE_OUTPUT_INSTRUCTIONS env var to support
 * other benchmark modes or custom Derive workflows.
 */
const DEFAULT_OUTPUT_INSTRUCTIONS = [
  'OUTPUT FORMAT REQUIREMENTS:',
  '- Write your complete solution in LaTeX format (document body only, no \\documentclass or preamble).',
  '- Use proper LaTeX math environments: $...$ for inline, \\[...\\] or equation/align for display math.',
  '- Structure your solution with clear steps using \\textbf{Step N:} or \\paragraph{} headings.',
  '- Show all mathematical work explicitly — do NOT delegate computation to code or tools.',
  '',
  'CRITICAL — FINAL ANSWER FORMAT:',
  '1. Fully simplify your final answer to a single integer before boxing it.',
  '2. Your response MUST end with exactly one \\boxed{} containing your final answer.',
  '3. Inside \\boxed{}, write ONLY the bare integer — fully evaluated and simplified.',
  '   No variables, no "N=", no units, no text, no formatting commands, no expressions.',
  '   Correct: \\boxed{42}',
  '   Wrong: \\boxed{N = 42}  Wrong: \\boxed{42 \\text{ mod } 1000}  Wrong: \\boxed{\\textbf{42}}',
  '   Wrong: \\boxed{7 \\cdot 6}  Wrong: \\boxed{\\frac{84}{2}}',
  '4. Do NOT use \\boxed{} anywhere else in your solution for intermediate results.',
  '5. The final \\boxed{} must be the very last mathematical content in your response.',
].join('\n');

const OUTPUT_INSTRUCTIONS = process.env.DERIVE_OUTPUT_INSTRUCTIONS ?? DEFAULT_OUTPUT_INSTRUCTIONS;

/* Track all spawned child processes for cleanup on Ctrl+C. */
const activeChildren = new Set();
function cleanup() {
  for (const child of activeChildren) {
    try { child.kill('SIGTERM'); } catch {}
  }
  process.exit(1);
}
process.on('SIGINT', cleanup);
process.on('SIGTERM', cleanup);

/** Read all of stdin. */
function readStdin() {
  return new Promise((resolve) => {
    let data = '';
    process.stdin.setEncoding('utf-8');
    process.stdin.on('data', (chunk) => { data += chunk; });
    process.stdin.on('end', () => resolve(data));
  });
}

/** Log a status line to stderr. */
function log(msg) {
  process.stderr.write(`${msg}\n`);
}

function claudeArgs(model) {
  return [
    '--print', '--output-format', 'json',
    '--model', model,
    '--strict-mcp-config',
  ];
}

function codexArgs(model) {
  return [
    'exec', '--json', '--sandbox', 'read-only', '--skip-git-repo-check',
    '-m', model, '-C', '/tmp',
  ];
}

function geminiArgs(model, prompt) {
  return ['-p', prompt, '--output-format', 'json', '-m', model];
}

/* Create clean scratch dirs for each role to avoid Gemini scanning /tmp
   and emitting thousands of EACCES warnings for systemd-private-* dirs.
   Separate dirs prevent session/state conflicts between agents. */
const SCRATCH_ROOT = `/tmp/derive-bench-${process.pid}`;
const SCRATCH_DIRS = {
  claude: `${SCRATCH_ROOT}/claude`,
  codex: `${SCRATCH_ROOT}/codex`,
  gemini: `${SCRATCH_ROOT}/gemini`,
  orchestrator: `${SCRATCH_ROOT}/orchestrator`,
};
for (const dir of Object.values(SCRATCH_DIRS)) mkdirSync(dir, { recursive: true });

/* Gemini CLI writes runtime/chat state to ~/.gemini/tmp/<project_hash>/...
   which may be unwritable in constrained benchmark environments. Give it an
   isolated writable home under our scratch root and seed required auth files. */
const GEMINI_HOME = `${SCRATCH_ROOT}/gemini-home`;

function seedGeminiHome() {
  const sourceHome = process.env.GEMINI_CLI_HOME || homedir();
  if (!sourceHome) return;

  const sourceGeminiDir = join(sourceHome, '.gemini');
  const targetGeminiDir = join(GEMINI_HOME, '.gemini');
  mkdirSync(targetGeminiDir, { recursive: true });
  mkdirSync(join(targetGeminiDir, 'tmp'), { recursive: true });

  const filesToCopy = [
    'settings.json',
    'oauth_creds.json',
    'google_accounts.json',
    'installation_id',
    'trustedFolders.json',
    'state.json',
  ];

  for (const file of filesToCopy) {
    const src = join(sourceGeminiDir, file);
    const dst = join(targetGeminiDir, file);
    if (!existsSync(src) || existsSync(dst)) continue;
    try {
      writeFileSync(dst, readFileSync(src));
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      log(`[derive] gemini: warning: failed to seed ${file}: ${msg}`);
    }
  }
}
seedGeminiHome();

/**
 * Spawn a CLI agent and return its output.
 * If the prompt is already embedded in `args` (e.g. gemini -p <prompt>),
 * set `promptInArgs: true` to skip piping via stdin.
 */
function runAgent(command, args, prompt, label, cwd, { promptInArgs = false } = {}) {
  return new Promise((resolve) => {
    const startTime = Date.now();
    let stdout = '';
    let stderr = '';
    const env = { ...process.env };
    delete env.CLAUDECODE;
    if (command === 'gemini') {
      env.GEMINI_CLI_HOME = GEMINI_HOME;
      env.HOME = GEMINI_HOME;
    }

    const child = spawn(command, args, {
      cwd: cwd || SCRATCH_DIRS[command] || SCRATCH_ROOT,
      env,
      stdio: ['pipe', 'pipe', 'pipe'],
    });
    activeChildren.add(child);

    /* Per-agent timeout to prevent indefinite hangs. */
    const timeout = setTimeout(() => {
      log(`[derive] ${label}: TIMEOUT after ${AGENT_TIMEOUT_MS / 1000}s — killing`);
      try { child.kill('SIGTERM'); } catch {}
      setTimeout(() => { try { child.kill('SIGKILL'); } catch {} }, 5000);
    }, AGENT_TIMEOUT_MS);

    child.stdin.on('error', () => { /* ignore EPIPE if child exits before stdin is written */ });
    if (promptInArgs) {
      child.stdin.end();
    } else {
      child.stdin.end(prompt);
    }
    child.stdout.on('data', (d) => { stdout += d.toString(); });
    child.stderr.on('data', (d) => { stderr += d.toString(); });
    child.on('close', (exitCode) => {
      clearTimeout(timeout);
      activeChildren.delete(child);
      resolve({ stdout, stderr, exitCode, durationMs: Date.now() - startTime, label });
    });
    child.on('error', (err) => {
      clearTimeout(timeout);
      activeChildren.delete(child);
      resolve({ stdout, stderr: err.message, exitCode: -1, durationMs: Date.now() - startTime, label });
    });
  });
}

/**
 * Run a Claude CLI call with automatic quota fallback and transient-error retry.
 * Quota errors → immediate failover to fallback model.
 * Transient errors → wait 60-90s then retry same model once.
 */
async function runClaude(prompt, label, model = COMBINER_MODEL) {
  const cwd = SCRATCH_DIRS.orchestrator;
  let result = await runAgent('claude', claudeArgs(model), prompt, label, cwd);
  if (result.exitCode !== 0) {
    const fallback = FALLBACKS[model];
    if (fallback && fallback !== model && isQuotaError(result.stderr)) {
      log(`[derive] ${label}: ${model} quota exceeded — retrying with ${fallback}`);
      return runAgent('claude', claudeArgs(fallback), prompt, label, cwd);
    }
    if (isTransientError(result.stderr, false)) {
      log(`[derive] ${label}: Transient error (model=${model}): ${result.stderr.slice(0, 200)}`);
      await transientBackoff();
      log(`[derive] ${label}: Retrying after transient error...`);
      result = await runAgent('claude', claudeArgs(model), prompt, label, cwd);
    }
  }
  return result;
}

/**
 * Run a generic agent with automatic quota fallback and transient-error retry.
 * Quota errors → immediate failover to fallback model.
 * Transient errors → wait 60-90s then retry same model once.
 *
 * @param {object} opts
 * @param {boolean} opts.promptInArgs  If true, prompt is in the args (e.g. gemini -p).
 * @param {function} opts.rebuildArgs  Called with (fallbackModel, prompt) to rebuild args on fallback.
 */
async function runAgentWithFallback(command, args, prompt, label, model, opts = {}) {
  const { promptInArgs = false, rebuildArgs } = opts;
  let result = await runAgent(command, args, prompt, label, undefined, { promptInArgs });
  if (result.exitCode !== 0) {
    const fallback = FALLBACKS[model];
    if (fallback && fallback !== model && isQuotaError(result.stderr)) {
      log(`[derive] ${label}: ${model} quota exceeded — retrying with ${fallback}`);
      const newArgs = rebuildArgs ? rebuildArgs(fallback, prompt) : args.map(a => a === model ? fallback : a);
      return runAgent(command, newArgs, prompt, label, undefined, { promptInArgs });
    }
    if (isTransientError(result.stderr, false)) {
      log(`[derive] ${label}: Transient error (model=${model}): ${result.stderr.slice(0, 200)}`);
      await transientBackoff();
      log(`[derive] ${label}: Retrying after transient error...`);
      result = await runAgent(command, args, prompt, label, undefined, { promptInArgs });
    }
  }
  return result;
}

/** Extract text content from agent output. */
function extractContent(result) {
  const raw = result.stdout.trim();
  if (!raw) {
    if (result.exitCode !== 0) {
      log(`[derive]   extractContent: stdout empty, exit=${result.exitCode}, stderr=${(result.stderr || '').length} chars`);
      log(`[derive]   stderr preview: ${(result.stderr || '').slice(0, 500).replace(/\n/g, ' | ')}`);
      /* Agent failed with no stdout — return marker so the filter excludes it.
       * Never return stderr as content; it's CLI noise, not a solution. */
      return '(no output)';
    }
    return '(no output)';
  }
  try {
    const parsed = JSON.parse(raw);
    if (parsed.result) return parsed.result;
    if (parsed.response) return parsed.response;
    if (parsed.text) return parsed.text;
    if (parsed.content) return parsed.content;
  } catch { /* not JSON */ }
  const lines = raw.split('\n');
  for (let i = lines.length - 1; i >= 0; i--) {
    try {
      const obj = JSON.parse(lines[i]);
      if (obj.item?.type === 'agent_message' && obj.item?.text) return obj.item.text;
    } catch { /* skip */ }
  }
  return raw;
}

/** Extract JSON from text, stripping markdown fences. */
function extractJson(text) {
  const trimmed = text.trim();
  const fenceMatch = trimmed.match(/```(?:json)?\s*([\s\S]*?)```/);
  const candidate = fenceMatch ? fenceMatch[1].trim() : trimmed;
  try { return JSON.parse(candidate); } catch {}
  const braceStart = candidate.indexOf('{');
  const braceEnd = candidate.lastIndexOf('}');
  if (braceStart >= 0 && braceEnd > braceStart) {
    try { return JSON.parse(candidate.slice(braceStart, braceEnd + 1)); } catch {}
  }
  return null;
}

/** Build critique prompt. */
function buildCritiquePrompt(problem, agentResponses, combinedResponse, round) {
  const agentNames = Object.keys(agentResponses);
  const targetNames = [...agentNames, 'combined'];
  let prompt = `You are a mathematical reviewer. Score each solution to the following problem.\n\n`;
  prompt += `<PROBLEM>\n${problem}\n</PROBLEM>\n\n`;
  for (const name of agentNames) {
    prompt += `<SOLUTION agent="${name}">\n${agentResponses[name]}\n</SOLUTION>\n\n`;
  }
  prompt += `<COMBINED_SOLUTION>\n${combinedResponse}\n</COMBINED_SOLUTION>\n\n`;
  prompt += `CRITIQUE ROUND: ${round}\n\n`;
  prompt += `INSTRUCTIONS:\n`;
  prompt += `1. Assess each solution for mathematical correctness, completeness, and rigor.\n`;
  prompt += `2. Score each on 0-100. Use 100 only when the solution is mathematically perfect.\n`;
  prompt += `3. Provide concise, actionable feedback for each.\n`;
  prompt += `4. Apply identical rigor to ALL solutions. Do NOT inflate or deflate any.\n\n`;
  const exampleObj = {};
  for (const name of targetNames) {
    exampleObj[name] = { score: '<0-100>', feedback: '<concise feedback>' };
  }
  prompt += `Respond with ONLY a JSON object mapping each target to {score, feedback}:\n`;
  prompt += JSON.stringify(exampleObj, null, 2) + '\n';
  return prompt;
}

/** Build refinement prompt. */
function buildRefinementPrompt(problem, agentName, previousResponse, scores, consensusResponse, round) {
  let prompt = `You are an expert mathematician refining your solution.\n\n`;
  prompt += `<PROBLEM>\n${problem}\n</PROBLEM>\n\n`;
  prompt += `YOUR PREVIOUS RESPONSE:\n${previousResponse}\n\n`;
  if (scores.length > 0) {
    prompt += `SCORES YOUR RESPONSE RECEIVED:\n`;
    for (const s of scores) prompt += `  ${s.reviewer}: ${s.score}/100 — ${s.feedback}\n`;
    prompt += '\n';
  }
  prompt += `CURRENT CONSENSUS RESPONSE:\n${consensusResponse}\n\n`;
  prompt += `REFINEMENT ROUND: ${round + 1}\n\n`;
  prompt += `Produce a solution that is BETTER than the consensus response. `;
  prompt += `Correct any errors, fill any gaps, and improve rigor. `;
  prompt += `Show all mathematical work explicitly in LaTeX format — do not use code execution.`;
  prompt += OUTPUT_INSTRUCTIONS;
  return prompt;
}

/** Build combine prompt. */
function buildCombinePrompt(problem, agentResponses) {
  let prompt = `You are combining independent solutions to a math problem. `;
  prompt += `Review all solutions, identify the correct approach, and produce a single authoritative solution.\n\n`;
  prompt += `<PROBLEM>\n${problem}\n</PROBLEM>\n\n`;
  for (const [name, content] of Object.entries(agentResponses)) {
    prompt += `<SOLUTION agent="${name}">\n${content}\n</SOLUTION>\n\n`;
  }
  prompt += `Synthesize the best solution. If solutions disagree, carefully verify each approach. `;
  prompt += `Show all mathematical work explicitly in LaTeX format.`;
  prompt += OUTPUT_INSTRUCTIONS;
  return prompt;
}

async function main() {
  const problem = (await readStdin()).trim();
  if (!problem) { log('[derive] No problem text provided on stdin'); process.exit(1); }

  const systemPrompt = `You are an expert mathematician. Solve this problem step by step with rigorous mathematical reasoning. Write your full solution showing all work — do not use code execution or external tools.\n\n${problem}\n${OUTPUT_INSTRUCTIONS}`;
  const startTime = Date.now();
  const agentNames = ['claude', 'codex', 'gemini'];

  /* Full trace for detailed output. */
  const trace = {
    initialAgents: {},
    rounds: [],
  };

  log(`[derive] Starting 3 agents in parallel...`);

  /* Run all three agents in parallel (with automatic fallback on quota errors).
   * Gemini receives its prompt via -p (CLI arg) instead of stdin because
   * the gemini CLI requires -p for non-interactive headless mode. */
  const [claudeResult, codexResult, geminiResult] = await Promise.all([
    runAgentWithFallback('claude', claudeArgs(CLAUDE_MODEL), systemPrompt, 'claude', CLAUDE_MODEL),
    runAgentWithFallback('codex', codexArgs(CODEX_MODEL), systemPrompt, 'codex', CODEX_MODEL),
    runAgentWithFallback('gemini', geminiArgs(GEMINI_MODEL, systemPrompt), systemPrompt, 'gemini', GEMINI_MODEL, {
      promptInArgs: true,
      rebuildArgs: (fallback, p) => geminiArgs(fallback, p),
    }),
  ]);

  const agentResponses = {
    claude: extractContent(claudeResult),
    codex: extractContent(codexResult),
    gemini: extractContent(geminiResult),
  };

  const agentResults = { claude: claudeResult, codex: codexResult, gemini: geminiResult };
  for (const name of agentNames) {
    const r = agentResults[name];
    const ok = r.exitCode === 0 ? 'OK' : 'FAIL';
    log(`[derive] ${name}: ${ok} (${(r.durationMs / 1000).toFixed(1)}s, ${agentResponses[name].length} chars)`);
    if (r.exitCode !== 0) {
      log(`[derive]   ${name} stdout: ${(r.stdout || '').length} chars — ${(r.stdout || '').slice(0, 500)}`);
      log(`[derive]   ${name} stderr: ${(r.stderr || '').length} chars — ${(r.stderr || '').slice(0, 500)}`);
    }
    trace.initialAgents[name] = {
      exitCode: r.exitCode,
      durationMs: r.durationMs,
      contentLength: agentResponses[name].length,
      content: agentResponses[name],
      stderr: r.stderr,
    };
  }

  /* Filter out agents that failed to produce meaningful output.
   * Use content quality, not exit code — some CLIs (e.g. gemini) exit non-zero
   * even when they produce valid output. */
  const activeAgents = agentNames.filter((name) => {
    const content = agentResponses[name];
    const r = agentResults[name];
    /* Basic length / error checks. */
    if (content.length < 50 || content.startsWith('(no output)') || content.startsWith('Error:')) {
      log(`[derive] ${name}: excluded from critique loop (exit=${r.exitCode}, len=${content.length})`);
      return false;
    }
    /* If agent failed AND content has no math indicators, it's noise. */
    if (r.exitCode !== 0) {
      const hasMath = content.includes('\\boxed') || content.includes('\\begin')
        || content.includes('\\frac') || content.includes('\\text')
        || (content.includes('$') && content.includes('='));
      if (!hasMath) {
        log(`[derive] ${name}: excluded — non-zero exit and no math content (exit=${r.exitCode}, len=${content.length})`);
        log(`[derive]   content preview: ${content.slice(0, 300)}`);
        return false;
      }
    }
    return true;
  });
  if (activeAgents.length === 0) {
    log(`[derive] All agents failed — no solutions to combine`);
    process.exit(1);
  }
  log(`[derive] Active agents for critique: ${activeAgents.join(', ')}`);

  /* Build the responses map for only active agents. */
  const activeResponses = {};
  for (const name of activeAgents) activeResponses[name] = agentResponses[name];

  /* === Critique Loop === */
  let combinedContent = '';
  let converged = false;
  let critiqueRounds = 0;

  for (let round = 0; round < MAX_CRITIQUE_ROUNDS; round++) {
    critiqueRounds = round + 1;
    const roundTrace = { round: round + 1, combine: {}, critiques: {}, refinements: {} };

    /* --- Combine --- */
    const combinePrompt = buildCombinePrompt(problem, activeResponses);
    log(`[derive] Round ${round + 1}: Combining via ${COMBINER_MODEL}...`);
    const combineStart = Date.now();
    const combineResult = await runClaude(combinePrompt, `combine-r${round + 1}`);
    combinedContent = extractContent(combineResult);
    roundTrace.combine = {
      durationMs: Date.now() - combineStart,
      contentLength: combinedContent.length,
      content: combinedContent,
    };
    log(`[derive] Round ${round + 1}: Combined ${combinedContent.length} chars (${((Date.now() - combineStart) / 1000).toFixed(1)}s)`);

    /* --- Critique --- */
    log(`[derive] Round ${round + 1}: Critiquing (${activeAgents.length} reviewers in parallel)...`);
    const critiquePrompt = buildCritiquePrompt(problem, activeResponses, combinedContent, round + 1);
    const critiqueStart = Date.now();
    const critiqueResults = await Promise.all(
      activeAgents.map((name) => runClaude(critiquePrompt, `critique-${name}-r${round + 1}`))
    );
    const critiqueDuration = Date.now() - critiqueStart;

    const critiques = {};
    const combinedScores = [];
    for (let i = 0; i < activeAgents.length; i++) {
      const name = activeAgents[i];
      const content = extractContent(critiqueResults[i]);
      const parsed = extractJson(content);
      const critiqueEntry = {
        durationMs: critiqueResults[i].durationMs,
        rawContent: content,
        parsed: null,
        parseFailed: true,
      };

      /* Coerce scores from strings to numbers (models sometimes quote them). */
      if (parsed && parsed.combined) {
        for (const key of Object.keys(parsed)) {
          if (parsed[key] && parsed[key].score !== undefined) {
            parsed[key].score = Number(parsed[key].score);
          }
        }
      }
      if (parsed && parsed.combined && typeof parsed.combined.score === 'number' && !isNaN(parsed.combined.score)) {
        critiques[name] = parsed;
        combinedScores.push(parsed.combined.score);
        critiqueEntry.parsed = parsed;
        critiqueEntry.parseFailed = false;

        /* Log all scores from this reviewer. */
        const parts = [];
        for (const target of [...activeAgents, 'combined']) {
          if (parsed[target]) parts.push(`${target}=${parsed[target].score}`);
        }
        log(`[derive] Round ${round + 1}: ${name} scores: ${parts.join(', ')} (${(critiqueResults[i].durationMs / 1000).toFixed(1)}s)`);
      } else {
        log(`[derive] Round ${round + 1}: ${name} critique parse FAILED (${(critiqueResults[i].durationMs / 1000).toFixed(1)}s)`);
        log(`[derive]   raw critique (${content.length} chars): ${content.slice(0, 400)}`);
        if (parsed) log(`[derive]   parsed keys: ${JSON.stringify(Object.keys(parsed))}`);
      }
      roundTrace.critiques[name] = critiqueEntry;
    }
    roundTrace.critiqueDurationMs = critiqueDuration;
    roundTrace.combinedScores = combinedScores;
    roundTrace.combinedScoreMean = combinedScores.length > 0
      ? combinedScores.reduce((a, b) => a + b, 0) / combinedScores.length : null;

    if (combinedScores.length > 0) {
      const mean = roundTrace.combinedScoreMean.toFixed(1);
      log(`[derive] Round ${round + 1}: combined score mean=${mean}, scores=[${combinedScores.join(', ')}]`);
    }

    /* Check convergence (only after at least 1 full round). */
    if (round >= 1 && combinedScores.length > 0) {
      if (combinedScores.every((s) => s >= CONVERGENCE_SCORE)) {
        log(`[derive] Converged after ${round + 1} rounds (all scores >= ${CONVERGENCE_SCORE})`);
        converged = true;
        trace.rounds.push(roundTrace);
        break;
      }
    }

    /* --- Refinement (skip on last round) --- */
    if (round >= MAX_CRITIQUE_ROUNDS - 1) {
      log(`[derive] Max rounds (${MAX_CRITIQUE_ROUNDS}) reached`);
      trace.rounds.push(roundTrace);
      break;
    }

    log(`[derive] Round ${round + 1}: Refining (${activeAgents.length} agents in parallel)...`);
    const refineStart = Date.now();
    const refinementResults = await Promise.all(
      activeAgents.map((name) => {
        const scores = [];
        for (const [reviewer, critique] of Object.entries(critiques)) {
          if (critique[name] && typeof critique[name].score === 'number') {
            scores.push({ reviewer, score: critique[name].score, feedback: critique[name].feedback || '' });
          }
        }
        return runClaude(
          buildRefinementPrompt(problem, name, activeResponses[name], scores, combinedContent, round),
          `refine-${name}-r${round + 1}`,
        );
      })
    );

    for (let i = 0; i < activeAgents.length; i++) {
      const name = activeAgents[i];
      const refined = extractContent(refinementResults[i]);
      roundTrace.refinements[name] = {
        durationMs: refinementResults[i].durationMs,
        contentLength: refined.length,
        content: refined,
      };
      if (refined && refined.length > 50) {
        activeResponses[name] = refined;
        log(`[derive] Round ${round + 1}: ${name} refined (${refined.length} chars, ${(refinementResults[i].durationMs / 1000).toFixed(1)}s)`);
      }
    }
    roundTrace.refineDurationMs = Date.now() - refineStart;
    trace.rounds.push(roundTrace);
  }

  const totalTime = (Date.now() - startTime) / 1000;
  log(`[derive] Done: ${combinedContent.length} chars, ${critiqueRounds} rounds, converged=${converged} (${totalTime.toFixed(1)}s)`);

  /* Extract first-round per-agent scores (raw agent performance before refinement). */
  const firstRoundScores = {};
  if (trace.rounds.length > 0) {
    const r1 = trace.rounds[0];
    for (const target of [...activeAgents, 'combined']) {
      const scores = [];
      for (const [reviewer, crit] of Object.entries(r1.critiques)) {
        if (crit.parsed && crit.parsed[target] && typeof crit.parsed[target].score === 'number') {
          scores.push(crit.parsed[target].score);
        }
      }
      if (scores.length > 0) {
        firstRoundScores[target] = {
          scores,
          mean: scores.reduce((a, b) => a + b, 0) / scores.length,
        };
      }
    }
    const scoreParts = Object.entries(firstRoundScores)
      .map(([t, s]) => `${t}=${s.mean.toFixed(0)}`)
      .join(', ');
    log(`[derive] First-round scores: ${scoreParts}`);
  }

  /* Output full result JSON. */
  const output = {
    response: combinedContent,
    cost: 0,
    time: totalTime,
    converged,
    critiqueRounds,
    maxRounds: MAX_CRITIQUE_ROUNDS,
    firstRoundScores,
    agents: {},
    trace,
  };
  for (const name of agentNames) {
    output.agents[name] = {
      exit: agentResults[name].exitCode,
      time: agentResults[name].durationMs / 1000,
      chars: agentResponses[name].length,
    };
  }
  console.log(JSON.stringify(output));
}

main().catch((err) => {
  console.error(`[derive] Fatal error: ${err.message}`);
  process.exit(1);
});
