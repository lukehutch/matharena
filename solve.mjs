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
import { mkdirSync } from 'fs';

const CLAUDE_MODEL = process.env.DERIVE_CLAUDE_MODEL ?? 'claude-opus-4-6';
const CODEX_MODEL = process.env.DERIVE_CODEX_MODEL ?? 'gpt-5.3-codex';
const GEMINI_MODEL = process.env.DERIVE_GEMINI_MODEL ?? 'gemini-3-pro-preview';
const COMBINER_MODEL = process.env.DERIVE_ORCHESTRATOR_MODEL_FAST ?? 'claude-sonnet-4-5';
const MAX_CRITIQUE_ROUNDS = parseInt(process.env.DERIVE_MAX_CRITIQUE_ROUNDS ?? '7', 10);
const CONVERGENCE_SCORE = 95;
const AGENT_TIMEOUT_MS = parseInt(process.env.DERIVE_AGENT_TIMEOUT_MS ?? '1500000', 10); /* 25 min per agent call */

/** Fallback models for rate-limit recovery. */
const FALLBACKS = {
  'claude-opus-4-6': 'claude-sonnet-4-5',
  'claude-sonnet-4-5': 'claude-haiku-4-5',
  'gpt-5.3-codex': 'gpt-5.1-codex-mini',
  'gemini-3-pro-preview': 'gemini-3-flash-preview',
};

/** Max supported reasoning effort per Codex model. */
const CODEX_MAX_EFFORT = {
  'gpt-5.3-codex': 'xhigh',
  'gpt-5.1-codex-mini': 'high',
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
let cleaningUp = false;
function cleanup() {
  if (cleaningUp) {
    for (const child of activeChildren) {
      try { child.kill('SIGKILL'); } catch {}
    }
    process.exit(1);
  }
  cleaningUp = true;
  log(`[derive] Interrupted — killing ${activeChildren.size} child processes...`);
  for (const child of activeChildren) {
    try { child.kill('SIGTERM'); } catch {}
  }
  setTimeout(() => {
    for (const child of activeChildren) {
      try { child.kill('SIGKILL'); } catch {}
    }
    process.exit(1);
  }, 2000);
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
  const effort = CODEX_MAX_EFFORT[model] ?? 'high';
  return [
    'exec', '--json', '--sandbox', 'read-only', '--skip-git-repo-check',
    '-m', model, '-c', `model_reasoning_effort="${effort}"`,
    '-C', '/tmp',
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
      /* Do NOT set GEMINI_CLI_HOME or HOME — gemini CLI uses GEMINI_CLI_HOME
         (falling back to HOME) to find ~/.gemini/ for auth credentials.
         Overriding it breaks auth with stale/missing tokens. Session isolation
         is handled by the per-agent scratch dir used as cwd. */
      delete env.GEMINI_CLI_HOME;
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
      setTimeout(() => { try { child.kill('SIGKILL'); } catch {} }, 5000).unref();
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
      const serverMsg = (result.stdout + '\n' + result.stderr).trim().replace(/\n/g, ' | ');
      log(`[derive] ${label}: ${model} quota exceeded — falling back to ${fallback}. Server: ${serverMsg}`);
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
      const serverMsg = (result.stdout + '\n' + result.stderr).trim().replace(/\n/g, ' | ');
      log(`[derive] ${label}: ${model} quota exceeded — falling back to ${fallback}. Server: ${serverMsg}`);
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

/** Extract the last \\boxed{...} answer from text. */
function extractBoxedAnswer(text) {
  if (!text) return null;
  let latest = null;
  let searchFrom = 0;
  const marker = '\\boxed{';
  while (true) {
    const markerIdx = text.indexOf(marker, searchFrom);
    if (markerIdx === -1) break;
    const contentStart = markerIdx + marker.length;
    let depth = 1;
    let idx = contentStart;
    while (idx < text.length && depth > 0) {
      if (text[idx] === '{') depth++;
      else if (text[idx] === '}') depth--;
      idx++;
    }
    if (depth === 0) {
      latest = text.slice(contentStart, idx - 1).trim();
      searchFrom = idx;
    } else {
      searchFrom = markerIdx + marker.length;
    }
  }
  return latest;
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
  let prompt = `You are a rigorous mathematical reviewer. Your job is to independently verify each solution to the following problem using Chain-of-Verification.\n\n`;
  prompt += `<PROBLEM>\n${problem}\n</PROBLEM>\n\n`;
  for (const name of agentNames) {
    prompt += `<SOLUTION agent="${name}">\n${agentResponses[name]}\n</SOLUTION>\n\n`;
  }
  prompt += `<COMBINED_SOLUTION>\n${combinedResponse}\n</COMBINED_SOLUTION>\n\n`;
  prompt += `CRITIQUE ROUND: ${round}\n\n`;

  prompt += `You MUST follow the Chain-of-Verification protocol below. Complete each phase in order.\n\n`;

  prompt += `═══════════════════════════════════════════════════════════════\n`;
  prompt += `PHASE 1 — CRITIQUE PLAN & BASELINE VERIFICATION QUESTIONS\n`;
  prompt += `═══════════════════════════════════════════════════════════════\n\n`;
  prompt += `Before evaluating any solution, prepare your verification framework:\n\n`;
  prompt += `A. SOLVE INDEPENDENTLY: Work through the problem yourself from scratch. Show your complete solution with all steps justified. Arrive at your own answer.\n\n`;
  prompt += `B. GENERATE VERIFICATION QUESTIONS: Based on your independent solution, formulate specific verification questions that any correct solution must answer. For example:\n`;
  prompt += `   - "Does the solution correctly identify all cases? (list them)"\n`;
  prompt += `   - "Is the key formula/identity applied with valid preconditions?"\n`;
  prompt += `   - "Does the arithmetic in step X check out? (show independent calculation)"\n`;
  prompt += `   - "Is the counting argument exhaustive and non-overlapping?"\n`;
  prompt += `   - "Does the final answer match an independent verification method?"\n`;
  prompt += `   List 3-7 verification questions specific to this problem.\n\n`;
  prompt += `C. COMPARE EACH SOLUTION against your verification questions. For each solution, answer every question, tracing through the logical steps and computations.\n\n`;

  prompt += `ANTI-BIAS WARNING:\n`;
  prompt += `You MUST NOT be biased toward or against any particular solution based on its source, length, style, or confidence of presentation.\n`;
  prompt += `A longer or more detailed solution is NOT necessarily more correct.\n`;
  prompt += `A solution labeled "combined" is NOT necessarily better than individual solutions.\n`;
  prompt += `Judge ONLY on mathematical correctness. If a minority solution has the right answer and the majority is wrong, say so.\n\n`;

  prompt += `═══════════════════════════════════════════════════════════════\n`;
  prompt += `PHASE 2 — EVALUATE AND SCORE\n`;
  prompt += `═══════════════════════════════════════════════════════════════\n\n`;
  prompt += `For each solution, declare:\n`;
  prompt += `   - "correct": true/false — is the final answer correct and the reasoning sound?\n`;
  prompt += `   - "answer": the extracted \\boxed{} value\n`;
  prompt += `   - "score": 0-100 reflecting answer correctness:\n`;
  prompt += `     100 = correct final answer (even if working is imperfect or verbose)\n`;
  prompt += `     50-99 = approach is sound but you cannot fully verify the answer\n`;
  prompt += `     0-49 = final answer is wrong or contains a fatal mathematical error\n`;
  prompt += `   - "justification": detailed reasoning for the score — walk through which verification questions passed/failed, cite the exact step where an error occurs, and show the correct calculation.\n`;
  prompt += `   Do NOT penalize correct answers for stylistic issues or suboptimal working.\n\n`;

  prompt += `═══════════════════════════════════════════════════════════════\n`;
  prompt += `PHASE 3 — RECOMMENDATION\n`;
  prompt += `═══════════════════════════════════════════════════════════════\n\n`;
  prompt += `Based on your verification, make a recommendation:\n`;
  prompt += `   - "accept": true if the combined solution's final answer is correct and should be accepted as optimal. false if another round of refinement is needed.\n`;
  prompt += `   - "recommendation": a brief explanation. If accept=false, explain what specifically needs to be fixed or re-derived in the next round.\n\n`;

  const exampleObj = {};
  for (const name of targetNames) {
    exampleObj[name] = { answer: '<extracted boxed value>', correct: '<true/false>', score: '<0-100>', justification: '<your reasoning>' };
  }
  exampleObj['_recommendation'] = { accept: '<true/false>', recommendation: '<explanation>' };
  prompt += `Respond with ONLY a JSON object:\n`;
  prompt += JSON.stringify(exampleObj, null, 2) + '\n';
  return prompt;
}

/** Build refinement prompt. */
function buildRefinementPrompt(problem, agentName, previousResponse, scores, consensusResponse, round, recommendations = []) {
  let prompt = `You are an expert mathematician refining your solution.\n\n`;
  prompt += `<PROBLEM>\n${problem}\n</PROBLEM>\n\n`;
  prompt += `YOUR PREVIOUS RESPONSE:\n${previousResponse}\n\n`;
  if (scores.length > 0) {
    prompt += `REVIEWER ASSESSMENTS OF YOUR RESPONSE:\n`;
    for (const s of scores) {
      const correctStr = s.correct === true ? 'CORRECT' : s.correct === false ? 'INCORRECT' : 'UNKNOWN';
      prompt += `  ${s.reviewer}: ${correctStr} (${s.score}/100) — ${s.feedback}\n`;
    }
    prompt += '\n';
  }
  if (recommendations.length > 0) {
    prompt += `REVIEWER RECOMMENDATIONS:\n`;
    for (const r of recommendations) {
      const acceptStr = r.accept ? 'ACCEPT' : 'NEEDS REVISION';
      prompt += `  ${r.reviewer}: ${acceptStr} — ${r.recommendation}\n`;
    }
    prompt += '\n';
  }
  prompt += `CURRENT CONSENSUS RESPONSE:\n${consensusResponse}\n\n`;
  prompt += `REFINEMENT ROUND: ${round + 1}\n\n`;
  prompt += `Produce a solution that is BETTER than the consensus response. `;
  prompt += `Correct any errors, fill any gaps, and improve rigor. `;
  prompt += `For every claim and intermediate result, provide a detailed justification explaining WHY it is true. `;
  prompt += `If reviewers flagged an error, re-derive that step from scratch and show the correct computation. `;
  prompt += `Pay close attention to the reviewer recommendations above — they identify what specifically needs fixing. `;
  prompt += `Show all mathematical work explicitly in LaTeX format — do not use code execution.`;
  prompt += OUTPUT_INSTRUCTIONS;
  return prompt;
}

/** Build combine prompt. */
function buildCombinePrompt(problem, agentResponses, critiqueFeedback = null) {
  let prompt = `You are combining independent solutions to a math problem into the most correct consensus response possible.\n\n`;
  prompt += `<PROBLEM>\n${problem}\n</PROBLEM>\n\n`;
  for (const [name, content] of Object.entries(agentResponses)) {
    prompt += `<SOLUTION agent="${name}">\n${content}\n</SOLUTION>\n\n`;
  }
  if (critiqueFeedback) {
    prompt += `<REVIEWER_FEEDBACK>\n`;
    prompt += `The following reviewers independently solved the problem and then verified each solution:\n\n`;
    for (const [reviewer, feedback] of Object.entries(critiqueFeedback)) {
      prompt += `Reviewer "${reviewer}":\n`;
      for (const [target, critique] of Object.entries(feedback)) {
        if (critique && critique.justification) {
          const correctStr = critique.correct ? 'CORRECT' : 'INCORRECT';
          prompt += `  ${target}: ${correctStr} (answer=${critique.answer ?? '?'}, score=${critique.score ?? '?'}) — ${critique.justification}\n`;
        }
      }
      prompt += '\n';
    }
    prompt += `</REVIEWER_FEEDBACK>\n\n`;
  }
  prompt += `COMBINING INSTRUCTIONS:\n\n`;
  prompt += `1. IDENTIFY ALL DISTINCT ANSWERS across the solutions and reviewer feedback. List them.\n`;
  prompt += `2. For each distinct answer, trace the logical chain that leads to it. Identify the KEY STEP where solutions diverge — the specific formula, counting argument, or computation that differs.\n`;
  prompt += `3. For each divergence point, independently verify which approach is correct by re-deriving that step from first principles. Show your full working.\n`;
  prompt += `4. Do NOT assume the majority answer is correct. A single solution with correct reasoning outweighs multiple solutions with the same error. Judge on mathematical merit alone.\n`;
  prompt += `5. Do NOT favor any solution based on its source, label, length, or presentation style.\n`;
  prompt += `6. Produce a single authoritative solution with every step fully justified. Show all mathematical work explicitly in LaTeX format.\n\n`;
  prompt += OUTPUT_INSTRUCTIONS;
  return prompt;
}

async function main() {
  /* Accept question as CLI argument or via stdin. */
  const cliQuestion = process.argv.slice(2).join(' ').trim();
  const problem = cliQuestion || (await readStdin()).trim();
  if (!problem) { log('[derive] No problem text provided. Usage: node solve.mjs "question" or echo "question" | node solve.mjs'); process.exit(1); }

  const systemPrompt = `You are an expert mathematician. Solve this problem step by step with rigorous mathematical reasoning. Write your full solution showing all work — do not use code execution or external tools.

IMPORTANT — DETAILED JUSTIFICATION REQUIRED:
- For every claim, formula, or intermediate result, provide a detailed justification explaining WHY it is true.
- When applying a theorem, identity, or counting principle, state it explicitly and verify that its preconditions hold.
- When computing combinatorial quantities, probabilities, or sums, show the full enumeration or derivation — do not skip steps or assert results without proof.
- When there are multiple cases, enumerate all of them explicitly and verify each one.
- If you make a simplifying assumption, state it and justify why it is valid.
- Double-check your final answer by verifying it with an independent method (e.g. compute the same quantity a different way, check boundary cases, or verify with small examples).

${problem}
${OUTPUT_INSTRUCTIONS}`;
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
  let previousCritiques = null; /* Critique feedback from previous round, fed into combine. */

  for (let round = 0; round < MAX_CRITIQUE_ROUNDS; round++) {
    critiqueRounds = round + 1;
    const roundTrace = { round: round + 1, combine: {}, critiques: {}, refinements: {} };

    /* --- Combine (with previous round's critique feedback if available) --- */
    const combinePrompt = buildCombinePrompt(problem, activeResponses, previousCritiques);
    log(`[derive] Round ${round + 1}: Combining via ${COMBINER_MODEL}...`);
    const combineStart = Date.now();
    const combineResult = await runClaude(combinePrompt, `combine-r${round + 1}`);
    combinedContent = extractContent(combineResult);
    roundTrace.combine = {
      durationMs: Date.now() - combineStart,
      contentLength: combinedContent.length,
      content: combinedContent,
    };
    const combinedBoxed = extractBoxedAnswer(combinedContent);
    log(`[derive] Round ${round + 1}: Combined ${combinedContent.length} chars (${((Date.now() - combineStart) / 1000).toFixed(1)}s) answer=${combinedBoxed ?? '?'}`);

    /* Log answer agreement status. */
    const answerMap = {};
    for (const name of activeAgents) {
      answerMap[name] = extractBoxedAnswer(activeResponses[name]) ?? '?';
    }
    answerMap['combined'] = combinedBoxed ?? '?';
    const answerParts = Object.entries(answerMap).map(([k, v]) => `${k}=${v}`).join(', ');
    log(`[derive] Round ${round + 1}: Answers: ${answerParts}`);

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
    const combinedCorrectVotes = [];
    const acceptVotes = [];
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

      /* Coerce scores from strings to numbers and correct booleans. */
      if (parsed && parsed.combined) {
        for (const key of Object.keys(parsed)) {
          if (parsed[key]) {
            if (parsed[key].score !== undefined) parsed[key].score = Number(parsed[key].score);
            if (typeof parsed[key].correct === 'string') parsed[key].correct = parsed[key].correct === 'true';
            if (typeof parsed[key].accept === 'string') parsed[key].accept = parsed[key].accept === 'true';
          }
        }
      }
      if (parsed && parsed.combined && typeof parsed.combined.score === 'number' && !isNaN(parsed.combined.score)) {
        critiques[name] = parsed;
        combinedScores.push(parsed.combined.score);
        if (typeof parsed.combined.correct === 'boolean') {
          combinedCorrectVotes.push(parsed.combined.correct);
        }
        /* Extract Phase 3 recommendation. */
        const rec = parsed._recommendation;
        if (rec) {
          if (typeof rec.accept === 'string') rec.accept = rec.accept === 'true';
          if (typeof rec.accept === 'boolean') acceptVotes.push(rec.accept);
        }
        critiqueEntry.parsed = parsed;
        critiqueEntry.parseFailed = false;

        /* Log all scores and correctness from this reviewer. */
        const parts = [];
        for (const target of [...activeAgents, 'combined']) {
          if (parsed[target]) {
            const correctMark = parsed[target].correct === true ? 'Y' : parsed[target].correct === false ? 'N' : '?';
            parts.push(`${target}=${parsed[target].score}(${correctMark})`);
          }
        }
        const acceptMark = rec?.accept === true ? ' accept=Y' : rec?.accept === false ? ' accept=N' : '';
        log(`[derive] Round ${round + 1}: ${name} scores: ${parts.join(', ')}${acceptMark} (${(critiqueResults[i].durationMs / 1000).toFixed(1)}s)`);
      } else {
        log(`[derive] Round ${round + 1}: ${name} critique parse FAILED (${(critiqueResults[i].durationMs / 1000).toFixed(1)}s)`);
        log(`[derive]   raw critique (${content.length} chars): ${content.slice(0, 400)}`);
        if (parsed) log(`[derive]   parsed keys: ${JSON.stringify(Object.keys(parsed))}`);
      }
      roundTrace.critiques[name] = critiqueEntry;
    }
    roundTrace.critiqueDurationMs = critiqueDuration;
    roundTrace.combinedScores = combinedScores;
    roundTrace.combinedCorrectVotes = combinedCorrectVotes;
    roundTrace.acceptVotes = acceptVotes;
    roundTrace.combinedScoreMean = combinedScores.length > 0
      ? combinedScores.reduce((a, b) => a + b, 0) / combinedScores.length : null;

    if (combinedScores.length > 0) {
      const mean = roundTrace.combinedScoreMean.toFixed(1);
      const correctCount = combinedCorrectVotes.filter(Boolean).length;
      const acceptCount = acceptVotes.filter(Boolean).length;
      log(`[derive] Round ${round + 1}: combined score mean=${mean}, scores=[${combinedScores.join(', ')}], correct=${correctCount}/${combinedCorrectVotes.length}, accept=${acceptCount}/${acceptVotes.length}`);
    }

    /* Check convergence — can happen on any round including the first. */
    {
      /* Primary: majority of reviewers recommend accepting the combined answer.
       * Ties go to accept — the consensus result gets priority when votes are split. */
      const acceptCount = acceptVotes.filter(Boolean).length;
      const majorityAccept = acceptVotes.length > 0 && acceptCount * 2 >= acceptVotes.length;
      if (majorityAccept) {
        log(`[derive] Converged after ${round + 1} rounds (${acceptCount}/${acceptVotes.length} reviewers recommend accept)`);
        converged = true;
        trace.rounds.push(roundTrace);
        break;
      }

      /* Secondary: all reviewers confirm the combined answer is correct. */
      const allReviewersCorrect = combinedCorrectVotes.length > 0 && combinedCorrectVotes.every(Boolean);
      if (allReviewersCorrect) {
        log(`[derive] Converged after ${round + 1} rounds (all reviewers confirm combined answer correct)`);
        converged = true;
        trace.rounds.push(roundTrace);
        break;
      }

      /* Tertiary: answer agreement — all agents + combined agree on the boxed answer. */
      const allAnswers = activeAgents.map((n) => extractBoxedAnswer(activeResponses[n])).filter(Boolean);
      const combinedAnswer = extractBoxedAnswer(combinedContent);
      if (combinedAnswer) allAnswers.push(combinedAnswer);
      const uniqueAnswers = new Set(allAnswers);
      if (allAnswers.length >= 2 && uniqueAnswers.size === 1) {
        log(`[derive] Converged after ${round + 1} rounds (all answers agree: \\boxed{${combinedAnswer}})`);
        converged = true;
        trace.rounds.push(roundTrace);
        break;
      }

      /* Quaternary: score-based convergence. */
      if (combinedScores.length > 0 && combinedScores.every((s) => s >= CONVERGENCE_SCORE)) {
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

    /* Collect reviewer recommendations for refinement prompts. */
    const recommendations = [];
    for (const [reviewer, critique] of Object.entries(critiques)) {
      const rec = critique._recommendation;
      if (rec && typeof rec.recommendation === 'string') {
        recommendations.push({
          reviewer,
          accept: rec.accept === true,
          recommendation: rec.recommendation,
        });
      }
    }

    log(`[derive] Round ${round + 1}: Refining (${activeAgents.length} agents in parallel)...`);
    const refineStart = Date.now();
    const refinementResults = await Promise.all(
      activeAgents.map((name) => {
        const scores = [];
        for (const [reviewer, critique] of Object.entries(critiques)) {
          if (critique[name] && typeof critique[name].score === 'number') {
            scores.push({
              reviewer,
              score: critique[name].score,
              correct: critique[name].correct,
              feedback: critique[name].justification || critique[name].feedback || '',
            });
          }
        }
        return runClaude(
          buildRefinementPrompt(problem, name, activeResponses[name], scores, combinedContent, round, recommendations),
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

    /* Save critique feedback for next round's combine step. */
    previousCritiques = critiques;

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
