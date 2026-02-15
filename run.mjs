#!/usr/bin/env node
/**
 * Parallel AIME 2025 benchmark runner for Derive.
 *
 * - Fetches all 30 AIME 2025 problems from HuggingFace
 * - Runs solve.mjs for each in parallel (each in its own tmp directory)
 * - Writes per-problem detailed logs and solutions to output directory
 * - Captures per-round agent scores and times
 * - Prints live status updates to the terminal
 * - Produces aggregate statistics at the end
 * - Fetches the MathArena leaderboard and shows Derive's position
 *
 * Usage:
 *   node run-parallel.mjs [--max-parallel N] [--output-dir DIR] [--problems 1,2,3]
 */

import { spawn } from 'child_process';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'fs';
import { dirname, resolve } from 'path';
import { fileURLToPath } from 'url';
import https from 'https';

const __dirname = dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = resolve(__dirname, '..');
const SOLVE_SCRIPT = resolve(__dirname, 'solve.mjs');
const FETCH_PROBLEMS_SCRIPT = resolve(__dirname, '_fetch_problems.py');
const GRADE_BATCH_SCRIPT = resolve(__dirname, '_grade_batch.py');
const PROBLEMS_CACHE_PATH = resolve(PROJECT_ROOT, '.derive', 'benchmark', 'cache', 'aime_2025_problems.json');

/* Parse CLI args. */
function getArg(name, fallback) {
  const idx = process.argv.indexOf(`--${name}`);
  return idx >= 0 && process.argv[idx + 1] ? process.argv[idx + 1] : fallback;
}
const MAX_PARALLEL = parseInt(getArg('max-parallel', '2'), 10);
const TIMEOUT_SEC = parseInt(getArg('timeout', '600'), 10);
const OUTPUT_DIR = resolve(getArg('output-dir', resolve(PROJECT_ROOT, '.derive', 'benchmark', 'outputs', 'aime_2025')));
const PROBLEM_FILTER = getArg('problems', null);
const problemFilter = PROBLEM_FILTER ? PROBLEM_FILTER.split(',').map(Number) : null;

/* ── Helpers ─────────────────────────────────────────────────────── */

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function runCommand(command, args, options = {}) {
  const { cwd = __dirname, env = process.env, timeoutMs = 30_000, stdinText } = options;
  return new Promise((resolve, reject) => {
    let stdout = '';
    let stderr = '';

    const child = spawn(command, args, {
      cwd,
      env,
      stdio: ['pipe', 'pipe', 'pipe'],
    });
    activeChildren.add(child);

    child.stdout.on('data', (chunk) => { stdout += chunk.toString(); });
    child.stderr.on('data', (chunk) => { stderr += chunk.toString(); });
    child.stdin.on('error', () => { /* ignore EPIPE if process exits early */ });

    let timeout = null;
    if (timeoutMs > 0) {
      timeout = setTimeout(() => {
        try { child.kill('SIGKILL'); } catch {}
      }, timeoutMs);
    }

    child.on('error', (err) => {
      if (timeout) clearTimeout(timeout);
      activeChildren.delete(child);
      reject(err);
    });

    child.on('close', (code, signal) => {
      if (timeout) clearTimeout(timeout);
      activeChildren.delete(child);
      resolve({
        exitCode: code ?? -1,
        signal: signal ?? null,
        stdout,
        stderr,
      });
    });

    if (typeof stdinText === 'string') {
      child.stdin.end(stdinText);
    } else {
      child.stdin.end();
    }
  });
}

async function runUvPython(scriptPath, { timeoutMs = 30_000, stdinText } = {}) {
  return runCommand('uv', ['run', 'python', scriptPath], { cwd: __dirname, timeoutMs, stdinText });
}

/** Fetch AIME 2025 problems from HuggingFace via matharena Python env. */
async function fetchProblems() {
  const maxAttempts = 3;
  let lastErr = null;

  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      const result = await runUvPython(FETCH_PROBLEMS_SCRIPT, { timeoutMs: 45_000 });
      if (result.exitCode !== 0) {
        throw new Error(result.stderr.trim() || `uv python exited ${result.exitCode}`);
      }
      const parsed = JSON.parse(result.stdout.trim());
      if (!Array.isArray(parsed)) {
        throw new Error('Problem fetch result is not an array');
      }

      mkdirSync(resolve(PROJECT_ROOT, '.derive', 'benchmark', 'cache'), { recursive: true });
      writeFileSync(PROBLEMS_CACHE_PATH, JSON.stringify(parsed, null, 2));
      return parsed;
    } catch (err) {
      lastErr = err;
      if (attempt < maxAttempts) {
        await sleep(500 * attempt);
      }
    }
  }

  if (existsSync(PROBLEMS_CACHE_PATH)) {
    process.stderr.write(`[benchmark] Problem fetch failed; using cached dataset at ${PROBLEMS_CACHE_PATH}\n`);
    return JSON.parse(readFileSync(PROBLEMS_CACHE_PATH, 'utf-8'));
  }

  throw lastErr ?? new Error('Failed to fetch problems');
}

/** Fetch a URL as text, following redirects (up to 5 hops). */
function fetchUrl(url, maxRedirects = 5, timeoutMs = 15_000) {
  return new Promise((resolve, reject) => {
    const req = https.get(url, (res) => {
      if ((res.statusCode === 301 || res.statusCode === 302) && res.headers.location && maxRedirects > 0) {
        const next = res.headers.location.startsWith('http')
          ? res.headers.location
          : new URL(res.headers.location, url).href;
        res.resume();
        return fetchUrl(next, maxRedirects - 1, timeoutMs).then(resolve, reject);
      }
      if (res.statusCode !== 200) {
        const bodyChunks = [];
        res.on('data', (chunk) => bodyChunks.push(chunk));
        res.on('end', () => {
          const bodyPreview = Buffer.concat(bodyChunks).toString('utf-8').slice(0, 200);
          reject(new Error(`HTTP ${res.statusCode}: ${bodyPreview}`));
        });
        return;
      }
      let data = '';
      res.on('data', (chunk) => { data += chunk; });
      res.on('end', () => resolve(data));
    });

    req.setTimeout(timeoutMs, () => {
      req.destroy(new Error(`Request timed out after ${timeoutMs}ms`));
    });
    req.on('error', reject);
  });
}

async function fetchUrlWithRetries(url, maxAttempts = 3) {
  let lastErr = null;
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      return await fetchUrl(url, 5, 15_000);
    } catch (err) {
      lastErr = err;
      if (attempt < maxAttempts) {
        await sleep(400 * attempt);
      }
    }
  }
  throw lastErr ?? new Error(`Failed to fetch URL: ${url}`);
}

/**
 * Grade a batch of (response, gold_answer) pairs using the matharena grader.
 * Returns array of { extracted, correct, warning } for each input.
 */
async function gradeResponses(items) {
  if (items.length === 0) return [];
  try {
    const result = await runUvPython(GRADE_BATCH_SCRIPT, {
      timeoutMs: 90_000,
      stdinText: JSON.stringify(items),
    });
    if (result.exitCode !== 0) {
      throw new Error(result.stderr.trim() || `uv python exited ${result.exitCode}`);
    }
    const parsed = JSON.parse(result.stdout.trim());
    if (!Array.isArray(parsed)) {
      throw new Error('grading result is not an array');
    }
    return parsed;
  } catch {
    return items.map(() => ({ extracted: null, correct: false, warning: 'grading failed' }));
  }
}

/** Extract the answer from a \\boxed{...} expression. */
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
      const ch = text[idx];
      if (ch === '{') depth++;
      else if (ch === '}') depth--;
      idx++;
    }
    if (depth === 0) {
      latest = text.slice(contentStart, idx - 1).trim();
      searchFrom = idx;
    } else {
      /* Unbalanced braces; skip this match and continue scanning. */
      searchFrom = markerIdx + marker.length;
    }
  }

  return latest;
}

/** Pad string to width. */
const pad = (s, w) => String(s).padEnd(w);
const rpad = (s, w) => String(s).padStart(w);

/** ANSI colors for terminal output. */
const C = {
  reset: '\x1b[0m',
  bold: '\x1b[1m',
  dim: '\x1b[2m',
  green: '\x1b[32m',
  red: '\x1b[31m',
  yellow: '\x1b[33m',
  cyan: '\x1b[36m',
  magenta: '\x1b[35m',
};

/* ── Process cleanup on Ctrl+C ──────────────────────────────────── */

const activeChildren = new Set();
let cancelled = false;
let cleaningUp = false;
function cleanup() {
  if (cleaningUp) {
    for (const child of activeChildren) {
      try { child.kill('SIGKILL'); } catch {}
    }
    process.exit(1);
  }
  cleaningUp = true;
  cancelled = true;
  process.stderr.write(`\n${C.yellow}Interrupted — killing ${activeChildren.size} child processes, cancelling remaining tasks...${C.reset}\n`);
  for (const child of activeChildren) {
    try { child.kill('SIGTERM'); } catch {}
  }
  /* Give children a moment to die, then force-kill. */
  setTimeout(() => {
    for (const child of activeChildren) {
      try { child.kill('SIGKILL'); } catch {}
    }
    process.exit(1);
  }, 2000);
}
process.on('SIGINT', cleanup);
process.on('SIGTERM', cleanup);

/* ── Per-problem solver ──────────────────────────────────────────── */

/** Track active problems for live status. */
const status = {};

function logStatus(idx, msg, color = '') {
  const ts = new Date().toISOString().slice(11, 19);
  status[idx] = msg;
  process.stderr.write(`${C.dim}[${ts}]${C.reset} ${C.cyan}P${String(idx).padStart(2, '0')}${C.reset} ${color}${msg}${color ? C.reset : ''}\n`);
}

/** Run solve.mjs for a single problem. */
function solveProblem(problem, tmpDir) {
  return new Promise((resolve) => {
    const startTime = Date.now();
    /* Only include the AIME-specific constraint here — solve.mjs appends
       its own OUTPUT_INSTRUCTIONS with full \\boxed{} formatting rules. */
    const prompt = `The answer is an integer between 0 and 999 inclusive.\n\n${problem.problem}`;

    let stdout = '';
    let stderrLog = '';
    const env = { ...process.env };
    delete env.CLAUDECODE;

    const child = spawn('node', [SOLVE_SCRIPT], {
      cwd: tmpDir,
      env,
      stdio: ['pipe', 'pipe', 'pipe'],
    });
    activeChildren.add(child);

    /* Per-problem timeout. */
    const timeoutHandle = setTimeout(() => {
      logStatus(problem.idx, `TIMEOUT after ${TIMEOUT_SEC}s — killing`, C.red);
      try { child.kill('SIGTERM'); } catch {}
      setTimeout(() => { try { child.kill('SIGKILL'); } catch {} }, 5000).unref();
    }, TIMEOUT_SEC * 1000);

    child.stdin.end(prompt);
    child.stdout.on('data', (d) => { stdout += d.toString(); });
    child.stderr.on('data', (d) => {
      const chunk = d.toString();
      stderrLog += chunk;
      /* Parse status lines from solve.mjs and relay to live status display. */
      for (const line of chunk.split('\n')) {
        const t = line.trim();
        if (!t || !t.includes('[derive]')) continue;
        const msg = t.replace(/^\[derive\]\s*/, '');
        /* Show key status updates. */
        if (t.includes('Starting 3 agents')) logStatus(problem.idx, 'Agents running...', C.yellow);
        else if (t.includes('excluded from critique')) logStatus(problem.idx, msg, C.red);
        else if (t.includes('Active agents')) logStatus(problem.idx, msg, C.cyan);
        else if (t.includes('Combining')) logStatus(problem.idx, msg, C.yellow);
        else if (t.includes('Combined ')) logStatus(problem.idx, msg, C.dim);
        else if (t.includes('Critiquing')) logStatus(problem.idx, msg, C.yellow);
        else if (t.includes('Refining')) logStatus(problem.idx, msg, C.yellow);
        else if (t.includes('scores:')) logStatus(problem.idx, msg, '');
        else if (t.includes('combined score mean')) logStatus(problem.idx, msg, C.magenta);
        else if (t.includes('First-round scores')) logStatus(problem.idx, msg, C.cyan);
        else if (t.includes('Converged')) logStatus(problem.idx, msg, C.green);
        else if (t.includes('Done:')) logStatus(problem.idx, msg, C.green);
        else if (t.includes('quota exceeded')) logStatus(problem.idx, msg, C.yellow);
        else if (t.includes('excluded')) logStatus(problem.idx, msg, C.red);
        else if (t.includes('stderr preview') || t.includes('stdout:') || t.includes('stderr:')) logStatus(problem.idx, msg, C.dim);
        else if (t.includes('content preview')) logStatus(problem.idx, msg, C.dim);
        else if (t.includes('Transient error') || t.includes('Waiting') || t.includes('Retrying')) logStatus(problem.idx, msg, C.yellow);
        else if (t.includes('FAIL') || t.includes('OK')) logStatus(problem.idx, msg, t.includes('FAIL') ? C.red : C.dim);
      }
    });

    child.on('close', (exitCode) => {
      clearTimeout(timeoutHandle);
      activeChildren.delete(child);
      const elapsed = (Date.now() - startTime) / 1000;
      let result;
      try {
        result = JSON.parse(stdout.trim());
      } catch {
        result = { response: stdout || stderrLog, cost: 0, time: elapsed };
      }
      resolve({ ...result, exitCode, elapsed, stderrLog });
    });

    child.on('error', (err) => {
      clearTimeout(timeoutHandle);
      activeChildren.delete(child);
      const elapsed = (Date.now() - startTime) / 1000;
      resolve({ response: `Error: ${err.message}`, cost: 0, time: elapsed, exitCode: -1, elapsed, stderrLog: '' });
    });
  });
}

/* ── Parallel runner ─────────────────────────────────────────────── */

async function runParallel(problems, maxParallel) {
  const results = new Array(problems.length);
  let nextIdx = 0;

  async function worker() {
    while (nextIdx < problems.length && !cancelled) {
      const i = nextIdx++;
      const problem = problems[i];
      const tmpDir = resolve(PROJECT_ROOT, '.derive', 'benchmark', 'work', `problem-${problem.idx}`);
      mkdirSync(tmpDir, { recursive: true });
      logStatus(problem.idx, 'Starting...', C.cyan);
      results[i] = await solveProblem(problem, tmpDir);
    }
  }

  const workers = [];
  for (let w = 0; w < Math.min(maxParallel, problems.length); w++) workers.push(worker());
  await Promise.all(workers);
  return results;
}

/* ── Leaderboard ─────────────────────────────────────────────────── */

/** Fetch MathArena AIME 2025 leaderboard entries as [{name, accuracy}]. */
async function fetchLeaderboard() {
  try {
    const raw = await fetchUrlWithRetries('https://matharena.ai/competition_tables/aime--aime_2025');
    const data = JSON.parse(raw);
    const html = data.table || '';

    const entries = [];
    const rowRegex = /<tr[^>]*>([\s\S]*?)<\/tr>/gi;
    let match;
    let isHeader = true;
    while ((match = rowRegex.exec(html)) !== null) {
      const cells = [];
      const cellRegex = /<t[dh][^>]*>([\s\S]*?)<\/t[dh]>/gi;
      let cm;
      while ((cm = cellRegex.exec(match[1])) !== null) {
        cells.push(cm[1].replace(/<[^>]+>/g, '').replace(/\s+/g, ' ').trim());
      }
      if (isHeader) { isHeader = false; continue; }
      if (cells.length < 4) continue;
      /* cells: [rank, modelName, provider, accuracy, ...] */
      const name = cells[1].replace(/⚠️/g, '').trim();
      const accMatch = cells[3].match(/([\d.]+)%/);
      if (accMatch) {
        entries.push({ name, accuracy: parseFloat(accMatch[1]), isDerive: false });
      }
    }
    return entries;
  } catch (e) {
    process.stderr.write(`[leaderboard] Failed to fetch: ${e.message}\n`);
    return null;
  }
}

/* ── Main ────────────────────────────────────────────────────────── */

async function main() {
  process.stderr.write(`\n${C.bold}╔══════════════════════════════════════════════════════════╗${C.reset}\n`);
  process.stderr.write(`${C.bold}║     Derive Multi-Agent Benchmark — AIME 2025             ║${C.reset}\n`);
  process.stderr.write(`${C.bold}╚══════════════════════════════════════════════════════════╝${C.reset}\n\n`);

  /* Fetch problems. */
  process.stderr.write(`${C.dim}Fetching AIME 2025 problems from HuggingFace...${C.reset}\n`);
  let problems = await fetchProblems();
  if (problemFilter) {
    problems = problems.filter((p) => problemFilter.includes(p.idx));
  }
  process.stderr.write(`${C.green}Loaded ${problems.length} problems${C.reset}\n`);
  process.stderr.write(`${C.dim}Max parallel: ${MAX_PARALLEL}, Max critique rounds: ${process.env.DERIVE_MAX_CRITIQUE_ROUNDS ?? '7'}, Timeout: ${TIMEOUT_SEC}s${C.reset}\n`);
  process.stderr.write(`${C.dim}Output dir: ${OUTPUT_DIR}${C.reset}\n\n`);

  mkdirSync(OUTPUT_DIR, { recursive: true });

  const startTime = Date.now();
  const results = await runParallel(problems, MAX_PARALLEL);
  const totalTime = (Date.now() - startTime) / 1000;

  /* ── Grade all per-round combined responses via matharena ─ */
  process.stderr.write(`\n${C.dim}Grading per-round responses via matharena...${C.reset}\n`);
  const gradeItems = [];
  const gradeIndex = []; // maps gradeItems index → { problemIdx, roundIdx }
  for (let i = 0; i < problems.length; i++) {
    const result = results[i];
    /* Grade the final response. */
    gradeItems.push({ response: result.response ?? '', gold: problems[i].gold_answer });
    gradeIndex.push({ problemIdx: i, roundIdx: -1 }); // -1 = final
    /* Grade each round's combined response. */
    if (result.trace?.rounds) {
      for (let r = 0; r < result.trace.rounds.length; r++) {
        const combinedContent = result.trace.rounds[r].combine?.content ?? '';
        gradeItems.push({ response: combinedContent, gold: problems[i].gold_answer });
        gradeIndex.push({ problemIdx: i, roundIdx: r });
      }
    }
  }
  const gradeResults = await gradeResponses(gradeItems);
  /* Map grade results back to problems. */
  const perProblemGrades = new Array(problems.length).fill(null).map(() => ({ final: null, rounds: [] }));
  for (let g = 0; g < gradeIndex.length; g++) {
    const { problemIdx, roundIdx } = gradeIndex[g];
    if (roundIdx === -1) {
      perProblemGrades[problemIdx].final = gradeResults[g];
    } else {
      perProblemGrades[problemIdx].rounds[roundIdx] = gradeResults[g];
    }
  }
  process.stderr.write(`${C.green}Graded ${gradeItems.length} responses${C.reset}\n`);

  /* ── Write per-problem files and score ──────────────────── */
  const scoreboard = [];
  let correct = 0;

  for (let i = 0; i < problems.length; i++) {
    const problem = problems[i];
    const result = results[i];
    const grade = perProblemGrades[i].final;
    const extracted = grade?.extracted ?? extractBoxedAnswer(result.response);
    const isCorrect = grade?.correct ?? (extracted !== null && extracted === problem.gold_answer);
    if (isCorrect) correct++;

    const roundGrades = perProblemGrades[i].rounds;
    const entry = {
      idx: problem.idx,
      type: problem.type,
      gold: problem.gold_answer,
      extracted: extracted ?? '(none)',
      correct: isCorrect,
      time: result.elapsed,
      converged: result.converged ?? false,
      rounds: result.critiqueRounds ?? 0,
      maxRounds: result.maxRounds ?? 0,
      firstRoundScores: result.firstRoundScores ?? {},
      roundScores: [],
      roundCorrectness: roundGrades.map((g) => g ? { extracted: g.extracted, correct: g.correct } : null),
    };

    /* Extract per-round scores from the trace. */
    if (result.trace?.rounds) {
      for (const roundData of result.trace.rounds) {
        const roundEntry = {
          round: roundData.round,
          combineDurationMs: roundData.combine?.durationMs,
          critiqueDurationMs: roundData.critiqueDurationMs,
          refineDurationMs: roundData.refineDurationMs,
          combinedScores: roundData.combinedScores ?? [],
          combinedScoreMean: roundData.combinedScoreMean,
          perAgent: {},
        };
        for (const [reviewer, critique] of Object.entries(roundData.critiques || {})) {
          if (critique.parsed) {
            roundEntry.perAgent[reviewer] = {};
            for (const [target, val] of Object.entries(critique.parsed)) {
              if (val && typeof val.score === 'number') {
                roundEntry.perAgent[reviewer][target] = { score: val.score, feedback: val.feedback };
              }
            }
          }
        }
        entry.roundScores.push(roundEntry);
      }
    }

    scoreboard.push(entry);

    /* Write detailed result file. */
    const detailFile = {
      idx: problem.idx,
      problem: problem.problem,
      gold_answer: problem.gold_answer,
      extracted_answer: extracted,
      correct: isCorrect,
      source: 'AIME 2025',
      types: Array.isArray(problem.type) ? problem.type : [problem.type],
      converged: result.converged ?? false,
      critiqueRounds: result.critiqueRounds ?? 0,
      maxRounds: result.maxRounds ?? 0,
      totalTimeSec: result.elapsed,
      /* Full solution text. */
      finalResponse: result.response,
      /* Initial agent outputs. */
      initialAgents: result.trace?.initialAgents ?? {},
      /* Per-round details. */
      rounds: result.trace?.rounds?.map((r, rIdx) => ({
        round: r.round,
        combinedResponse: r.combine?.content,
        combineDurationMs: r.combine?.durationMs,
        critiques: Object.fromEntries(
          Object.entries(r.critiques || {}).map(([k, v]) => [k, {
            parsed: v.parsed,
            parseFailed: v.parseFailed,
            durationMs: v.durationMs,
          }])
        ),
        refinements: Object.fromEntries(
          Object.entries(r.refinements || {}).map(([k, v]) => [k, {
            contentLength: v.contentLength,
            durationMs: v.durationMs,
            content: v.content,
          }])
        ),
        combinedScores: r.combinedScores,
        combinedScoreMean: r.combinedScoreMean,
        /* Matharena grade for this round's combined response. */
        grade: roundGrades[rIdx] ? { extracted: roundGrades[rIdx].extracted, correct: roundGrades[rIdx].correct } : null,
      })) ?? [],
      /* Matharena-compatible fields. */
      N: 1,
      cost: { cost: 0, input_tokens: 0, output_tokens: 0, time: result.elapsed ?? 0 },
      pass_at_1: isCorrect ? 1.0 : 0.0,
      answers: [extracted ?? ''],
      correct_array: [isCorrect],
      messages: [[
        { role: 'user', content: `Put your final answer within \\boxed{}.\nThe answer is an integer between 0 and 999 inclusive.\n\n${problem.problem}` },
        { role: 'assistant', type: 'response', content: result.response ?? '' },
      ]],
    };
    writeFileSync(resolve(OUTPUT_DIR, `${problem.idx}.json`), JSON.stringify(detailFile, null, 2));

    /* Write the full stderr log. */
    if (result.stderrLog) {
      writeFileSync(resolve(OUTPUT_DIR, `${problem.idx}.log`), result.stderrLog);
    }

    const mark = isCorrect ? `${C.green}PASS${C.reset}` : `${C.red}FAIL${C.reset}`;
    logStatus(problem.idx, `${mark} gold=${problem.gold_answer} got=${extracted ?? '(none)'} (${result.elapsed?.toFixed(0)}s, ${result.critiqueRounds ?? '?'} rounds)`, '');
  }

  /* ── Results Table ─────────────────────────────────────── */
  process.stderr.write(`\n${C.bold}${'═'.repeat(110)}${C.reset}\n`);
  process.stderr.write(`${C.bold}  AIME 2025 Results — Derive Multi-Agent System${C.reset}\n`);
  process.stderr.write(`${C.bold}${'═'.repeat(110)}${C.reset}\n\n`);

  const hdr = `${rpad('#', 3)} │ ${pad('Result', 6)} │ ${pad('Gold', 5)} │ ${pad('Got', 8)} │ ${rpad('Time', 6)} │ ${rpad('Rnds', 4)} │ ${pad('Conv', 4)} │ ${pad('Score Trajectory', 30)} │ ${pad('Correct by Round', 20)} │ Type`;
  process.stderr.write(`  ${hdr}\n`);
  process.stderr.write(`  ${'─'.repeat(130)}\n`);

  for (const row of scoreboard) {
    const mark = row.correct ? `${C.green} PASS ${C.reset}` : `${C.red} FAIL ${C.reset}`;
    const convMark = row.converged ? `${C.green}yes ${C.reset}` : `${C.dim}no  ${C.reset}`;

    /* Build score trajectory string showing combined mean each round. */
    const trajectory = row.roundScores
      .map((r) => r.combinedScoreMean !== null ? r.combinedScoreMean.toFixed(0) : '?')
      .join(' → ');

    /* Build correctness trajectory showing PASS/FAIL per round. */
    const correctTrajectory = row.roundCorrectness
      .map((g) => g ? (g.correct ? `${C.green}✓${C.reset}` : `${C.red}✗${C.reset}`) : '?')
      .join(' → ');
    /* Plain text version for padding calculation (ANSI codes have zero width). */
    const correctPlain = row.roundCorrectness
      .map((g) => g ? (g.correct ? '✓' : '✗') : '?')
      .join(' → ');

    const typeStr = Array.isArray(row.type) ? row.type.join(', ') : row.type;

    process.stderr.write(
      `  ${String(row.idx).padStart(2, '0')}  │ ${mark} │ ${pad(row.gold, 5)} │ ${pad(row.extracted, 8)} │ ${rpad(row.time?.toFixed(0) + 's', 6)} │ ${rpad(row.rounds, 4)} │ ${convMark} │ ${pad(trajectory, 30)} │ ${correctTrajectory}${' '.repeat(Math.max(0, 20 - correctPlain.length))} │ ${typeStr}\n`
    );
  }
  process.stderr.write(`  ${'─'.repeat(130)}\n`);
  const acc = problems.length > 0 ? (100 * correct / problems.length).toFixed(1) : '0.0';
  process.stderr.write(`\n  ${C.bold}Accuracy: ${correct}/${problems.length} (${acc}%)${C.reset}\n`);
  process.stderr.write(`  Wall clock: ${totalTime.toFixed(0)}s (${(totalTime / 60).toFixed(1)} min)\n`);

  /* ── Aggregate Statistics ──────────────────────────────── */
  process.stderr.write(`\n${C.bold}  Aggregate Statistics${C.reset}\n`);
  process.stderr.write(`  ${'─'.repeat(50)}\n`);

  const times = scoreboard.map((r) => r.time).filter(Boolean);
  const rounds = scoreboard.map((r) => r.rounds);
  const convergedCount = scoreboard.filter((r) => r.converged).length;

  if (times.length > 0) {
    const avgTime = times.reduce((a, b) => a + b, 0) / times.length;
    const minTime = Math.min(...times);
    const maxTime = Math.max(...times);
    const medianTime = [...times].sort((a, b) => a - b)[Math.floor(times.length / 2)];
    process.stderr.write(`  Time per problem:  avg=${avgTime.toFixed(0)}s  median=${medianTime.toFixed(0)}s  min=${minTime.toFixed(0)}s  max=${maxTime.toFixed(0)}s\n`);
  }
  if (rounds.length > 0) {
    const avgRounds = rounds.reduce((a, b) => a + b, 0) / rounds.length;
    process.stderr.write(`  Critique rounds:   avg=${avgRounds.toFixed(1)}  converged=${convergedCount}/${problems.length}\n`);
  }

  /* Per-round average combined score and accuracy across all problems. */
  const maxRoundSeen = Math.max(...scoreboard.map((r) => r.roundScores.length));
  if (maxRoundSeen > 0) {
    process.stderr.write(`  Per-round combined score and accuracy:\n`);
    for (let r = 0; r < maxRoundSeen; r++) {
      const scores = scoreboard
        .filter((row) => row.roundScores[r]?.combinedScoreMean != null)
        .map((row) => row.roundScores[r].combinedScoreMean);
      const roundCorrect = scoreboard
        .filter((row) => row.roundCorrectness[r]?.correct != null)
        .map((row) => row.roundCorrectness[r]);
      const correctCount = roundCorrect.filter((g) => g?.correct === true).length;
      const roundTotal = roundCorrect.length;
      const avgScore = scores.length > 0 ? (scores.reduce((a, b) => a + b, 0) / scores.length).toFixed(1) : '?';
      const accPct = roundTotal > 0 ? `${(100 * correctCount / roundTotal).toFixed(0)}%` : '?';
      process.stderr.write(`    Round ${r + 1}: score=${avgScore} (n=${scores.length})  accuracy=${correctCount}/${roundTotal} (${accPct})\n`);
    }
  }

  /* First-round per-agent scores (raw agent performance before refinement). */
  const firstRoundByAgent = {};
  for (const row of scoreboard) {
    if (row.firstRoundScores) {
      for (const [agent, data] of Object.entries(row.firstRoundScores)) {
        if (!firstRoundByAgent[agent]) firstRoundByAgent[agent] = [];
        firstRoundByAgent[agent].push(data.mean);
      }
    }
  }
  if (Object.keys(firstRoundByAgent).length > 0) {
    process.stderr.write(`  First-round agent scores (raw, before refinement):\n`);
    for (const [agent, means] of Object.entries(firstRoundByAgent).sort((a, b) => a[0].localeCompare(b[0]))) {
      const avg = means.reduce((a, b) => a + b, 0) / means.length;
      process.stderr.write(`    ${pad(agent, 12)} avg=${avg.toFixed(1)} (n=${means.length})\n`);
    }
  }

  /* Per-category accuracy. */
  const cats = {};
  for (const row of scoreboard) {
    const types = Array.isArray(row.type) ? row.type : [row.type];
    for (const t of types) {
      if (!cats[t]) cats[t] = { correct: 0, total: 0 };
      cats[t].total++;
      if (row.correct) cats[t].correct++;
    }
  }
  process.stderr.write(`  Per-category accuracy:\n`);
  for (const [cat, { correct: c, total: t }] of Object.entries(cats).sort((a, b) => a[0].localeCompare(b[0]))) {
    process.stderr.write(`    ${pad(cat, 20)} ${c}/${t} (${(100 * c / t).toFixed(0)}%)\n`);
  }

  /* ── Leaderboard ───────────────────────────────────────── */
  process.stderr.write(`\n${C.bold}  MathArena AIME 2025 Leaderboard${C.reset}\n`);
  process.stderr.write(`  ${'─'.repeat(70)}\n`);

  const entries = await fetchLeaderboard();
  if (entries && entries.length > 0) {
    const deriveAcc = problems.length > 0 ? (100 * correct / problems.length) : 0;
    entries.push({ name: '>>> DERIVE (Multi-Agent) <<<', accuracy: deriveAcc, isDerive: true });
    entries.sort((a, b) => b.accuracy - a.accuracy);

    /* Print top entries and Derive's neighborhood. */
    const deriveRank = entries.findIndex((e) => e.isDerive) + 1;
    process.stderr.write(`  ${C.bold}Derive rank: #${deriveRank} of ${entries.length}${C.reset}\n\n`);

    const printRange = new Set();
    /* Always show top 10. */
    for (let i = 0; i < Math.min(10, entries.length); i++) printRange.add(i);
    /* Show 2 above and 2 below Derive. */
    for (let i = Math.max(0, deriveRank - 3); i <= Math.min(entries.length - 1, deriveRank + 1); i++) printRange.add(i);

    const sortedPrint = [...printRange].sort((a, b) => a - b);
    let lastPrinted = -1;
    for (const idx of sortedPrint) {
      if (lastPrinted >= 0 && idx > lastPrinted + 1) {
        process.stderr.write(`  ${C.dim}  ...${C.reset}\n`);
      }
      const e = entries[idx];
      const rank = idx + 1;
      const color = e.isDerive ? C.green + C.bold : '';
      const reset = e.isDerive ? C.reset : '';
      process.stderr.write(`  ${color}${rpad(rank, 3)}. ${pad(e.name, 40)} ${e.accuracy.toFixed(2)}%${reset}\n`);
      lastPrinted = idx;
    }
  } else {
    process.stderr.write(`  ${C.dim}Could not fetch leaderboard data.${C.reset}\n`);
  }

  /* Write summary JSON to output dir instead of dumping to console. */
  const summaryPath = resolve(OUTPUT_DIR, 'summary.json');
  writeFileSync(summaryPath, JSON.stringify({
    competition: 'AIME 2025',
    model: 'Derive Multi-Agent',
    accuracy: problems.length > 0 ? correct / problems.length : 0,
    correct,
    total: problems.length,
    totalTime,
    scoreboard: scoreboard.map((s) => ({ ...s, roundScores: undefined, roundCorrectness: s.roundCorrectness })),
  }, null, 2));

  process.stderr.write(`\n  ${C.dim}Results written to: ${OUTPUT_DIR}/${C.reset}\n\n`);
}

main().catch((err) => {
  console.error(`Fatal error: ${err.message}`);
  process.exit(1);
});
