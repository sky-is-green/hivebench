#!/usr/bin/env node
/**
 * Claims gate — coordination protocol v2, Rules 2 & 3 (MULTI-OX-PROJECT-PLAN.md).
 *
 * Every staged file must be covered by a registered worker claim under
 * `.claims/<Task_ID>.json` (see HIVE-OPS.md:4):
 *
 *   {
 *     "task_id": "S1",
 *     "worker": "BEE-BETA-2 @ ../worktrees/hivebench-studio/hivebench-S1",
 *     "coordination_doc": "HIVE-PLAN.md",
 *     "provider": "free-groq",
 *     "target_files": ["packages/client/ui-models-manager/**"],
 *     "beacon": "writing store; 3/5 specs green",
 *     "updated": "2026-08-27T00:00:00Z",
 *     "lease": "4h"
 *   }
 *
 * `provider` selects the LLM for this bee (`local-big` for the single BEE-GUARD
 * big-brain per HIVE-OPS.md:9, `free-*` for the BEE-BETA swarm). Any value is
 * allowed; GUARD is validated to be `local-*` when present.
 *
 * `target_files` entries are exact paths or glob patterns (`*` within a
 * segment, `**` across segments); a trailing `/` covers the whole directory.
 *
 * Hotspot files (registry in MULTI-OX-PROJECT-PLAN.md) additionally enforce the
 * single-commit rule: staging a hotspot allows ONLY files under that same
 * hotspot, and the commit's claim must cover it.
 *
 * `vendor/**` is rejected outright — vendored framework changes go upstream
 * through the sync procedure.
 *
 * Exempt from claims: this gate's own registry (`.claims/**`),
 * `MULTI-OX-PROJECT-PLAN.md`, and `.gitignore`.
 *
 * Emergency bypass: CLAIMS_GATE=skip (reserved for the human operator;
 * note the reason in the plan doc's working notes).
 */

import { execSync, execFileSync } from 'node:child_process';
import { readFileSync, readdirSync, existsSync } from 'node:fs';
import { join } from 'node:path';

// Session-proof decoding: some worker sessions hand git's stdout back as
// UTF-16LE (console codepage mismatch), which corrupts staged-path matching
// when read naively as UTF-8. The -z delimiter plus the NUL-byte signature
// makes the encoding detectable and both forms decodable. (PROPOSALS.md:
// claims-gate-encoding.)
function decodeGitOutput(buffer) {
  if (!Buffer.isBuffer(buffer)) buffer = Buffer.from(buffer);
  let zeros = 0;
  for (let i = 0; i < buffer.length; i++) if (buffer[i] === 0) zeros++;
  const looksUtf16 = buffer.length > 0 && zeros > buffer.length * 0.25;
  return looksUtf16 ? buffer.toString('utf16le') : buffer.toString('utf8');
}

function stagedFiles() {
  const raw = execFileSync(
    'git',
    ['-c', 'core.quotepath=false', 'diff', '--cached', '--name-only', '-z'],
    { encoding: 'buffer' }
  );
  return decodeGitOutput(raw)
    .split('\0')
    .map((s) => s.trim())
    .filter(Boolean);
}

if (process.argv.includes('--selftest')) {
  const assert = (cond, msg) => { if (!cond) { console.error('x FAIL: ' + msg); process.exit(1); } };
  const sample = Buffer.from('a.txt\0b dir/c.txt\0');
  assert(decodeGitOutput(sample) === 'a.txt\0b dir/c.txt\0', 'utf8 -z passthrough');
  const u16 = Buffer.from('a.txt\0b dir/c.txt\0', 'utf16le');
  assert(decodeGitOutput(u16) === 'a.txt\0b dir/c.txt\0', 'utf16le detected and decoded');
  console.log('claims gate selftest: OK');
  process.exit(0);
}

const EXEMPT = ['.claims/', 'MULTI-OX-PROJECT-PLAN.md', '.gitignore', '.agents/notes/', 'HIVE-PLAN.md', 'MULTI_AGENT_PLAN.md', 'SPLINTER-PLAN.md'];

// Hotspot registry mirror (keep in sync with MULTI-OX-PROJECT-PLAN.md §3).
const HOTSPOTS = [
  '/pyproject.toml',
  '/requirements.txt',
  '/requirements-dev.txt',
  '/providers.example.json',
  'docs/INTEGRATE.md',
  'splinter/__init__.py',
];

function fail(messages) {
  for (const m of messages) console.error(`x ${m}`);
  console.error(
    '\nclaims gate: register .claims/<Task_ID>.json (task_id, worker, ' +
      'target_files) covering every staged path, then re-add and commit.' +
      '\n           Human/emergency bypass: CLAIMS_GATE=skip git commit ...'
  );
  process.exit(1);
}

function globToRegExp(pattern) {
  // Tokenize so a bare '**' stays '.*': re-scanning the replacement would
  // re-match its '*' as a single-star and produce '.[^/]*'.
  let out = '';
  for (let i = 0; i < pattern.length; i++) {
    const c = pattern[i];
    if (c === '*') {
      if (pattern[i + 1] === '*') {
        i++;
        if (pattern[i + 1] === '/') {
          i++;
          out += '(?:.*/)?';
        } else {
          out += '.*';
        }
      } else {
        out += '[^/]*';
      }
    } else {
      out += c.replace(/[.+^${}()|[\]\\]/g, '\\$&');
    }
  }
  return new RegExp(`^${out}$`);
}

function covers(patterns, path) {
  return patterns.some((p) => {
    if (p === path) return true;
    const dirPattern = p.endsWith('/') ? p + '**' : p;
    return globToRegExp(dirPattern).test(path);
  });
}

// Membership shares one predicate with the alone-check below: a root-anchored
// hotspot ('/package.json') names the exact file without a leading slash.
function inHotspot(path, hotspot) {
  return hotspot.startsWith('/')
    ? path === hotspot.slice(1)
    : covers([hotspot], path);
}

function hotspotOf(path) {
  return HOTSPOTS.find((h) => inHotspot(path, h)) ?? null;
}

if (process.env.CLAIMS_GATE === 'skip') {
  console.log('claims gate: skipped via CLAIMS_GATE=skip');
  process.exit(0);
}

// Global stop lever: if .claims/HIVE-FREEZE exists, every commit is rejected.
// See HIVE-OPS.md §4 - one file creation freezes the whole workforce.
const freezeFile = join(process.cwd(), '.claims', 'HIVE-FREEZE');
if (existsSync(freezeFile)) {
  const reason = readFileSync(freezeFile, 'utf8').trim() || 'no reason recorded';
  fail([
    `HIVE FROZEN: ${reason}`,
    'Remove .claims/HIVE-FREEZE to unfreeze the workforce.',
  ]);
}

const staged = stagedFiles();

if (staged.length === 0) process.exit(0);

const exempted = staged.filter((f) => EXEMPT.some((e) => f === e || f.startsWith(e)));
const checked = staged.filter((f) => !exempted.includes(f));

if (checked.length === 0) process.exit(0);

const vendored = checked.filter((f) => f === 'vendor' || f.startsWith('vendor/'));
if (vendored.length > 0) {
  fail([
    'vendor/** changes are forbidden here:',
    ...vendored.map((f) => `  ${f}`),
    'Vendored framework edits go upstream through the sync procedure.',
  ]);
}

const claimsDir = join(process.cwd(), '.claims');
const claims = [];
if (existsSync(claimsDir)) {
  for (const name of readdirSync(claimsDir)) {
    if (!name.endsWith('.json')) continue;
    try {
      // PowerShell 5.1's `-Encoding utf8` emits a BOM; tolerate it.
      const raw = JSON.parse(
        readFileSync(join(claimsDir, name), 'utf8').replace(/^\uFEFF/, '')
      );
      // provider / coordination_doc / beacon / updated / lease are allowed and ignored here;
      // GUARD must be local-big (single big-brain herding free swarm per HIVE-OPS.md:9)
      if (raw.provider && String(raw.worker ?? "").includes("BEE-GUARD") && !String(raw.provider).startsWith("local")) {
        fail([`claim .claims/${name}: BEE-GUARD must use provider "local-big" (or local-*) per HIVE-OPS.md:9, got "${raw.provider}".`]);
      }
      claims.push({ file: name, patterns: raw.target_files ?? [], task_id: raw.task_id ?? null, worker: raw.worker ?? "?", provider: raw.provider ?? null });
    } catch (e) {
      if (e.message && e.message.startsWith("claim .claims/")) throw e;
      fail([`Malformed claim file .claims/${name}: invalid JSON.`]);
    }
  }
}

const unclaimed = checked.filter((f) => !claims.some((c) => covers(c.patterns, f)));
if (unclaimed.length > 0) {
  fail([
    'staged files not covered by any claim in .claims/:',
    ...unclaimed.map((f) => `  ${f}`),
    claims.length > 0
      ? 'open claims: ' + claims.map((c) => c.file).join(', ')
      : 'no claims are currently registered.',
    ...(claims.length === 0
      ? ['Create one, e.g. .claims/S1.json:', '{"task_id":"S1","worker":"you","target_files":["packages/client/ui-models-manager/**"]}']
      : []),
  ]);
}

// Seat-latch: no two active claims may share a Task_ID.
const seatMap = new Map();
for (const c of claims) {
  if (!c.task_id) continue;
  if (seatMap.has(c.task_id)) {
    fail([`duplicate Task_ID "${c.task_id}": held by ${seatMap.get(c.task_id)} and ${c.file}`]);
  }
  seatMap.set(c.task_id, c.file);
}
// Single-commit rule: a staged hotspot excludes everything else.
const hotspotSets = new Set(checked.map(hotspotOf).filter(Boolean));
if (hotspotSets.size > 1) {
  fail([
    'multiple hotspots staged in one commit:',
    ...[...hotspotSets].map((h) => `  ${h}`),
    'Split the commit - one hotspot per commit, nothing bundled.',
  ]);
}
if (hotspotSets.size === 1) {
  const [hotspot] = hotspotSets;
  const outside = checked.filter((f) => !inHotspot(f, hotspot));
  if (outside.length > 0) {
    fail([
      `hotspot "${hotspot}" must be committed ALONE.`,
      'bundled non-hotspot files detected:',
      ...outside.map((f) => `  ${f}`),
    ]);
  }
}

console.log(
  `claims gate: ${checked.length} staged file(s) covered by ${claims.length} claim(s)` +
    (exempted.length ? ` (+${exempted.length} exempt)` : '')
);
