"""Server-rendered report views (Seam B) + the Studio console page.

Plain HTML, no JS dependencies. Renders one ``run_report.json`` bundle: the
post-run PES headline with its weighted components, the P1–P11 verdict table,
the deterministic P2 retrieval diagnostic, comb (P11) totals, and the
baselines comparison. Every dynamic value is HTML-escaped; missing blocks
render as em dashes rather than erroring, so partial bundles from in-flight
runs stay viewable.

This file is UTF-8 and must stay that way — it contains arrows, stars and
ellipses in the console UI. Never round-trip it through PowerShell text
cmdlets (they decode as ANSI on Windows and mojibake every non-ASCII char).
"""

from __future__ import annotations

import html
from pathlib import Path

from fastapi import HTTPException

# Paper weights for the PES composite (README §3.1).
_PES_COMPONENTS = [
    ("retrieval_precision", 0.30),
    ("routing_accuracy", 0.20),
    ("latency_health", 0.20),
    ("throughput_health", 0.15),
    ("context_utilization", 0.15),
]
_BAND_FLOORS = [(80, "band-green"), (60, "band-yellow"), (40, "band-red"),
                (0, "band-critical")]

_CSS = """
 body { font-family: Segoe UI, system-ui, sans-serif; margin: 2rem auto; max-width: 60rem;
        color: #1c2733; background: #f7f9fb; padding: 0 1rem; }
 h1 { font-size: 1.4rem; } h2 { font-size: 1.05rem; margin-top: 2rem;
      border-bottom: 1px solid #dde5ec; padding-bottom: .3rem; }
 table { border-collapse: collapse; width: 100%; background: #fff; }
 th, td { text-align: left; padding: .45rem .7rem; border-bottom: 1px solid #e8edf2;
          font-size: .92rem; }
 th { background: #eef2f6; }
 .cards { display: flex; gap: 1rem; flex-wrap: wrap; }
 .card { background: #fff; border-radius: 8px; padding: 1rem 1.4rem;
         box-shadow: 0 1px 3px rgba(16,32,48,.08); min-width: 150px; }
 .card span { color: #5a6b7d; font-size: .85rem; }
 .card b { display: block; font-size: 1.6rem; }
 .kv-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(230px, 1fr));
            gap: .4rem 1.4rem; background: #fff; padding: 1rem; border-radius: 8px; }
 .kv span { color: #5a6b7d; } .kv b { float: right; font-variant-numeric: tabular-nums; }
 .band-green { color: #157a3e; } .band-yellow { color: #9a6b00; }
 .band-red { color: #b3372c; } .band-critical { color: #8a1f1f; font-weight: 700; }
 .st-pass { color: #157a3e; font-weight: 600; } .st-fail { color: #b3372c; font-weight: 600; }
 .st-skip { color: #9a6b00; } .st-report { color: #456; }
 .note { color: #1c2733; font-size: .85rem; }
 code { background: #000; color: #FFDD00; padding: .15rem .4rem; border-radius: 4px; border: 1px solid #222; }
 .note code { background: #000; color: #FFDD00; border-color: #FFDD00; }
 ul.runs { list-style: none; padding: 0; } ul.runs li { margin: .35rem 0; }
"""


def _esc(value: object) -> str:
    return html.escape(str(value))


def _fmt(value: object, nd: int = 1) -> str:
    if value is None or value == "":
        return "&mdash;"
    try:
        return f"{float(value):.{nd}f}"
    except (TypeError, ValueError):
        return _esc(value)


def _pes_band_class(score: object) -> str:
    try:
        s = float(score)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ""
    for floor, cls in _BAND_FLOORS:
        if s >= floor:
            return cls
    return ""


def _status_cell(status: object) -> str:
    s = _esc(status or "")
    return f"<span class='st-{s.lower()}'>{s}</span>"


def resolve_run_dir(runs_root: Path, run_dir: str) -> Path:
    """Resolve + confine a run_dir under runs_root (path-traversal safe)."""
    root = runs_root.resolve()
    target = (root / run_dir).resolve()
    if target != root and root not in target.parents:
        raise HTTPException(400, "run_dir must stay inside the runs directory")
    return target


def _comb_totals(comb: dict) -> dict:
    totals = {"archived": 0, "resurrected": 0, "comb_hits": 0, "gate_fired": 0}
    conversations = comb.get("conversations")
    if isinstance(conversations, list):
        for entry in conversations:
            if not isinstance(entry, dict):
                continue
            for key in totals:
                value = entry.get(key)
                if isinstance(value, (int, float)):
                    totals[key] += int(value)
    return totals


def _baseline_rows(report: dict, composite: object) -> list[tuple[str, object]]:
    rows: list[tuple[str, object]] = [("Splinter (post-run PES)", composite)]
    for label, key in (("LM-Studio rolling", "baseline_lm_studio"),
                       ("FIFO truncation", "baseline_fifo")):
        blob = report.get(key)
        if not isinstance(blob, dict):
            nested = report.get("baselines")
            blob = nested.get(key) if isinstance(nested, dict) else None
        if isinstance(blob, dict):
            aggregate = blob.get("aggregate")
            if isinstance(aggregate, dict):
                rows.append((label, aggregate.get("avg_pes")))
    return rows


def render_report_page(report: dict, run_name: str) -> str:
    """One run_report.json bundle as a standalone HTML page."""
    if isinstance(report, dict) and report.get("kind") == "training":
        return render_training_report(report, run_name)
    aggregate = report.get("aggregate") if isinstance(report.get("aggregate"), dict) else {}
    post = report.get("post_run_pes") if isinstance(report.get("post_run_pes"), dict) else {}
    components = post.get("components") if isinstance(post.get("components"), dict) else {}
    diag = report.get("retrieval_diagnostic") \
        if isinstance(report.get("retrieval_diagnostic"), dict) else {}
    comb = report.get("comb") if isinstance(report.get("comb"), dict) else {}
    comb_t = _comb_totals(comb)
    # generate_data has used two shapes across versions: composite/components
    # and pes/breakdown — accept both.
    composite = post.get("composite", post.get("pes"))
    components = post.get("components")
    if not isinstance(components, dict):
        components = post.get("breakdown")
    if not isinstance(components, dict):
        components = {}
    band = str(post.get("band") or "")

    def _band_class() -> str:
        if band:
            for token in ("green", "yellow", "red", "critical"):
                if token in band.lower():
                    return f"band-{token}"
            return ""
        return _pes_band_class(composite)

    protocol_rows = "".join(
        "<tr>"
        f"<td>{_esc(p.get('id', ''))}</td>"
        f"<td>{_status_cell(p.get('status'))}</td>"
        f"<td>{_esc(p.get('title', ''))}</td>"
        f"<td class='note'>{_esc(p.get('note', ''))}</td>"
        "</tr>"
        for p in (report.get("protocol") or [])
        if isinstance(p, dict)
    ) or "<tr><td colspan='4' class='note'>no protocol block</td></tr>"

    component_rows = "".join(
        "<tr>"
        f"<td>{_esc(name.replace('_', ' ').title())}</td>"
        f"<td>{weight:.2f}</td>"
        f"<td>{_fmt(components.get(name))}</td>"
        "</tr>"
        for name, weight in _PES_COMPONENTS
    )

    baseline_rows = "".join(
        "<tr>"
        f"<td>{_esc(label)}</td>"
        f"<td class='{_pes_band_class(pes)}'><b>{_fmt(pes)}</b></td>"
        "</tr>"
        for label, pes in _baseline_rows(report, composite)
    )

    def kv(label: str, value: object, nd: int = 1) -> str:
        return f"<div class='kv'><span>{label}</span><b>{_fmt(value, nd)}</b></div>"

    avg_ms = aggregate.get("avg_total_ms")
    return f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>HiveBench report &mdash; {_esc(run_name)}</title>
<style>{_CSS}</style></head><body>
<h1>HiveBench report &mdash; <code>{_esc(run_name)}</code></h1>
<p class="note">mode <b>{_esc(report.get('mode') or '&mdash;')}</b> &middot;
backend <b>{_esc(report.get('backend') or '&mdash;')}</b> &middot; run
<b>{_esc(report.get('run_id') or '&mdash;')}</b></p>
<div class="cards">
<div class="card"><span>PES (post-run)</span><b class="{_band_class()}">{_fmt(composite)}</b></div>
<div class="card"><span>Avg per-turn PES</span><b>{_fmt(aggregate.get('avg_pes'))}</b></div>
<div class="card"><span>User turns</span><b>{_fmt(aggregate.get('user_turns'), 0)}</b></div>
<div class="card"><span>Avg turn</span><b>{_fmt(avg_ms / 1000 if isinstance(avg_ms, (int, float)) else None)}s</b></div>
</div>

<h2>PES components (post-run)</h2>
<table><tr><th>Component</th><th>Weight</th><th>Score</th></tr>
{component_rows}
</table>

<h2>P1&ndash;P11 verdicts</h2>
<table><tr><th>ID</th><th>Status</th><th>Prediction</th><th>Note</th></tr>
{protocol_rows}
</table>

<h2>Deterministic retrieval diagnostic (P2)</h2>
<div class="kv-grid">
{kv('Recall (all turns)', diag.get('retrieval_recall'))}
{kv('Recall (retrievable turns)', diag.get('retrieval_recall_retrievable'))}
{kv('Ingestion rate', diag.get('ingestion_rate'))}
{kv('Perfect-splinter ceiling', diag.get('perfect_hive_ceiling'))}
{kv('Precision (sentence proxy)', diag.get('retrieval_precision'))}
</div>

<h2>Comb (P11 surplus tier)</h2>
<div class="kv-grid">
{kv('Archived', comb_t['archived'], 0)}
{kv('Resurrected', comb_t['resurrected'], 0)}
{kv('Comb hits', comb_t['comb_hits'], 0)}
{kv('Gate fired', comb_t['gate_fired'], 0)}
</div>

<h2>Baselines comparison</h2>
<table><tr><th>System</th><th>PES</th></tr>
{baseline_rows}
</table>

<p class="note"><a href="/runs">&larr; all runs</a></p>
</body></html>"""


def render_runs_page(entries: list[dict]) -> str:
    """Index of available run bundles."""
    def _kind(e: dict) -> str:
        kind = e.get("kind") or "benchmark"
        return f" · <b>{_esc(kind)}</b>" if kind != "benchmark" else ""

    items = "".join(
        f"<li><a href='/view/{_esc(e['name'])}'><code>{_esc(e['name'])}</code></a> "
        f"<span class='note'>modified {_esc(e['modified'])}"
        f"{_kind(e)}{'' if e.get('has_report') else ' · no run_report.json yet'}</span></li>"
        for e in entries
    ) or "<li class='note'>no runs yet</li>"
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>HiveBench runs</title>
<style>{_CSS}</style></head><body>
<h1>HiveBench run bundles</h1>
<p><a href="/server"><button>← Studio console</button></a> <a href="/docs"><button>API docs</button></a></p>
<ul class="runs">{items}</ul>
</body></html>"""


def _training_kv(label: str, value: object, nd: int = 2) -> str:
    return f"<div class='kv'><span>{label}</span><b>{_fmt(value, nd)}</b></div>"


def render_training_report(report: dict, run_name: str) -> str:
    """One training run bundle (``kind == 'training'``) as a standalone page."""
    aggregate = report.get("aggregate") if isinstance(report.get("aggregate"), dict) else {}
    progress = report.get("progress") if isinstance(report.get("progress"), dict) else {}
    config = report.get("config") if isinstance(report.get("config"), dict) else {}
    regions = report.get("regions") if isinstance(report.get("regions"), list) else []

    ratio = aggregate.get("mean_ratio")
    retention = report.get("retention")
    if retention is None and isinstance(ratio, (int, float)) and ratio:
        retention = round(100.0 / float(ratio), 1)

    region_rows = "".join(
        "<tr>"
        f"<td>{_esc(r.get('name', ''))}</td>"
        f"<td>{_fmt(r.get('start_token'), 0)}</td>"
        f"<td>{_fmt(r.get('n_windows'), 0)}</td>"
        f"<td>{_fmt(r.get('ppl'), 3)}</td>"
        f"<td>{_fmt(r.get('teacher_ppl'), 3)}</td>"
        f"<td><b>{_fmt(r.get('ratio'), 4)}</b></td>"
        "</tr>"
        for r in regions if isinstance(r, dict)
    ) or "<tr><td colspan='6' class='note'>no evaluation yet &mdash; run still in progress</td></tr>"

    return f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>Hive-Ternary run &mdash; {_esc(run_name)}</title>
<style>{_CSS}</style></head><body>
<h1>Hive-Ternary &mdash; <code>{_esc(run_name)}</code></h1>
<p class="note">engine <b>{_esc(report.get('engine') or '&mdash;')}</b> &middot;
recipe <b>{_esc(report.get('method') or '&mdash;')}</b> &middot;
status <b>{_esc(report.get('status') or '&mdash;')}</b></p>
<div class="cards">
<div class="card"><span>Mean ratio</span><b>{_fmt(ratio, 4)}</b></div>
<div class="card"><span>Retention</span><b>{_fmt(retention, 1)}%</b></div>
<div class="card"><span>Best region</span><b>{_fmt(aggregate.get('min_ratio'), 4)}</b></div>
<div class="card"><span>Regions</span><b>{_fmt(aggregate.get('n_regions'), 0)}</b></div>
</div>

<h2>Per-region retention (lower ratio is better)</h2>
<table><tr><th>Region</th><th>Start</th><th>Windows</th><th>PPL</th><th>Teacher</th><th>Ratio</th></tr>
{region_rows}
</table>

<h2>Run</h2>
<div class="kv-grid">
{_training_kv('Step', progress.get('step'), 0)}
{_training_kv('Total steps', progress.get('total_steps'), 0)}
{_training_kv('Loss', progress.get('loss'))}
{_training_kv('Deployed ratio', progress.get('deployed_ratio'))}
{_training_kv('Init ratio', progress.get('init_ratio'))}
{_training_kv('Elapsed (s)', progress.get('elapsed_s'), 0)}
</div>

<h2>Config</h2>
<div class="kv-grid">
{_training_kv('Model', config.get('model_dir'))}
{_training_kv('Corpus', config.get('corpus'))}
{_training_kv('Steps', config.get('steps'), 0)}
{_training_kv('Seq', config.get('seq'), 0)}
{_training_kv('LR', config.get('lr'))}
{_training_kv('Device', config.get('device'))}
{_training_kv('Eval regions', config.get('eval_regions'), 0)}
{_training_kv('Eval windows', config.get('eval_windows'), 0)}
</div>

<p class="note"><a href="/runs">&larr; all runs</a></p>
</body></html>"""


_TRAINING_PANE = r"""
<div id="tab-training" class="tabpane" style="display:none">
<section>
<h2 style="margin-top:0">Training <span class="note">Unsloth Core &middot; Hive-Ternary</span></h2>
<div class="row">
  <label class="inline">Engine <select id="tr-engine" onchange="trSyncMethods()"></select></label>
  <label class="inline">Recipe <select id="tr-method"></select></label>
  <label class="inline">Run name <input id="tr-name" size="16" placeholder="ste-rotate 20k"></label>
</div>
<div class="row"><label class="inline" style="flex:1">Model dir
  <input id="tr-model" size="34" placeholder="/path/to/hf base model"></label></div>
<div class="row"><label class="inline" style="flex:1">Corpus
  <input id="tr-corpus" size="34" placeholder="/path/to/corpus.txt"></label></div>
<div class="row">
  <label class="inline">steps <input id="tr-steps" type="number" value="20000" size="6"></label>
  <label class="inline">seq <input id="tr-seq" type="number" value="512" size="5"></label>
  <label class="inline">lr <input id="tr-lr" value="5e-5" size="7"></label>
  <label class="inline">device <input id="tr-device" value="cuda:1" size="7"></label>
  <label class="inline">seed <input id="tr-seed" type="number" value="1337" size="6"></label>
  <label class="inline">eval regions <input id="tr-regions" type="number" value="8" size="3"></label>
  <label class="inline">eval windows <input id="tr-windows" type="number" value="8" size="3"></label>
</div>
<div class="row">
  <button onclick="trLaunch(this)">Start training</button>
  <span id="tr-msg" class="note"></span>
</div>
<div id="tr-status" class="kv-grid"></div>
<pre id="tr-log" style="max-height:240px;overflow:auto"></pre>
<p class="note">One 512-token window per step &mdash; batch size is not a control here.
History lives in <a href="/runs">Runs</a>; ternary reports render at <code>/view/&lt;run&gt;</code>.</p>
</section>
<script>
const TR_CATALOG = __CATALOG__;
let trRun = null, trTimer = null;
function trFillEngines(){
  const es = document.getElementById('tr-engine'); if(!es) return;
  es.innerHTML = '';
  for (const e of TR_CATALOG){
    const o = document.createElement('option'); o.value = e.id;
    o.textContent = e.label + (e.external ? ' (external)' : '');
    es.appendChild(o);
  }
  trSyncMethods();
}
function trSyncMethods(){
  const ev = document.getElementById('tr-engine').value;
  const ms = document.getElementById('tr-method'); if(!ms) return;
  ms.innerHTML = '';
  const engine = TR_CATALOG.find(e => e.id === ev); if(!engine) return;
  for (const m of engine.methods){
    const o = document.createElement('option'); o.value = m.id;
    o.textContent = m.label + (m.implemented ? '' : ' \u2014 not implemented');
    o.disabled = !m.implemented;
    ms.appendChild(o);
  }
}
function trFields(){
  const val = id => { const el = document.getElementById(id); return el && el.value.trim() !== '' ? el.value.trim() : undefined; };
  return {model_dir: val('tr-model'), corpus: val('tr-corpus'),
          steps: val('tr-steps'), seq: val('tr-seq'), lr: val('tr-lr'),
          device: val('tr-device'), seed: val('tr-seed'),
          eval_regions: val('tr-regions'), eval_windows: val('tr-windows')};
}
async function trLaunch(btn){
  const msg = document.getElementById('tr-msg');
  const method = document.getElementById('tr-method').value;
  if(!method){ msg.textContent = 'pick a recipe'; return; }
  msg.textContent = 'launching\u2026';
  try{
    const r = await api('/v1/training/launch', 'POST',
      {engine: document.getElementById('tr-engine').value, method,
       name: document.getElementById('tr-name').value, fields: trFields()});
    trRun = r.run_dir;
    msg.textContent = 'started ' + r.run_dir + ' (pid ' + r.pid + ')';
    trPoll();
    if(trTimer) clearInterval(trTimer);
    trTimer = setInterval(trPoll, 3000);
  }catch(e){ msg.textContent = 'launch failed: ' + e.message; }
}
async function trPoll(){
  if(!trRun) return;
  try{
    const s = await api('/v1/training/status/' + encodeURIComponent(trRun));
    const p = s.progress || {};
    const rows = [['state', s.state], ['step', (p.step||0) + ' / ' + (p.total_steps||'?')],
                  ['loss', p.loss], ['deployed ratio', p.deployed_ratio],
                  ['init ratio', p.init_ratio], ['elapsed s', p.elapsed_s]];
    document.getElementById('tr-status').innerHTML = rows.map(([k,v]) =>
      "<div class='kv'><span>" + k + "</span><b>" +
      (v === undefined || v === null ? '&mdash;' : v) + "</b></div>").join('');
    if(s.state === 'finished' || s.state === 'stopped'){
      if(trTimer){ clearInterval(trTimer); trTimer = null; }
      document.getElementById('tr-msg').innerHTML =
        "done &mdash; <a href='/view/" + encodeURIComponent(trRun) + "'>open report</a>";
    }
  }catch(e){ /* keep polling while the run settles */ }
}
document.addEventListener('DOMContentLoaded', trFillEngines);
const trTabBtn = document.querySelector('[data-tab="tab-training"]');
if(trTabBtn) trTabBtn.addEventListener('click', () => setTimeout(trFillEngines, 10));
</script>
</div>
"""


def render_training_pane() -> str:
    """The Training tab pane (self-contained HTML + script)."""
    import json as _json

    from harness.training import engine_catalog

    return _TRAINING_PANE.replace("__CATALOG__", _json.dumps(engine_catalog()))


_STUDIO_CSS_PATH = Path(__file__).with_name("studio.css")

def _get_server_css() -> str:
    """Studio CSS: base + studio overrides, re-read on each request so refresh picks up edits without restart."""
    try:
        extra = _STUDIO_CSS_PATH.read_text(encoding="utf-8")
    except OSError:
        extra = ""
    return _CSS + extra

# Backwards compat: keep _SERVER_CSS as snapshot at import, but renderers call _get_server_css()
_SERVER_CSS = _get_server_css()


def tip(text: str) -> str:
    """One inline help glyph whose hover/focus tooltip explains the control it follows."""
    return ""


def render_server_page() -> str:
    """Model-management console: server + settings | loaded-model chat | hub.

    Discovery is live against the Hugging Face hub API — no model catalog is
    hardcoded here."""
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate"><meta http-equiv="Pragma" content="no-cache"><meta http-equiv="Expires" content="0"><title>Studio server &amp; models</title>
<style>{_get_server_css()}</style><style>input[type=number]::-webkit-inner-spin-button,input[type=number]::-webkit-outer-spin-button{{-webkit-appearance:none;margin:0}}input[type=number]{{-moz-appearance:textfield;appearance:textfield}}</style></head><body>
<div id="top-right-status" title="How: GET /v1/server/status hardware poll + process check. Does: Shows loaded model health. Changing: Green=ready, else Start needed." style="position:absolute; top:1rem; right:1.2rem; background:#000; color:#FFDD00; border:2px solid #000; padding:.45rem .9rem; border-radius:8px; font-weight:700; font-size:.88rem; z-index:10; max-width:40vw; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">Launch: No model loaded</div>
<h1>Splinter Studio console</h1>
<p style="margin:.3rem 0"><button id="afkbtn" onclick="toggleAfk(this)" title="How: toggleAfk sets SPLINTER-MODE.json. Does: QUEEN autonomy (GREEN/YELLOW auto, RED contained). Changing: On queues pushes/merges until return.">AFK</button> <a href="/runs"><button>Runs →</button></a> <a href="/docs"><button>API docs</button></a></p>
<form style="display:inline" onsubmit="event.preventDefault(); return false"><input id="researchq" placeholder="deep-research question..." size="30" autocomplete="off" title="How: researchAdd queues to RESEARCH-QUEUE.md. Does: Deep research task (Auditor only). Changing: Adds entry, not instant." onkeydown="if (event.key === &quot;Enter&quot;) researchAdd(this)"> <button id="researchsubmit" onclick="researchAdd(this)">Research</button> <span id="researchcount" class="meta" title="Queues a deep-research question for QUEEN. Execution is master-only; reports land in RESEARCH/&lt;slug&gt;.md and are summarized on wake."></span></form>

<div class="grid">

<!-- ==================== LEFT: tabs ==================== -->
<div class="col">
<div class="tabs">
<button class="tab" data-tab="tab-setup">Linux/Docker setup</button>
<button class="tab" data-tab="tab-agent">Agent</button>
<button class="tab" data-tab="tab-engines">Engines</button>
<button class="tab" data-tab="tab-library">Local Library</button>
<button class="tab" data-tab="tab-splinter">Splinter</button>
<button class="tab" data-tab="tab-providers">Providers</button>
<button class="tab" data-tab="tab-hub">Hub</button>
<button class="tab" data-tab="tab-inspect">Inspector</button>
<button class="tab" data-tab="tab-training">Training</button>
<button class="tab active" data-tab="tab-settings">Settings</button>
</div>

<div id="tab-setup" class="tabpane" style="display:none">
<section>
<h2 style="margin-top:0">Docker Setup — WebUI ↔ Linux model <span class="note">ROCm + VHDX bare</span></h2>
<div class="note" style="margin-bottom:.6rem; line-height:1.5">For extra large models (DeepSeek v4 or similar). Your chat app sends prompts to a Linux AI server that loads the model from a virtual drive <code>E:/dsh_storage.vhdx</code> (shown as <code>/mnt/dsh_storage</code> inside Linux). <span title="How: WebUI OPENAI_API_BASE_URLS includes http://dsh-compute-backend:8000/v1 (hivebench-studio defaults :3000 → :8000); Docker runs custom-dsh-rocm-backend with /dev/kfd + HSA 11.0.0 + FLASH3 FP8 on /mnt/dsh_storage/models. Does: Serves large model with tiered spill 20 VRAM+24 RAM+NVMe. Changing: No user action — Bootstrap sets it up." style="background:#000;color:#FFDD00;padding:.2rem .5rem;border-radius:4px;border:1px dashed #FFDD00;cursor:help;font-weight:700;text-decoration:underline dotted">Hover for technical details.</span> One click does: expose drive → mount in Linux → start AI container → WebUI connects.</div>
<div class="note" style="margin-bottom:.6rem;background:#000;color:#FFDD00;border:2px solid #FFDD00;padding:.6rem .8rem;border-radius:8px"><b>Easy Setup (first time: have WSL2 + Docker, no drive yet):</b> 1) Select <b>drive</b> + <b>size</b> → <b>Create Drive</b> (creates selected GB dynamic sparse, formats to ext4 on first mount — no auto-create, initially small) → 2) <b>Mount AI Drive (Admin)</b> → <b>Yes</b> on UAC → 3) <b>Bootstrap Docker</b> → 4) Add model to <code style="background:#1a1a00;color:#FFDD00;border:1px solid #FFDD00">/mnt/dsh_storage/models</code> (via WSL) → 5) <b>Verify:live</b> green = WebUI talks to Linux model</div>
<div id="setup-box" style="background:#000; border:1.5px solid #FFB703; color:#FFDD00; border-radius:8px; padding:12px; display:grid; gap:8px; margin-top:.7rem">
  <div style="grid-column:1 / -1; font-weight:700; color:#FFDD00; border-bottom:1px solid rgba(255,221,0,.18); padding-bottom:.35rem; margin-bottom:.2rem">Linux/Docker Setup</div>
  <ul style="list-style:none; padding:0; margin:0; display:grid; gap:8px">
    <li><div style="display:flex; gap:8px 10px; align-items:center"><div style="flex:0 0 160px; color:#FFDD00; background:rgba(255,255,255,.04); border:1px solid rgba(255,221,0,.18); border-radius:6px; padding:.45rem .6rem">Engine</div><div style="flex:1; color:#FFDD00; background:rgba(255,255,255,.04); border:1px solid rgba(255,221,0,.18); border-radius:6px; padding:.45rem .6rem"><span class="note" style="color:#FFDD00">linux-rocm-docker (Docker)</span></div></div></li>
    <li><div style="display:flex; gap:8px 10px; align-items:center"><div style="flex:0 0 160px; color:#FFDD00; background:rgba(255,255,255,.04); border:1px solid rgba(255,221,0,.18); border-radius:6px; padding:.45rem .6rem">Drive</div><div style="flex:1; color:#FFDD00; background:rgba(255,255,255,.04); border:1px solid rgba(255,221,0,.18); border-radius:6px; padding:.45rem .6rem"><label class="inline" title="How: GET /v1/setup/drives lists >50GB drives with free/total. Does: Lets you pick drive for LLM Linux storage — VHDX updates to X:/dsh_storage.vhdx on change. Changing: Choose your LLM drive (e.g. E:) — no auto-best, you decide; needs Create Drive after."><select id="setup-drive"><option>auto-detecting…</option></select></label></div></div></li>
    <li><div style="display:flex; gap:8px 10px; align-items:center"><div style="flex:0 0 160px; color:#FFDD00; background:rgba(255,255,255,.04); border:1px solid rgba(255,221,0,.18); border-radius:6px; padding:.45rem .6rem">VHDX</div><div style="flex:1; color:#FFDD00; background:rgba(255,255,255,.04); border:1px solid rgba(255,221,0,.18); border-radius:6px; padding:.45rem .6rem"><label class="inline" title="How: String path sent as ?vhdx to _setup_health and as vhdx to create/mount/bootstrap; health won't auto-pick best. Does: Selects which VHDX every check targets (default E:/dsh_storage.vhdx). Changing: Edit to D:/dsh_storage.vhdx to use different drive — health shows vhdxExists for that path, bootstrap/bare will target it."><input id="setup-vhdx" size="22" placeholder="E:/dsh_storage.vhdx"></label></div></div></li>
    <li><div style="display:flex; gap:8px 10px; align-items:center"><div style="flex:0 0 160px; color:#FFDD00; background:rgba(255,255,255,.04); border:1px solid rgba(255,221,0,.18); border-radius:6px; padding:.45rem .6rem">Size</div><div style="flex:1; color:#FFDD00; background:rgba(255,255,255,.04); border:1px solid rgba(255,221,0,.18); border-radius:6px; padding:.45rem .6rem"><label class="inline" title="How: POST /v1/setup/create-vhdx size_gb. Does: Max size for the dynamic sparse VHDX (initially small, grows to this cap). Changing: Pick 50–1000; 250 is typical for a few 30B models."><select id="setup-vhdx-size"><option value="50">50GB</option><option value="100">100GB</option><option value="250" selected>250GB</option><option value="500">500GB</option><option value="1000">1000GB</option></select></label></div></div></li>
    <li><div style="display:flex; gap:8px 10px; align-items:center"><div style="flex:0 0 160px; color:#FFDD00; background:rgba(255,255,255,.04); border:1px solid rgba(255,221,0,.18); border-radius:6px; padding:.45rem .6rem">Mount</div><div style="flex:1; color:#FFDD00; background:rgba(255,255,255,.04); border:1px solid rgba(255,221,0,.18); border-radius:6px; padding:.45rem .6rem"><label class="inline" title="How: Display only — backend hardcodes /mnt/dsh_storage and docker compose mounts /mnt/dsh_storage/models:/workspace/models:ro. Does: Where WSL exposes VHDX ext4 for Docker. Changing: No effect — bootstrap always tries mkdir+mount /dev/sdd1/sdd/sde→/mnt/dsh_storage."><input id="setup-mount" size="16" placeholder="/mnt/dsh_storage"></label></div></div></li>
    <li><div style="display:flex; gap:8px 10px; align-items:center"><div style="flex:0 0 160px; color:#FFDD00; background:rgba(255,255,255,.04); border:1px solid rgba(255,221,0,.18); border-radius:6px; padding:.45rem .6rem">Ready to run?</div><div style="flex:1; color:#FFDD00; background:rgba(255,255,255,.04); border:1px solid rgba(255,221,0,.18); border-radius:6px; padding:.45rem .6rem"><span id="setup-msg" class="note" style="color:#FFDD00"></span></div></div></li>
  </ul>
  <div class="row" style="display:flex; gap:.35rem; flex-wrap:nowrap; padding:8px 0; border-top:1px solid rgba(255,221,0,.12); border-bottom:1px solid rgba(255,221,0,.12); margin:4px 0; overflow:hidden">
    <button onclick="createDrive()" title="How: POST /v1/setup/create-vhdx {{vhdx, size_gb}} → _ensure_vhdx creates selected GB dynamic sparse (New-VHD→diskpart→fsutil) only when you click — no free-space pre-check, OS creates sparse file then formats to ext4. Does: Materialises VHDX at chosen path — no mount/docker. Changing: Must pick drive + size first; if exists returns already." style="flex:1 1 0; min-width:0; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; font-size:.78rem; padding:.3rem .35rem">Create Drive</button>
    <button onclick="mountBare()" title="How: POST /v1/setup/mount-bare vhdx → wsl --mount --vhd {{vhdx}} --bare, checks sdd/sde already. Does: Exposes VHDX as bare block to WSL (no filesystem). Changing: Needs Admin — pops UAC (powershell Start-Process -Verb RunAs); if no vhdx auto-create disabled → 400 need Create Drive first." style="flex:1 1 0; min-width:0; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; font-size:.78rem; padding:.3rem .35rem">Mount AI Drive</button>
    <button onclick="bootstrapDocker()" title="How: POST /v1/setup/bootstrap → checks VHDX exists (fail 400 need Create Drive), bare sdd/sde exists (fail 400 need Mount Admin), mkdir+mount /dev/sdd1→/mnt/dsh_storage, docker compose up dsh-compute-backend (auto-tags server-rocm if missing) + checks /dev/kfd. Does: Mounts ext4 and starts Docker (no Admin). Changing: 400 if not bare/mounted; 500 if /dev/kfd missing → wsl --update + AMD driver." style="flex:1 1 0; min-width:0; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; font-size:.78rem; padding:.3rem .35rem">Bootstrap Docker</button>
    <button onclick="refreshStatus()" title="How: GET /v1/setup/status → health grid + Ready + tier, then harness GET /v1/setup/docker-models (proxies 8000/v1/models, no CORS) + fetch :3000 no-cors (merged Test). Does: Health + Verify:live + Docker/WebUI in one click. Changing: One click updates all." style="flex:1 1 0; min-width:0; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; font-size:.78rem; padding:.3rem .35rem">Refresh Status</button>
  </div>
  <div id="setup-health" style="display:grid; gridTemplateColumns:160px 1fr; gap:8px 10px; border-top:1px solid rgba(255,221,0,.18); padding-top:8px; margin-top:4px"></div>
</div>
<details style="display:none"><summary>Docker details (compose + logs) — now in console (F12)</summary>
<pre id="setup-docker" style="max-height:180px;overflow:auto; display:none">click Refresh to load</pre>
<div class="row" style="gap:.4rem;flex-wrap:wrap;margin-top:.4rem; display:none"><code>wsl --mount --vhd E:/dsh_storage.vhdx --bare</code> → <code>mount /dev/sdX1 /mnt/dsh_storage</code> → <code>docker compose up -d dsh-compute-backend</code> → <code>curl http://127.0.0.1:8000/health</code></div>
<div class="note" style="display:none">Compose: <code>dsh-compute-backend:8000</code> <code>/dev/kfd:/dev/kfd</code> <code>/dev/dri:/dev/dri</code> <code>seccomp:unconfined</code> <code>group_add video/render</code> <code>/mnt/dsh_storage/models:/workspace/models:ro</code> <code>OPENAI_API_BASE_URLS=http://dsh-compute-backend:8000/v1</code></div>
</details>
<div id="setup-tier" class="kv-grid" style="margin-top:.6rem"></div>
<div class="row" style="gap:.6rem;margin-top:.5rem; flex-wrap:wrap">
  <label class="inline">context <select id="setup-ctx" onchange="refreshStatus()"><option value="32768">32k</option><option value="131072">131k (dual)</option><option value="128000">128k</option><option value="1000000">1M</option></select></label>
  <label class="inline"><input id="setup-dual" type="checkbox" onchange="refreshStatus()"> dual 2× GPU</label>
  <span class="note">FP8 KV 0.07/1024 · spill to NVMe · cap from leftover</span>
</div>
<pre id="setup-raw" style="max-height:160px;overflow:auto;display:none"></pre>
<div class="note" style="margin-top:.5rem">Docker <code>GET /health</code> + <code>/v1/models</code> must be 200 for WebUI <code>:3000</code> to list Linux model. VHDX bare bypasses 9P.</div>
</section>
</div>

<div id="tab-agent" class="tabpane" style="display:none">
<section>
<h2 style="margin-top:0">Agent profiles <span class="note">(presets)</span></h2>
<div class="row"><button onclick="loadAgentPresets()">Refresh</button><button onclick="createAgentPreset()">New</button></div>
<div id="agent-presets-list" class="liblist">loading…</div>
<div id="agent-preset-detail" style="display:none"><h3 id="agent-preset-name"></h3><pre id="agent-preset-content" style="max-height:300px;overflow:auto"></pre><div class="row"><button onclick="saveAgentPreset()">Save</button><button onclick="closeAgentPreset()">Close</button><span id="agent-preset-msg" class="note"></span></div></div>
<div class="note">Create your own, customise ones you have — like DSH <code>apps/cli/config/agent-presets</code> (<code>preset.yml</code> + <code>agent.cordis.yml</code>). Duplicate any system preset to <code>~/.dsh/.agent-presets/&lt;id&gt;/</code> and edit.</div>
</section>
</div>

<div id="tab-engines" class="tabpane" style="display:none">
<section>
<h2 style="margin-top:0">Engine profiles — unified</h2>
<form id="engine-profile-form" class="engine-grid" onsubmit="event.preventDefault(); saveEngineProfile();">
  <div class="row" style="gap:.6rem; flex-wrap:wrap">
    <select id="eng-select" style="min-width:180px" onchange="engineSelected()"></select>
    <label class="inline" title="How: Stored via POST /v1/engines default:true, resolved when engine param missing. Does: Default for new conversations. Changing: Check to make this fallback — only one active, persisted to engines.local.json.">default  <input id="eng-default" type="checkbox" onchange="engDirty=true"></label>
    <button type="button" onclick="engineAdd()">+ Add</button>
  </div>

  <!-- Model -->
  <div class="engine-group">
    <h3 title="How: GET /v1/models/local lists GGUFs from models_dir; resolve checks file else needs hf. Does: Sets weights for -m (size_gb for fit). Changing: Pick different GGUF changes VRAM need — Save+Load to restart.">Model </h3>
    <div class="grid-2col">
      <label title="How: GET /v1/models/local scans models_dir. Does: Selects GGUF for -m (advisory, on Start). Changing: Different file → different size_gb/KV fit — Save+Load required.">Model  <select id="eng-model" onchange="engDirty=true; updateFit();"><option value="">— choose local model —</option></select></label>
      <span class="note">Selected: <b id="eng-model-display">—</b> <span id="eng-model-note" class="note" style="margin-left:.5rem"></span></span>
    </div>
  </div>

  <!-- Endpoint -->
  <div class="engine-group">
    <h3>Endpoint</h3>
    <div class="grid-2col">
      <label title="How: Required EngineProfile.name, validated non-empty, key for registry.resolve. Does: Lookup for provider/engine param. Changing: Rename breaks saved refs.">name  <input id="eng-name" size="16" placeholder="local-bonsai" oninput="engDirty=true"></label>
      <label title="How: UI alias not in profile. Does: Cosmetic picker label. Changing: No backend effect.">displayName  <input id="eng-displayName" placeholder="local-bonsai" oninput="engDirty=true"></label>
      <label title="How: Provider base_url → resolve_endpoint → POST /v1/chat/completions. Does: Routes generation. Changing: Wrong URL→502; 1234 vs 1236 hits different ServerInstance.">baseURL  <input id="eng-url" placeholder="http://localhost:1234/v1" oninput="engDirty=true"></label>
      <label title="How: api_key → Bearer, redacted as ***; blank→lm-studio. Does: Auth only. Changing: Needed for hosted, blank fine for local.">apiKey  <input id="eng-apikey" type="password" placeholder="(none)" oninput="engDirty=true"></label>
    </div>
  </div>

  <!-- Context -->
  <div class="engine-group">
    <h3>Context</h3>
    <div class="grid-2col">
      <label title="How: ctx_size → -c (8192 default), fit needs=size+ctx*0.25GB/1k. Does: Caps prompt+completion, KV cost. Changing: 8k→32k +6GB spill→NVMe/oom; >16k Auto forces q8_0.">contextLength  <input id="eng-ctxlen" type="number" value="8192" min="256" step="256" oninput="engDirty=true; updateFit();"></label>
      <div id="eng-fit" class="fit-panel" style="display:block">
        <div class="fit-grid">
          <span id="fit-needs" class="fit-needs">needs —</span>
          <span id="fit-has" class="fit-has">you have —</span>
          <span id="fit-pct" class="fit-pct"></span>
        </div>
        <div class="fit-bar"><div id="fit-fill" class="fit-fill" style="width:0%"></div></div>
        <div class="row fit-controls">
          <label class="inline">context <input id="fit-ctx" type="range" min="2048" max="131072" step="1024" value="8192"> <span id="fit-ctx-label" title="How: fit needs vs hardware via /v1/server/status. Does: Visual guide, disables Load if needs>has. Changing: Slider only visual — Auto/Save commits.">8k</span> → <b id="fit-needs-val">—</b> </label>
          <button type="button" id="eng-load-fit" onclick="engineLoadFromFit()" disabled>Load</button>
        </div>
      </div>
    </div>
  </div>

  <!-- Sampling 4x2 -->
  <div class="engine-group">
    <h3>Sampling</h3>
    <div class="grid-2col sampling-grid">
      <label title="How: payload.temperature 0..2 via sampling, no range check. Does: Logit scale. Changing: 0.2→1.2 more creative/hallucination; 0 deterministic.">temp  <input id="s-temp" type="number" step="0.05" min="0" max="2" oninput="engDirty=true"></label>
      <label title="How: payload.top_p 0..1. Does: Nucleus mass cut. Changing: 1→0.7 tighter, less tail.">top_p  <input id="s-topp" type="number" step="0.05" min="0" max="1" oninput="engDirty=true"></label>
      <label title="How: payload.top_k int. Does: Keep K most likely. Changing: 0→40 prunes weird tokens.">top_k  <input id="s-topk" type="number" oninput="engDirty=true"></label>
      <label title="How: payload.min_p 0..1. Does: Drop <fraction*top. Changing: 0.05 prunes low prob without top_p.">min_p  <input id="s-minp" type="number" step="0.01" oninput="engDirty=true"></label>
      <label title="How: repeat_penalty via llama.cpp. Does: Penalise present. Changing: 1.0→1.2 less looping.">repeat_penalty  <input id="s-rep" type="number" step="0.05" oninput="engDirty=true"></label>
      <label title="How: presence_penalty OpenAI. Does: Once-per-token. Changing: 0→0.6 broader explore.">presence_penalty  <input id="s-pres" type="number" step="0.1" oninput="engDirty=true"></label>
      <label title="How: frequency_penalty growing. Does: Per-repeat. Changing: 0→0.5 curbs loops.">frequency_penalty  <input id="s-freq" type="number" step="0.1" oninput="engDirty=true"></label>
      <label title="How: payload.seed int, blank random. Does: Reproducible. Changing: Fixed → same output.">seed  <input id="s-seed" type="number" oninput="engDirty=true"></label>
    </div>
  </div>

  <!-- Load -->
  <div class="engine-group">
    <h3>Load</h3>
    <div class="grid-2col load-grid">
      <label title="How: -t <n> or omit auto. Does: CPU parallelism. Changing: 4→16 up tok/s, over→down.">threads  <input id="eng-threads" type="number" placeholder="auto" oninput="engDirty=true"></label>
      <label title="How: -ngl <n> 999=all, clamped to est_layers. Does: GPU VRAM linear. Changing: 999→28 on 8GB fits but slower; 999 oom→exit.">gpu_layers  <input id="eng-gpu" type="number" value="999" oninput="engDirty=true"></label>
      <label title="How: auto_fit → ngl from free VRAM + KV (GET /v1/server/status). Does: Offloads overflow layers to CPU/GTT instead of OOM. Changing: on overrides gpu_layers.">auto-fit <input id="eng-autofit" type="checkbox" onchange="engDirty=true"></label>
      <label title="How: visible_devices → HIP/CUDA_VISIBLE_DEVICES for this server. Does: Pins the model to specific card(s). Changing: e.g. 1 = second HIP device (PCI order).">GPU(s) <select id="eng-devices" onfocus="fillGpuDevices()" onchange="engDirty=true"><option value="">auto (all visible)</option></select></label>
      <label title="How: -fa on if set, auto on when ctx>=8192. Does: Faster long ctx. Changing: off→on +10-30% at 32k, needs GPU.">flash_attn  <select id="eng-flash" onchange="engDirty=true"><option value="">off</option><option value="on">on</option><option value="auto">auto</option></select></label>
      <label title="How: -np <n> parallel slots. Does: Concurrent decode, ctx/slots. Changing: 1→4 throughput up, per-slot ctx down.">parallel  <input id="eng-parallel" type="number" placeholder="1" oninput="engDirty=true"></label>
      <label title="How: -b <n> 512 default. Does: Tokens/step RAM. Changing: 512→2048 prompt faster, more VRAM/spill.">batch  <input id="eng-batch" type="number" placeholder="512" oninput="engDirty=true"></label>
      <label title="How: -ub <n> 512. Does: Micro-batch physically. Changing: 512→128 lower peak but more steps; tune with batch.">ubatch  <input id="eng-ubatch" type="number" placeholder="512" oninput="engDirty=true"></label>
      <label title="How: --cache-type-k q8_0/q4_0 else f16, Auto q8_0 if ctx>16384. Does: KV quant saves 50%/75%. Changing: f16→q8 halves KV.">cache K  <select id="eng-ctk" onchange="engDirty=true"><option value="">f16</option><option>q8_0</option><option>q4_0</option></select></label>
      <label title="How: --cache-type-v same. Does: V cache quant. Changing: Same as K.">cache V  <select id="eng-ctv" onchange="engDirty=true"><option value="">f16</option><option>q8_0</option><option>q4_0</option></select></label>
    </div>
  </div>

  <!-- Advanced -->
  <div class="engine-group">
    <h3>Advanced</h3>
    <div class="grid-2col advanced-grid">
      <label title="How: mirostat 0..2 payload. Does: Perplexity target. Changing: 0 off →2 adaptive.">mirostat  <input id="s-miro" type="number" min="0" max="2" oninput="engDirty=true"></label>
      <label title="How: mirostat_tau float. Does: Entropy target. Changing: Higher more surprise.">mirostat_tau  <input id="s-tau" type="number" step="0.1" oninput="engDirty=true"></label>
      <label title="How: mirostat_eta float. Does: Adapt rate. Changing: 0.1 fast vs 0.01 slow.">mirostat_eta  <input id="s-eta" type="number" step="0.01" oninput="engDirty=true"></label>
      <label title="How: stop array payload. Does: Early stop. Changing: Add </s> to truncate.">stop  <input id="s-stop" placeholder="a,b" oninput="engDirty=true"></label>
      <label title="How: --alias <id>. Does: Model id vs path. Changing: Clients see alias.">alias  <input id="eng-alias" placeholder="model id" oninput="engDirty=true"></label>
      <label title="How: kind in llama_cpp,lmstudio,vllm,ollama,hosted validated. Does: Routing hint. Changing: Wrong→422; hosted skips launch.">kind  <select id="eng-kind" onchange="engDirty=true"><option>llama_cpp</option><option>lmstudio</option><option>vllm</option><option>ollama</option><option>hosted</option></select></label>
    </div>
  </div>

  <!-- A/B compare — grid cell with two profile selects, Bench button, winner badge -->
  <div class="engine-group" id="ab-compare">
    <h3>A/B compare</h3>
    <div class="grid-2col">
      <label title="How: POST /v1/engines/ab/bench profile_a,basePort reuses or loads. Does: Bench A vs B. Changing: Pick different to compare tok/s.">A profile  <select id="ab-select-a"></select></label>
      <label title="How: Second on basePort+1. Does: Bench vs A. Changing: Determines B load.">B profile  <select id="ab-select-b"></select></label>
      <label title="How: basePort 1024-65534, B=basePort+1 via _ab_ensure_started. Does: Ports for AB. Changing: 1234→5678 avoids clash.">basePort  <input id="ab-baseport" type="number" value="1234" min="1024" max="65534"></label>
      <div class="row" style="gap:.5rem; align-items:center">
        <button type="button" id="ab-bench-btn" onclick="benchAb()">Bench</button>
        <span id="ab-winner" class="winner-badge">—</span>
        <span id="ab-badge" style="display:none"></span>
      </div>
      <span id="ab-result-a" class="note"></span>
      <span id="ab-result-b" class="note"></span>
      <!-- aliases for alternative test ids -->
      <select id="ab-profile-a" style="display:none"></select>
      <select id="ab-profile-b" style="display:none"></select>
    </div>
  </div>

  <div class="row" style="margin-top:.6rem; gap:.5rem; align-items:center; flex-wrap:wrap"><span id="eng-msg" class="note"></span><button type="submit" id="eng-save">Save</button> <button type="button" id="eng-auto" onclick="engineAuto()" title="How: GET /v1/engines/preset → _auto_preset size_gb+hardware table + est_layers, writes load_options. Does: Sets gpu/context/flash/cache. Changing: Click to fit model to VRAM (e.g. 32b 8GB→12/4k, 4b 8GB→999/32k).">Auto</button> <span id="eng-auto-msg" class="note" style="margin-left:.5rem"></span> <button type="button" onclick="exportEngineProfiles()">Export</button><button type="button" onclick="document.getElementById('import-engine-profiles').click()">Import</button><input type="file" id="import-engine-profiles" accept=".json" style="display:none" onchange="importEngineProfiles(this)"><span id="import-export-msg" class="note"></span></div>
  <pre id="eng-loadopts" style="display:none"></pre>
</form>
</section>
</div>

<div id="tab-library" class="tabpane" style="display:none">
<section>
<h2 style="margin-top:0">Local Library</h2>
<div class="lib-container">
<div class="tabs" style="margin-bottom:.6rem">
<div class="lib-tab active" data-rtab="lib-system" onclick="switchLibraryTab('lib-system', event)" role="button" tabindex="0">System</div>
<div class="lib-tab" data-rtab="lib-linux" onclick="switchLibraryTab('lib-linux', event)" role="button" tabindex="0">Linux / Docker</div>
</div>
<div id="lib-system" class="lib-tabpane">
<div class="note" style="color:#000">System models on Windows host (<span id="library-path-note"></span>) — right-click to move to Linux</div>
<div class="row"><input type="text" id="library-path" placeholder="C:/Users/you/.lmstudio/models" size="38" style="flex:1" title="How: POST /v1/models/local/path writes to harness_state/models_dir.txt, list_local scans there. Does: Sets local library root for GGUF dropdowns. Changing: Point to LM Studio folder to see your models; persists across restarts."> <button onclick="setLibraryPath()">Use this folder</button><span id="import-status" class="note" style="margin-left:.5rem; color:#000"></span></div>
<div class="row" style="margin-top:.6rem"><input type="text" id="library-filter" placeholder="Filter by name…" size="28" style="flex:1" oninput="filterLibrary()" title="How: Client filterLibrary() substring on filenames. Does: Filters list + dropdown options, shows visible/total. Changing: Type qwen to narrow; clear to show all."> <span class="note" id="library-filter-count" style="margin-left:.5rem; color:#000"></span></div>
<div id="local-system" class="liblist">loading…</div>
<div id="local" style="display:none"></div>
</div>
<div id="lib-linux" class="lib-tabpane" style="display:none; margin-top:.8rem; border-top:1.5px solid #000; padding-top:.6rem">
<h3 style="margin:.2rem 0 .4rem; color:#000">Linux / Docker — <code>/mnt/dsh_storage/models</code></h3>
<div class="note" style="color:#000">Right-click to send back to Windows or delete</div>
<div id="local-linux" class="liblist">loading…</div>
</div>
</div>
</section>
</div>

<div id="tab-splinter" class="tabpane" style="display:none">
<section>
<h2 style="margin-top:0">Splinter tuning <span class="note">(new conversations)</span></h2>
<div class="note">Applied when a conversation is created — hit
"New conversation" in the chat pane after changing.</div>
<div class="row">
<label class="inline" title="How: SplinterConfig.max_context (8192 default) caps assembly.py focal budget vs drone budget (1-6k). Does: Token ceiling for curated prompt. Changing: Up → more chunks fit but higher token_count/latency; down → truncates even high-relevance facts.">max_context  <input id="h-maxctx" type="number" size="6"></label>
<label class="inline" title="How: SplinterConfig.max_tokens (None=backend default) → sampling max_tokens via app:1379/stream. Does: Caps reply length. Changing: Up longer answers; down ≤256 on reasoning models → empty_reply_reasoning_starved.">max_tokens  <input id="h-maxtok" type="number" size="5" placeholder="4096 ceiling"></label>
</div>
<div class="row">
<label class="inline" title="How: stale_threshold 20 → decay.py age>20 *0.5 and archive to comb. Does: Stale penalty + archiving gate. Changing: Up zombies linger; down faster forgetting, comb sooner, cleaner but lose mid-horizon.">stale wall  <input id="h-stale" type="number" size="3"></label>
<label class="inline" title="How: dedup_threshold 0.92 → ContextDeduplicator cosine>0.92 keeps denser info. Does: Filters duplicates pre-scoring. Changing: 0.98 keeps near-variants; 0.85 aggressively merges distinct facts.">dedup  <input id="h-dedup" type="number" step="0.01" size="4"></label>
<label class="inline" title="How: drift_threshold 0.6 → TopicDriftDetector 1-cosine(recent,history)>0.6 penalises old topics *0.1. Does: Detects shift. Changing: Up less sensitive (stale surfaces); down isolates recent but hurts cross-topic.">drift  <input id="h-drift" type="number" step="0.05" size="4"></label>
<label class="inline" title="How: remembrance_threshold 0.65 (currently unwired — RemembrancePass hardcoded). Does: Would save at deletion if relevant. Changing: No effect until wired; lower would save more, higher fewer.">remem  <input id="h-remem" type="number" step="0.05" size="4"></label>
</div>
<div class="row">
<label class="inline" title="How: vocab_boost 0.15 via UltraSmallDrone vocab match. Does: Keyword bonus on cosine. Changing: Up domain terms dominate; 0 pure semantic may miss code.">vocab boost  <input id="h-vocab" type="number" step="0.05" size="4"></label>
<label class="inline" title="How: confidence off/single/mcdropout via ultra_small 1 vs 3 passes (std). Does: Drives escalation to medium. Changing: off no escalation fastest; mcdropout enables medium re-score but 3× encode, needs medium drone for effect.">confidence  <select id="h-conf">
<option>off</option><option>single</option><option>mcdropout</option></select></label>
</div>
<div class="row">
<label class="inline"><input id="h-sanitize" type="checkbox" title="How: sanitize_context True → sanitize.py wraps <user_data> + neutralises ignore/system prompts. Does: Prevents injection. Changing: Off raw chunks to LLM (risk); on may alter legit code mentioning system:."> sanitize context </label>
<label class="inline"><input id="h-hedge" type="checkbox" title="How: filter_hedge True → hedges.py 90ch lead-anchored drops before add_chunk (also curate/observe). Does: Drops ~50% refusals. Changing: Off stores hedges → later retrieved loops; on may drop edge factual hedges."> filter hedge replies </label>
<label class="inline"><input id="h-medium" type="checkbox" title="How: enable_medium False → MediumDrone graphcodebert 400MB vs stub 0.5. Does: Second-pass for score>=2 complex. Changing: On better recall +20-50ms, VRAM heavy; off ultra only."> medium drone </label>
</div>
<details><summary>Comb (P11 surplus tier)</summary>
<div class="row">
<label class="inline"><input id="h-comb" type="checkbox" title="How: comb_enabled False → CombStore JSONL at harness_comb, evicted not deleted. Does: Surplus tier for returns. Changing: On long-horizon recall, forces comb_dir set."> enabled </label>
<label class="inline" title="How: comb_top_k 5 → comb.retrieve k lexically ranked. Does: Competes vs store. Changing: Up more resurrect candidates but budget competition; down fewer faster.">top_k  <input id="h-combk" type="number" size="3"></label>
<label class="inline" title="How: comb_gate 0.85 → fires when top_raw<0.85 or echo. Does: Gate decides comb consult (~820ms). Changing: 0.70 fires often crowding; 0.95 rarely misses returns.">gate  <input id="h-combgate" type="number" step="0.05" size="4"></label>
<label class="inline" title="How: comb_max_records 2000 → prune LRU cap + 1000 turn prune. Does: Disk cap. Changing: Up longer horizon but disk; down sooner forgetting.">max records  <input id="h-combmax" type="number" size="5"></label>
<label class="inline"><input id="h-combrel" type="checkbox" title="How: comb_relevant_only True → archive only once_curated (relevance_history). Does: Lean archive. Changing: Off archives every evicted (more noise/recall)."> curated-only </label>
</div>
</details>
<div class="row"><span id="splinter-msg" class="note"></span>
<button onclick="resetHiveDefaults()">Reset to defaults</button></div>
</section>
</div>

<div id="tab-providers" class="tabpane" style="display:none">
<section>
<h2 style="margin-top:0">Providers <span class="note">(OpenAI-compatible)</span></h2>
<div id="prov-list"></div>




<div class="row">
<button onclick="providerAdd()">+ Add provider</button>
<button onclick="saveProviders()">Save providers</button>
<span id="prov-msg" class="note"></span></div>
<div class="note">Keys echo as *** — leave untouched rows as-is to keep the
stored secret; type a new key to replace it. Saved to
providers.local.json (gitignored).</div>
</section>
</div>

<div id="tab-hub" class="tabpane" style="display:none">
<section>
<h2 style="margin-top:0">Hugging Face hub <span class="note">(live)</span></h2>
<div class="row"><span class="sugwrap" style="width:100%"><input id="q" placeholder="search gguf repos…" style="width:100%" title="How: GET /v1/models/hub?q=2+chars → HF API 8 results. Does: Live search repos. Changing: Type to get suggestions; click to fill repo field.">
<div class="sugbox" id="sug-q"></div></span></div>
<datalist id="repo-suggestions"></datalist>
<pre id="hub">(search above)</pre>
<div class="row"><span class="sugwrap" style="width:100%"><input id="drepo" placeholder="repo id (type for suggestions)" style="width:100%" title="How: Input for --hf-repo, suggestions from hub search. Does: Sets HF repo for download/start. Changing: Different repo → different weights; needs exact file next.">
<div class="sugbox" id="sug-drepo"></div></span><br>
<input id="dfile" placeholder="file.gguf" size="24" title="How: Value for --hf-file exact filename. Does: Picks file inside repo. Changing: Wrong name → 400 not found; copy from search.">
<button onclick="download(this)">Download</button></div>
<pre id="downloads"></pre>
</section>
</div>

<div id="tab-inspect" class="tabpane" style="display:none">
<section>
<h2 style="margin-top:0">Prompt inspector <span class="note">(last turn)</span></h2>
<div id="inspect-summary" class="kv-grid"></div>
<h2 style="font-size:.95rem;margin-top:1rem">Selected chunks (in the context)</h2>
<div id="inspect-selected" class="inspector-list"></div>
<h2 style="font-size:.95rem;margin-top:1rem">Dropped chunks <span class="note" id="inspect-dropped-n"></span></h2>
<div id="inspect-dropped" class="inspector-list"></div>
<h2 style="font-size:.95rem;margin-top:1rem">Assembled context (preview)</h2>
<pre id="inspect-assembled" style="max-height:200px"></pre>
<h2 style="font-size:.95rem;margin-top:1rem">Stage timings</h2>
<pre id="inspect-timings"></pre>
</section>
</div>

<div id="tab-settings" class="tabpane">
<section>
<h2 style="margin-top:0">Settings</h2>
<div class="row"><label class="inline" title="How: setTheme() writes localStorage. Does: Switches CSS tokens. Changing: Affects only UI chrome, no backend.">Theme  <select id="settings-theme" onchange="setTheme(this.value)"><option value="light">Light</option><option value="dark">Dark</option><option value="system">System</option></select></label></div>
<div class="row"><span class="note">Engine profiles, model library, and provider settings are in their own tabs. This tab will hold general Studio preferences.</span></div>
</section>
</div>

{render_training_pane()}

</div>

<!-- ==================== MIDDLE: loaded AI chat ==================== -->
<div class="col mid">
<section class="chatpane">
<div id="sess-tabs" class="sesstabs"></div>
<div class="chat-head">
<h2 id="chat-title">Loaded model</h2>
<div class="chat-controls">
<span class="modesel">
<label class="inline">model <select id="chat-provider" onchange="saveConvProvider(this.value)" title="How: select saves to localStorage splinter-console-convprov, sent as provider on /v1/splinter/turn vs /v1/agent/stream, swaps Splinter.backend via registry. Does: Per-conversation model/endpoint. Changing: Pick different engine → different base_url/model for this tab only."></select></label>
<label class="inline"><input type="radio" name="chatmode" value="splinter" checked title="How: POST /v1/splinter/stream → splinter.process_turn curates assembled_content → single generate. Does: No tools, fast curated. Changing: Good for chat/memory; Agent needed for bash/code."> Splinter </label>
<label class="inline"><input type="radio" name="chatmode" value="agent" title="How: POST /v1/agent/stream → DshAgentService loop with tools/session log. Does: Full agent (bash/fs/web/subagent). Changing: Use for code/tasks; slower but multi-step."> Agent (dsh) </label>
</span>
<button onclick="newConversation()" title="How: newConversation() creates console-<uuid>, localStorage SESS_KEY, clears chatlog. Does: New splinter conversation. Changing: Old tab kept; config changes apply only after New.">New conversation</button>
</div>
</div>
<div id="chatlog" class="chatlog"></div>
<div class="composer">
<span class="sugwrap composer-input"><input id="chatin" placeholder="Talk to the loaded AI…  (/ for commands)" title="How: Enter → chatSubmit routes / → /v1/commands/run else sendChat/sendAgent via Splinter vs Agent. Does: Sends prompt through splinter curate+generate or agent loop. Changing: /command vs message decides path."
       onkeydown="if (event.key === 'Enter') chatSubmit()" autocomplete="off">
<div class="sugbox" id="sug-chat"></div></span>
<button id="sendbtn" onclick="chatSubmit()">Send</button>
<button id="stopbtn" onclick="cancelStream()">Stop</button>
<button id="savebtn" onclick="saveSession()" title="How: saveSession prompts title → sessions[convId].title → localStorage. Does: Persists tab+transcript (restoreTranscript caps 400). Changing: Name to keep; close × deletes + POST /v1/splinter/reset.">Save session</button>
<button onclick="newConversation()" title="How: Same as top New. Does: New session immediately. Changing: Same effect.">+ New session</button>
</div>
<div class="note"><b>Splinter</b>: direct curated generation. <b>Agent (dsh)</b>:
the full DeepSeek Harness agent loop — bash/files/code tools, multi-step
turns, durable session log.</div>
</section>
</div>

<!-- ==================== RIGHT: Server (standalone, no tabs) ==================== -->
<div class="col">
<section>
<h2 style="margin-top:0">Launch</h2>
<div class="row"><button onclick="api('/v1/server/status').then(s => show('status', s))">Refresh</button>
<button onclick="api('/v1/server/stop', 'POST').then(() => refresh())">Stop</button></div>
<div class="row">
<label class="inline" style="flex:1">Model <select id="launch-model-select" style="flex:1;min-width:220px"><option value="">— choose local model —</option></select></label>
<span class="sugwrap"><input id="model" placeholder="or type path" size="18" list="local-suggestions" title="How: launchBody model → resolve_model local else needs hf_repo/file. Does: Sets -m path. Changing: Pick dropdown or type path; blank → must fill hf below or 400.">
<datalist id="local-suggestions"></datalist></span>
<label class="inline" title="How: ctx_size → -c (8192), KV 0.07/1024 GB tier vs 0.25GB/1k fit. Does: Caps prompt+reply. Changing: 8k→32k +2GB KV → spill/oom; >16k Auto→q8_0.">ctx  <input id="ctx" type="number" value="8192" size="4"></label>
<label class="inline" title="How: ngl → -ngl, 999=all clamped to est_layers. Does: GPU offload. Changing: 999 needs VRAM≥size+KV else exit; lower fits but slower.">gpu  <input id="ngl" type="number" value="999" size="3"></label>
<label class="inline" title="How: api_key → --api-key + provider auth. Does: Protects server. Changing: Set → clients need Bearer.">api-key  <input id="l-apikey" size="10" placeholder="(none)"></label><br>
<span class="sugwrap"><input id="hfrepo" placeholder="--hf-repo (type to search)" size="30">
<div class="sugbox" id="sug-hfrepo"></div></span>
<input id="hffile" placeholder="--hf-file" size="18" title="How: hffile → --hf-file with hfrepo. Does: Direct HF pull. Changing: Both needed if no local model.">
<button onclick="startServer(this)">Start</button></div>
<details><summary>Advanced launch flags</summary>
<div class="row">
<label class="inline" title="How: -t <n> or omit auto. Does: CPU parallelism. Changing: 4→16 up tok/s, over→down.">threads  <input id="l-threads" type="number" size="3" placeholder="auto"></label>
<label class="inline"><input id="l-fa" type="checkbox" title="How: -fa on if set, auto on when ctx>=8192. Does: Faster long ctx. Changing: off→on +10-30% at 32k, needs GPU."> flash-attn </label>
<label class="inline" title="How: -np <n> parallel slots. Does: Concurrent decode, ctx/slots. Changing: 1→4 throughput up, per-slot ctx down.">parallel  <input id="l-parallel" type="number" size="2" placeholder="1"></label>
<label class="inline"><input id="l-mlock" type="checkbox" title="How: mlock → --mlock. Does: Pins RAM, no swap. Changing: Slower start, stable."> mlock </label>
<label class="inline"><input id="l-nommap" type="checkbox" title="How: no_mmap → --no-mmap. Does: Full RAM vs mmap. Changing: More RSS, avoids faults."> no-mmap </label>
</div>
<div class="row">
<label class="inline" title="How: cache K → --cache-type-k f16/q8_0/q4_0, Auto q8_0 >16k. Does: Halves KV. Changing: f16→q8 saves 50%.">kv-K  <select id="l-ctk"><option value="">f16</option><option>q8_0</option><option>q4_0</option></select></label>
<label class="inline" title="How: --cache-type-v same. Does: V cache quant. Changing: Same as K.">kv-V  <select id="l-ctv"><option value="">f16</option><option>q8_0</option><option>q4_0</option></select></label>
<label class="inline" title="How: -b <n> 512 default. Does: Tokens/step RAM. Changing: 512→2048 prompt faster, more VRAM/spill.">batch  <input id="l-batch" type="number" size="4" placeholder="512"></label>
<label class="inline" title="How: -ub <n> 512. Does: Micro-batch physically. Changing: 512→128 lower peak but more steps; tune with batch.">ubatch  <input id="l-ubatch" type="number" size="4" placeholder="512"></label>
<label class="inline" title="How: --alias <id>. Does: Model id vs path. Changing: Clients see alias.">alias  <input id="l-alias" size="14" placeholder="model id"></label>
</div>
</details>
<pre id="status">loading…</pre>
<div id="instances"></div>
<section id="proc-manager" style="margin-top:1rem">
<h3 style="margin:.6rem 0 .3rem">Process manager <span class="note">CPU/RAM · kill</span></h3>
<div class="row" style="gap:.5rem;align-items:center"><button onclick="refreshProcesses()">Refresh processes</button><span id="proc-msg" class="note"></span><span id="proc-sidecar" class="note" style="margin-left:.4rem"></span></div>
<table style="width:100%;margin-top:.4rem"><thead><tr><th>PID</th><th>Kind</th><th>Name</th><th title="CPU % (process)">CPU%</th><th title="Resident memory MB">RAM MB</th><th>Port</th><th>Status</th><th></th></tr></thead><tbody id="proc-table"><tr><td colspan="8" class="note">loading…</td></tr></tbody></table>
</section>
<pre id="srvlog" title="llama_server.log tail">log: loading…</pre>
</section>

</div></div>

<script>
let convId = localStorage.getItem('splinter-console-conv');
if (!convId) {{
  convId = 'console-' + crypto.randomUUID().slice(0, 8);
  localStorage.setItem('splinter-console-conv', convId);
}}
let hiveOverrides = {{}};
// engDirty declared in unified grid block above

async function api(path, method, body, signal) {{
  const headers = {{'content-type': 'application/json'}};
  const token = localStorage.getItem('splinter-token');
  if (token) headers['x-splinter-token'] = token;
  const opts={{method: method || 'GET', headers}};
  if(body !== undefined && body !== null) opts.body=JSON.stringify(body);
  else if(method === 'POST' && body === undefined) opts.body='{{}}';
  if(signal) opts.signal=signal;
  const r = await fetch(path, opts);
  if (r.status === 401) {{
    const t = prompt('This server requires an access token (HARNESS_TOKEN):');
    if (t !== null) {{ localStorage.setItem('splinter-token', t); }}
    throw new Error('unauthorized — token saved, retry');
  }}
  const t = await r.text();
  if (!r.ok) throw new Error(r.status + ': ' + t.slice(0, 400));
  return t.startsWith('{{') ? JSON.parse(t) : t;
}}
window.onerror=(msg, src, line, col, err)=>{{ console.error('ONERROR', msg, src+':'+line+':'+col, err); const t=document.getElementById('top-right-status'); if(t) t.textContent='JS ERROR: '+msg.slice(0,120); alert('JS ERROR: '+msg+'\\n'+src+':'+line); }};
window.onunhandledrejection=(e)=>{{ console.error('UNHANDLED REJECTION', e.reason); alert('UNHANDLED: '+String(e.reason).slice(0,300)); }};
function show(id, obj) {{ 
  try{{
    const el=document.getElementById(id);
    if(!el){{ console.warn('show missing', id); return; }}
    el.textContent=typeof obj === 'string' ? obj : JSON.stringify(obj, null, 1);
  }} catch(err){{ console.error('show failed', id, err); }}
}}
function val(id) {{ const el=document.getElementById(id); if(!el){{ console.warn('val missing', id); return ''; }} return el.value.trim(); }}
function num(id) {{ const v = val(id); return v === '' ? null : +v; }}
function fmtCap(n) {{ if(n==null||n==='-') return '-'; n=Number(n); if(n>=1000000) return (n/1000000).toFixed(1)+'M'; if(n>=1000) return (n/1000).toFixed(1)+'K'; return String(n); }}

/* ------------------------------ tabs -------------------------------- */
for (const btn of document.querySelectorAll('.tab[data-tab]')) {{
  btn.addEventListener('click', (e) => {{
    try{{
      console.log('outer tab click', btn.dataset.tab);
      for (const b of document.querySelectorAll('.tab[data-tab]')) b.classList.remove('active');
      for (const p of document.querySelectorAll('.tabpane')) p.style.display = 'none';
      btn.classList.add('active');
      const pane=document.getElementById(btn.dataset.tab);
      if(pane) pane.style.display = '';
      // Defer library load to avoid blocking tab switch
      if (btn.dataset.tab === 'tab-library') setTimeout(loadLibrary, 10);
      else if (btn.dataset.tab === 'tab-setup') {{ loadSetup(); loadDrives(); }}
      else if (btn.dataset.tab === 'tab-agent') loadAgentPresets();
      else if (btn.dataset.tab === 'tab-engines') loadEngines();
      else if (btn.dataset.tab === 'tab-splinter') loadHiveDefaults();
      else if (btn.dataset.tab === 'tab-providers') loadProviders();
    }} catch(err){{ console.error('tab click failed', err); alert('tab error: '+err.message); }}
  }});
}}
let _libCache={{system: null, linux: null, ts: 0}};
function switchLibraryTab(tab, ev) {{
  if(ev) {{ ev.preventDefault(); ev.stopPropagation(); ev.stopImmediatePropagation(); }}
  // Ensure outer Local Library stays open
  const outer=document.getElementById('tab-library');
  if(outer) outer.style.display='';
  // Keep outer tab active
  for(const b of document.querySelectorAll('.tab[data-tab]')) b.classList.toggle('active', b.dataset.tab==='tab-library');
  const prevSel=window._selectedFile;
  for (const b of document.querySelectorAll('[data-rtab^="lib-"]')) b.classList.remove('active');
  for (const p of document.querySelectorAll('.lib-tabpane')) p.style.display = 'none';
  const btn=document.querySelector(`[data-rtab="${{tab}}"]`);
  const pane=document.getElementById(tab);
  if(btn) btn.classList.add('active');
  if(pane) pane.style.display = '';
  const now=Date.now();
  const fresh=now - _libCache.ts < 30000;
  if (tab === 'lib-system') {{
    if(_libCache.system && fresh) renderLibrary('local-system', _libCache.system, 'system');
    else loadLibrarySystem();
  }}
  if (tab === 'lib-linux') {{
    if(_libCache.linux && fresh) renderLibrary('local-linux', _libCache.linux, 'linux');
    else loadLibraryLinux();
  }}
  // Preserve selection across tab switches — restore after render
  setTimeout(()=>{{
    if(prevSel) {{
      window._selectedFile=prevSel;
      for(const id of ['local-system','local-linux']) {{
        const w=document.getElementById(id);
        if(!w) continue;
        for(const r of w.children) r.classList.toggle('selected', r.dataset.file===prevSel);
      }}
    }}
  }}, 10);
  return false;
}}

/* ------------------------------ setup wizard (splinter console) ------------------------------ */
async function loadSetup() {{
  const ctxSel = document.getElementById('setup-ctx');
  const dualCb = document.getElementById('setup-dual');
  const ctx = ctxSel ? ctxSel.value : '32768';
  const dual = dualCb ? dualCb.checked : false;
  await refreshSetup(ctx, dual);
}}
async function refreshSetup(ctx, dual) {{
  if (ctx === undefined) {{
    const c = document.getElementById('setup-ctx');
    ctx = c ? c.value : '32768';
  }}
  if (dual === undefined) {{
    const d = document.getElementById('setup-dual');
    dual = d ? d.checked : false;
  }}
  const msg = document.getElementById('setup-msg');
  const healthEl = document.getElementById('setup-health');
  const tierEl = document.getElementById('setup-tier');
  const rawEl = document.getElementById('setup-raw');
  if (msg) msg.textContent = 'loading…';
  try {{
    const vhdxQ = document.getElementById('setup-vhdx')?.value.trim() || '';
    const modelGbQ = document.getElementById('setup-model-gb')?.value.trim() || '';
    const s = await api('/v1/setup/status?context=' + encodeURIComponent(ctx) + '&dual=' + (dual ? 'true' : 'false') + (vhdxQ ? '&vhdx=' + encodeURIComponent(vhdxQ) : '') + (modelGbQ ? '&model_gb=' + encodeURIComponent(modelGbQ) : ''));
    const eng = document.getElementById('setup-engine');
    if (eng && s.state) eng.value = s.state.engine || 'windows-vulkan';
    const vhdx = document.getElementById('setup-vhdx');
    if (vhdx && s.state) vhdx.value = s.state.vhdxPath || 'E:/dsh_storage.vhdx';
    const mdir = document.getElementById('setup-modelsdir');
    if (mdir && s.state) mdir.value = s.state.modelsDir || 'E:/models';
    const mp = document.getElementById('setup-mount');
    if (mp && s.state) mp.value = s.state.mountPoint || '/mnt/dsh_storage';
    const lin = s.health?.linux || {{}};
    const win = s.health?.windows || {{}};
    if (healthEl && s.health) {{
      const orderFor=(key, ok)=> {{
        if (ok) return '';
        if (key==='vhdx-missing') return ' → Fix: 1. Create Drive';
        if (key==='vhdx-not-mounted') return ' → Fix: 1. Mount AI Drive → 2. Bootstrap Docker';
        if (key==='linux-stopped') {{
          if (!lin.vhdxExists) return ' → Fix: 1. Create Drive → 2. Mount AI Drive → 3. Bootstrap Docker';
          if (!lin.vhdxMounted) return ' → Fix: 1. Mount AI Drive → 2. Bootstrap Docker';
          return ' → Fix: 1. Bootstrap Docker';
        }}
        if (key==='docker-down') {{
          if (!lin.vhdxExists) return ' → Fix: 1. Create Drive → 2. Mount AI Drive → 3. Bootstrap Docker';
          if (!lin.vhdxMounted) return ' → Fix: 1. Mount AI Drive → 2. Bootstrap Docker';
          return ' → Fix: 1. Bootstrap Docker';
        }}
        if (key==='complete' || key==='ready') {{
          if (!lin.vhdxExists) return ' → Fix: 1. Create Drive → 2. Mount AI Drive → 3. Bootstrap Docker';
          if (!lin.vhdxMounted) return ' → Fix: 1. Mount AI Drive → 2. Bootstrap Docker';
          if (!lin.shardsFound) return ' → Fix: add .gguf to /mnt/dsh_storage/models (no button)';
          if (!lin.dockerRunning) return ' → Fix: 1. Bootstrap Docker';
          return ' → Fix: check diskFull';
        }}
        return '';
      }};
      const st=(ok,msg, key)=> {{
        const tip = msg + (ok ? '' : orderFor(key, ok));
        const esc = tip.replace(/"/g, '&quot;');
        return ok?`<b style="color:#157a3e" title="${{esc}}">✔ ${{msg.split(' — ')[0]}}</b>`:`<b style="color:#b3372c" title="${{esc}}">✘ ${{msg}}</b>`;
      }};
      healthEl.style.display='grid'; healthEl.style.gridTemplateColumns='160px 1fr'; healthEl.style.gap='8px 10px'; healthEl.style.padding='0'; healthEl.style.background='transparent'; healthEl.style.border='none'; healthEl.style.color='#FFDD00';
      healthEl.innerHTML = ''
        + '<div style="color:#FFDD00; background:rgba(255,255,255,.04); border:1px solid rgba(255,221,0,.18); border-radius:6px; padding:.45rem .6rem">Windows</div><div style="color:#FFDD00; background:rgba(255,255,255,.04); border:1px solid rgba(255,221,0,.18); border-radius:6px; padding:.45rem .6rem">' + st(win.state==='running', win.state==='running'?'running — Windows sidecar on :'+(win.port||8765):'Windows not running — start Splinter Studio', 'windows') + '</div>'
        + '<div style="color:#FFDD00; background:rgba(255,255,255,.04); border:1px solid rgba(255,221,0,.18); border-radius:6px; padding:.45rem .6rem">Linux</div><div style="color:#FFDD00; background:rgba(255,255,255,.04); border:1px solid rgba(255,221,0,.18); border-radius:6px; padding:.45rem .6rem">' + st(lin.state==='running', lin.state==='running'?'Linux running — Docker on :'+(lin.port||8000):'Linux stopped — Bootstrap Docker', 'linux-stopped') + '</div>'
        + '<div style="color:#FFDD00; background:rgba(255,255,255,.04); border:1px solid rgba(255,221,0,.18); border-radius:6px; padding:.45rem .6rem">VHDX</div><div style="color:#FFDD00; background:rgba(255,255,255,.04); border:1px solid rgba(255,221,0,.18); border-radius:6px; padding:.45rem .6rem">' + st(lin.vhdxExists, lin.vhdxExists?(lin.vhdxMounted?'VHDX mounted — ready':'VHDX exists but not mounted — click Mount AI Drive and approve UAC'):'VHDX missing — click Create Drive and pick size', lin.vhdxExists ? 'vhdx-not-mounted' : 'vhdx-missing') + '</div>'
        + '<div style="color:#FFDD00; background:rgba(255,255,255,.04); border:1px solid rgba(255,221,0,.18); border-radius:6px; padding:.45rem .6rem">Shards</div><div style="color:#FFDD00; background:rgba(255,255,255,.04); border:1px solid rgba(255,221,0,.18); border-radius:6px; padding:.45rem .6rem">' + st(lin.shardsFound, lin.shardsFound?'Shards found — '+(lin.shardPath||'').split('/').pop():'Shards missing — add .gguf to /mnt/dsh_storage/models via WSL or move from System', 'shards') + '</div>'
        + '<div style="color:#FFDD00; background:rgba(255,255,255,.04); border:1px solid rgba(255,221,0,.18); border-radius:6px; padding:.45rem .6rem">Docker 8000</div><div style="color:#FFDD00; background:rgba(255,255,255,.04); border:1px solid rgba(255,221,0,.18); border-radius:6px; padding:.45rem .6rem">' + st(lin.dockerRunning, lin.dockerRunning?'Docker healthy — :8000 200':'Docker down — Bootstrap Docker or docker compose up dsh-compute-backend', 'docker-down') + '</div>'
        + '<div style="color:#FFDD00; background:rgba(255,255,255,.04); border:1px solid rgba(255,221,0,.18); border-radius:6px; padding:.45rem .6rem">Complete</div><div style="color:#FFDD00; background:rgba(255,255,255,.04); border:1px solid rgba(255,221,0,.18); border-radius:6px; padding:.45rem .6rem">' + st(s.complete, s.complete?'Complete — WebUI :3000 → :8000 ready':'Not complete — fix VHDX/mount/shards/docker above', 'complete') + '</div>';
    }}
    if (tierEl && s.tier) {{
      const m = s.tier.metrics || {{}};
      const f = s.tier.flags || {{}};
      tierEl.style.display='grid'; tierEl.style.gridTemplateColumns='160px 1fr'; tierEl.style.gap='8px 10px'; tierEl.style.padding='12px'; tierEl.style.background='#000'; tierEl.style.border='1.5px solid #FFB703'; tierEl.style.borderRadius='8px'; tierEl.style.color='#FFDD00';
      tierEl.innerHTML = ''
        + '<div style="color:#FFDD00; background:rgba(255,255,255,.04); border:1px solid rgba(255,221,0,.18); border-radius:6px; padding:.45rem .6rem" title="Combined VRAM — cross-platform">VRAM</div><div style="color:#FFDD00; background:rgba(255,255,255,.04); border:1px solid rgba(255,221,0,.18); border-radius:6px; padding:.45rem .6rem">' + (m.tier1VramGb ?? '-') + ' GB</div>'
        + '<div style="color:#FFDD00; background:rgba(255,255,255,.04); border:1px solid rgba(255,221,0,.18); border-radius:6px; padding:.45rem .6rem" title="Total RAM — cross-platform">RAM</div><div style="color:#FFDD00; background:rgba(255,255,255,.04); border:1px solid rgba(255,221,0,.18); border-radius:6px; padding:.45rem .6rem">' + (m.tier2RamGb ?? '-') + ' GB</div>'
        + '<div style="color:#FFDD00; background:rgba(255,255,255,.04); border:1px solid rgba(255,221,0,.18); border-radius:6px; padding:.45rem .6rem; display:flex; flex-direction:column; gap:.35rem" title="T3 spill estimator: model+KV - VRAM - RAM. Change model GB here."><span>T3 spill estimator</span><label class="inline" style="margin:0; display:flex; gap:.2rem; align-items:center; background:rgba(0,0,0,.25); border:1px solid rgba(255,221,0,.18); border-radius:4px; padding:.15rem .3rem; width:fit-content">model <input id="setup-model-gb" type="number" value="' + (m.modelGb ?? 104) + '" min="1" max="2000" step="1" style="width:45px; text-align:right; background:#000; color:#FFDD00; border:1px solid rgba(255,221,0,.4); border-radius:3px; padding:.1rem .2rem; -moz-appearance:textfield; appearance:textfield" onchange="refreshStatus()"><span style="color:#FFDD00; opacity:.9; font-size:.85em">GB</span></label></div><div style="color:#FFDD00; background:rgba(255,255,255,.04); border:1px solid rgba(255,221,0,.18); border-radius:6px; padding:.45rem .6rem">' + (m.tier3NvmeGb ?? '-') + ' GB</div>'
        + '<div style="color:#FFDD00; background:rgba(255,255,255,.04); border:1px solid rgba(255,221,0,.18); border-radius:6px; padding:.45rem .6rem" title="Max context from leftover without clamp">Max context</div><div style="color:#FFDD00; background:rgba(255,255,255,.04); border:1px solid rgba(255,221,0,.18); border-radius:6px; padding:.45rem .6rem">' + fmtCap(f.recommendCap) + '</div>'
        + '<div style="color:#FFDD00; background:rgba(255,255,255,.04); border:1px solid rgba(255,221,0,.18); border-radius:6px; padding:.45rem .6rem" title="Effective bandwidth weighted by tiers">Speed</div><div style="color:#FFDD00; background:rgba(255,255,255,.04); border:1px solid rgba(255,221,0,.18); border-radius:6px; padding:.45rem .6rem">' + (m.estEffectiveBw ?? '-') + ' GB/s</div>'
        + '<div style="color:#FFDD00; background:rgba(255,255,255,.04); border:1px solid rgba(255,221,0,.18); border-radius:6px; padding:.45rem .6rem" title="Space left after spill">Free space</div><div style="color:#FFDD00; background:rgba(255,255,255,.04); border:1px solid rgba(255,221,0,.18); border-radius:6px; padding:.45rem .6rem">' + (m.freeAfterSpillGb ?? '-') + ' GB</div>';
    }}
    if (rawEl) {{
      rawEl.textContent = JSON.stringify(s, null, 1);
      rawEl.style.display = 'none';
      console.log('[Splinter] setup status', s);
    }}
    const dockerEl = document.getElementById('setup-docker');
    if (dockerEl) {{
      const lin2 = s.health.linux || {{}};
      dockerEl.textContent = 'VHDX ' + (lin2.vhdxExists ? 'exists' : 'missing') + ' (' + (s.state.vhdxPath || '-') + ')\\n'
        + 'Mount ' + (lin2.vhdxMounted ? 'mounted' : 'not mounted') + ' → ' + (s.state.mountPoint || '/mnt/dsh_storage') + '\\n'
        + 'Shards ' + (lin2.shardsFound ? 'found ' + (lin2.shardPath || '').split('/').pop() : 'missing') + '\\n'
        + 'Docker http://127.0.0.1:8000/health → ' + (lin2.dockerRunning ? '200 healthy' : 'down — docker compose up dsh-compute-backend') + '\\n'
        + 'WebUI :3000 → dsh-compute-backend:8000/v1 ' + (lin2.dockerRunning ? 'route ready' : 'route down');
      console.log('[Splinter] Docker details', dockerEl.textContent);
    }} else {{
      const lin2 = s.health.linux || {{}};
      console.log('[Splinter] Docker details', 'VHDX ' + (lin2.vhdxExists ? 'exists' : 'missing') + ' (' + (s.state.vhdxPath || '-') + ') — Mount ' + (lin2.vhdxMounted ? 'mounted' : 'not mounted'));
    }}
    if (msg) {{
      const readyTip = s.complete ? 'Ready to run — WebUI :3000 → :8000 ready' : 'Not ready — fix: ' + (!lin.vhdxExists ? '1. Create Drive' : !lin.vhdxMounted ? '1. Mount AI Drive → 2. Bootstrap Docker' : !lin.shardsFound ? 'add .gguf to /mnt/dsh_storage/models (no button)' : !lin.dockerRunning ? '1. Bootstrap Docker' : 'check diskFull');
      const readyEsc = readyTip.replace(/"/g, '&quot;');
      msg.innerHTML = s.complete ? '<b style="color:#157a3e" title="'+readyEsc+'">✔</b>' : '<b style="color:#b3372c" title="'+readyEsc+'">✘</b>';
      console.log('[Splinter] Ready', msg.textContent);
    }}
    try {{ if (typeof updateFit === 'function') updateFit(); }} catch(e) {{}}
  }} catch(e) {{
    if (msg) msg.textContent = 'load failed: ' + String(e).slice(0,120);
    if (healthEl) healthEl.textContent = String(e).slice(0,200);
  }}
}}
async function verifySetup() {{
  const msg = document.getElementById('setup-msg');
  const rawEl = document.getElementById('setup-raw');
  if (msg) msg.innerHTML = '<b style="color:#5a6b7d">…</b> verifying…';
  try {{
    const ctx = document.getElementById('setup-ctx')?.value || '32768';
    const dual = document.getElementById('setup-dual')?.checked ? 'true' : 'false';
    const vhdxV = document.getElementById('setup-vhdx')?.value.trim() || '';
    const modelGbV = document.getElementById('setup-model-gb')?.value.trim() || '';
    const s = await api('/v1/setup/status?context=' + encodeURIComponent(ctx) + '&dual=' + dual + (vhdxV ? '&vhdx=' + encodeURIComponent(vhdxV) : '') + (modelGbV ? '&model_gb=' + encodeURIComponent(modelGbV) : ''));
    const lin = s.health.linux || {{}};
    const ok = lin.vhdxExists && lin.vhdxMounted && lin.shardsFound && lin.dockerRunning && !s.tier.flags.diskFull;
    if (msg) msg.innerHTML = ok ? '<b style="color:#157a3e">✔</b> verify:live LINKED — ready to launch (32k)' : '<b style="color:#b3372c">✘</b> verify:live NOT LINKED — fix: ' + (!lin.vhdxExists ? 'VHDX missing' : !lin.vhdxMounted ? 'not mounted → wsl --mount --vhd E:/dsh_storage.vhdx --bare' : !lin.shardsFound ? 'shards missing at ' + (s.state.mountPoint || '/mnt/dsh_storage') : !lin.dockerRunning ? 'docker 8000 down → docker compose up dsh-compute-backend' : s.tier.flags.diskFull ? 'disk >80% full' : 'check');
    if (rawEl) {{
      rawEl.textContent = JSON.stringify(s, null, 1);
      rawEl.style.display = rawEl.style.display === 'none' ? 'block' : 'none';
    }}
    try {{
      const vhdxH = document.getElementById('setup-vhdx')?.value.trim() || '';
      const h = await api('/v1/setup/health' + (vhdxH ? '?vhdx=' + encodeURIComponent(vhdxH) : ''));
      if (rawEl && rawEl.style.display !== 'none') rawEl.textContent += '\\n\\nhealth: ' + JSON.stringify(h, null, 1);
    }} catch(e) {{}}
  }} catch(e) {{
    if (msg) msg.textContent = 'verify failed: ' + String(e).slice(0,150);
  }}
}}
async function refreshStatus() {{
  const msg = document.getElementById('setup-msg');
  const healthEl = document.getElementById('setup-health');
  const tierEl = document.getElementById('setup-tier');
  const rawEl = document.getElementById('setup-raw');
  const dockerEl = document.getElementById('setup-docker');
  if (msg) msg.innerHTML = '<b style="color:#5a6b7d">…</b>';
  try {{
    const ctx = document.getElementById('setup-ctx')?.value || '32768';
    const dual = document.getElementById('setup-dual')?.checked ? 'true' : 'false';
    const vhdxV = document.getElementById('setup-vhdx')?.value.trim() || '';
    const modelGbV = document.getElementById('setup-model-gb')?.value.trim() || '';
    const s = await api('/v1/setup/status?context=' + encodeURIComponent(ctx) + '&dual=' + dual + (vhdxV ? '&vhdx=' + encodeURIComponent(vhdxV) : '') + (modelGbV ? '&model_gb=' + encodeURIComponent(modelGbV) : ''));
    // Update health/tier like refreshSetup (single fetch, no double load)
    const lin = s.health.linux || {{}};
    const win = s.health.windows || {{}};
    const st=(ok,mm)=> ok?`<b style="color:#157a3e" title="${{mm}}">✔ ${{mm.split(' — ')[0]}}</b>`:`<b style="color:#b3372c" title="${{mm}}">✘ ${{mm}}</b>`;
    if (healthEl && s.health) {{
      healthEl.style.display='grid'; healthEl.style.gridTemplateColumns='160px 1fr'; healthEl.style.gap='8px 10px'; healthEl.style.padding='0'; healthEl.style.background='transparent'; healthEl.style.border='none'; healthEl.style.color='#FFDD00';
      healthEl.innerHTML = ''
        + '<div style="color:#FFDD00; background:rgba(255,255,255,.04); border:1px solid rgba(255,221,0,.18); border-radius:6px; padding:.45rem .6rem">Windows</div><div style="color:#FFDD00; background:rgba(255,255,255,.04); border:1px solid rgba(255,221,0,.18); border-radius:6px; padding:.45rem .6rem">' + st(win.state==='running', win.state==='running'?'running — Windows sidecar on :'+(win.port||8765):'Windows not running — start Splinter Studio') + '</div>'
        + '<div style="color:#FFDD00; background:rgba(255,255,255,.04); border:1px solid rgba(255,221,0,.18); border-radius:6px; padding:.45rem .6rem">Linux</div><div style="color:#FFDD00; background:rgba(255,255,255,.04); border:1px solid rgba(255,221,0,.18); border-radius:6px; padding:.45rem .6rem">' + st(lin.state==='running', lin.state==='running'?'Linux running — Docker on :'+(lin.port||8000):'Linux stopped — Bootstrap Docker') + '</div>'
        + '<div style="color:#FFDD00; background:rgba(255,255,255,.04); border:1px solid rgba(255,221,0,.18); border-radius:6px; padding:.45rem .6rem">VHDX</div><div style="color:#FFDD00; background:rgba(255,255,255,.04); border:1px solid rgba(255,221,0,.18); border-radius:6px; padding:.45rem .6rem">' + st(lin.vhdxExists, lin.vhdxExists?(lin.vhdxMounted?'VHDX mounted — ready':'VHDX exists but not mounted — click Mount AI Drive and approve UAC'):'VHDX missing — click Create Drive and pick size') + '</div>'
        + '<div style="color:#FFDD00; background:rgba(255,255,255,.04); border:1px solid rgba(255,221,0,.18); border-radius:6px; padding:.45rem .6rem">Shards</div><div style="color:#FFDD00; background:rgba(255,255,255,.04); border:1px solid rgba(255,221,0,.18); border-radius:6px; padding:.45rem .6rem">' + st(lin.shardsFound, lin.shardsFound?'Shards found — '+(lin.shardPath||'').split('/').pop():'Shards missing — add .gguf to /mnt/dsh_storage/models via WSL or move from System') + '</div>'
        + '<div style="color:#FFDD00; background:rgba(255,255,255,.04); border:1px solid rgba(255,221,0,.18); border-radius:6px; padding:.45rem .6rem">Docker 8000</div><div style="color:#FFDD00; background:rgba(255,255,255,.04); border:1px solid rgba(255,221,0,.18); border-radius:6px; padding:.45rem .6rem">' + st(lin.dockerRunning, lin.dockerRunning?'Docker healthy — :8000 200':'Docker down — Bootstrap Docker or docker compose up dsh-compute-backend') + '</div>'
        + '<div style="color:#FFDD00; background:rgba(255,255,255,.04); border:1px solid rgba(255,221,0,.18); border-radius:6px; padding:.45rem .6rem">Complete</div><div style="color:#FFDD00; background:rgba(255,255,255,.04); border:1px solid rgba(255,221,0,.18); border-radius:6px; padding:.45rem .6rem">' + st(s.complete, s.complete?'Complete — WebUI :3000 → :8000 ready':'Not complete — fix VHDX/mount/shards/docker above') + '</div>';
    }}
    if (tierEl && s.tier) {{
      const m = s.tier.metrics || {{}};
      const f = s.tier.flags || {{}};
      tierEl.style.display='grid'; tierEl.style.gridTemplateColumns='160px 1fr'; tierEl.style.gap='8px 10px'; tierEl.style.padding='12px'; tierEl.style.background='#000'; tierEl.style.border='1.5px solid #FFB703'; tierEl.style.borderRadius='8px'; tierEl.style.color='#FFDD00';
      tierEl.innerHTML = ''
        + '<div style="color:#FFDD00; background:rgba(255,255,255,.04); border:1px solid rgba(255,221,0,.18); border-radius:6px; padding:.45rem .6rem" title="Combined VRAM — cross-platform">VRAM</div><div style="color:#FFDD00; background:rgba(255,255,255,.04); border:1px solid rgba(255,221,0,.18); border-radius:6px; padding:.45rem .6rem">' + (m.tier1VramGb ?? '-') + ' GB</div>'
        + '<div style="color:#FFDD00; background:rgba(255,255,255,.04); border:1px solid rgba(255,221,0,.18); border-radius:6px; padding:.45rem .6rem" title="Total RAM — cross-platform">RAM</div><div style="color:#FFDD00; background:rgba(255,255,255,.04); border:1px solid rgba(255,221,0,.18); border-radius:6px; padding:.45rem .6rem">' + (m.tier2RamGb ?? '-') + ' GB</div>'
        + '<div style="color:#FFDD00; background:rgba(255,255,255,.04); border:1px solid rgba(255,221,0,.18); border-radius:6px; padding:.45rem .6rem; display:flex; flex-direction:column; gap:.35rem" title="T3 spill estimator: model+KV - VRAM - RAM. Change model GB here."><span>T3 spill estimator</span><label class="inline" style="margin:0; display:flex; gap:.2rem; align-items:center; background:rgba(0,0,0,.25); border:1px solid rgba(255,221,0,.18); border-radius:4px; padding:.15rem .3rem; width:fit-content">model <input id="setup-model-gb" type="number" value="' + (m.modelGb ?? 104) + '" min="1" max="2000" step="1" style="width:45px; text-align:right; background:#000; color:#FFDD00; border:1px solid rgba(255,221,0,.4); border-radius:3px; padding:.1rem .2rem; -moz-appearance:textfield; appearance:textfield" onchange="refreshStatus()"><span style="color:#FFDD00; opacity:.9; font-size:.85em">GB</span></label></div><div style="color:#FFDD00; background:rgba(255,255,255,.04); border:1px solid rgba(255,221,0,.18); border-radius:6px; padding:.45rem .6rem">' + (m.tier3NvmeGb ?? '-') + ' GB</div>'
        + '<div style="color:#FFDD00; background:rgba(255,255,255,.04); border:1px solid rgba(255,221,0,.18); border-radius:6px; padding:.45rem .6rem" title="Max context from leftover without clamp">Max context</div><div style="color:#FFDD00; background:rgba(255,255,255,.04); border:1px solid rgba(255,221,0,.18); border-radius:6px; padding:.45rem .6rem">' + fmtCap(f.recommendCap) + '</div>'
        + '<div style="color:#FFDD00; background:rgba(255,255,255,.04); border:1px solid rgba(255,221,0,.18); border-radius:6px; padding:.45rem .6rem" title="Effective bandwidth weighted by tiers">Speed</div><div style="color:#FFDD00; background:rgba(255,255,255,.04); border:1px solid rgba(255,221,0,.18); border-radius:6px; padding:.45rem .6rem">' + (m.estEffectiveBw ?? '-') + ' GB/s</div>'
        + '<div style="color:#FFDD00; background:rgba(255,255,255,.04); border:1px solid rgba(255,221,0,.18); border-radius:6px; padding:.45rem .6rem" title="Space left after spill">Free space</div><div style="color:#FFDD00; background:rgba(255,255,255,.04); border:1px solid rgba(255,221,0,.18); border-radius:6px; padding:.45rem .6rem">' + (m.freeAfterSpillGb ?? '-') + ' GB</div>';
    }}
    if (dockerEl) {{
      const lin2 = s.health.linux || {{}};
      dockerEl.textContent = 'VHDX ' + (lin2.vhdxExists ? 'exists' : 'missing') + ' (' + (s.state.vhdxPath || '-') + ')\\n'
        + 'Mount ' + (lin2.vhdxMounted ? 'mounted' : 'not mounted') + ' → ' + (s.state.mountPoint || '/mnt/dsh_storage') + '\\n'
        + 'Shards ' + (lin2.shardsFound ? 'found ' + (lin2.shardPath || '').split('/').pop() : 'missing') + '\\n'
        + 'Docker http://127.0.0.1:8000/health → ' + (lin2.dockerRunning ? '200 healthy' : 'down — docker compose up dsh-compute-backend') + '\\n'
        + 'WebUI :3000 → dsh-compute-backend:8000/v1 ' + (lin2.dockerRunning ? 'route ready' : 'route down');
      console.log('[Splinter] Docker details', dockerEl.textContent, s.health, s.state);
    }} else {{
      console.log('[Splinter] Docker details', s.health.linux, s.state);
    }}
    const ok = lin.vhdxExists && lin.vhdxMounted && lin.shardsFound && lin.dockerRunning && !s.tier.flags.diskFull;
    const orderTip = !ok ? (!lin.vhdxExists ? 'Fix: 1. Create Drive' : !lin.vhdxMounted ? 'Fix: 1. Mount AI Drive → 2. Bootstrap Docker' : !lin.shardsFound ? 'Fix: add .gguf (no button)' : !lin.dockerRunning ? 'Fix: 1. Bootstrap Docker' : 'Fix: check diskFull') : 'Ready — WebUI :3000 → :8000 ready';
    const orderEsc = orderTip.replace(/"/g, '&quot;');
    const tick = ok ? '<b style="color:#157a3e" title="'+orderEsc+'">✔</b>' : '<b style="color:#b3372c" title="'+orderEsc+'">✘</b>';
    const baseDetail = ok ? 'verify:live LINKED — ready to launch' : 'fix: ' + (!lin.vhdxExists ? 'VHDX missing' : !lin.vhdxMounted ? 'not mounted → wsl --mount --vhd E:/dsh_storage.vhdx --bare' : !lin.shardsFound ? 'shards missing' : !lin.dockerRunning ? 'docker down' : 'disk >80%');
    if (rawEl) {{ rawEl.textContent = JSON.stringify(s, null, 1); rawEl.style.display = 'none'; console.log('[Splinter] setup status', s); }}
    // merged Test WebUI ↔ Docker — wait for Docker 8000 + WebUI 3000, then merge into one sentence (Docker loads last) — console only (details hidden)
    let dockerExtra = '';
    let webuiExtra = '';
    let dockerModels = null;
    let dockerErr = null;
    let webuiOk = false;
    try {{
      dockerModels = await api('/v1/setup/docker-models').catch(e => {{ throw new Error('Docker 8000 /v1/models failed: ' + (e.message || e)); }});
      const count = dockerModels && dockerModels.data ? dockerModels.data.length : 0;
      dockerExtra = ' · Docker ' + (count ? count + ' model' + (count===1?'':'s') : 'reachable') + ' (' + (lin.dockerRunning ? '8000 ok' : '8000 ok but health says down') + ')';
      if (dockerEl) dockerEl.textContent += '\\n\\nTest WebUI ↔ Docker:\\nDocker /v1/models:\\n' + JSON.stringify(dockerModels, null, 1).slice(0,1200);
      console.log('[Splinter] Docker /v1/models (via harness proxy, no CORS)', dockerModels);
    }} catch(e) {{
      dockerErr = String(e.message || e).slice(0,600);
      dockerExtra = ' · Docker unreachable (http://127.0.0.1:8000/health — ' + (lin.dockerRunning ? 'health says up but fetch failed' : 'docker down') + ')';
      if (dockerEl) dockerEl.textContent += '\\n\\nTest WebUI ↔ Docker failed:\\n' + dockerErr + '\\nDocker health: http://127.0.0.1:8000/health';
      console.warn('[Splinter] Docker fetch failed (via harness proxy)', dockerErr);
    }}
    try {{
      webuiOk = await fetch('http://127.0.0.1:3000', {{method: 'GET', mode: 'no-cors'}}).then(() => true).catch(() => false);
      webuiExtra = ' · WebUI :3000 ' + (webuiOk ? 'reachable → dsh-compute-backend:8000/v1' : 'not reachable — open http://127.0.0.1:3000');
      if (dockerEl) dockerEl.textContent += '\\n\\nWebUI :3000 → dsh-compute-backend:8000/v1 — ' + (webuiOk ? 'reachable (no-cors) route likely ok' : 'open http://127.0.0.1:3000 and check Settings → Connections');
      console.log('[Splinter] WebUI :3000', webuiOk ? 'reachable' : 'not reachable');
      if (dockerEl) console.log('[Splinter] Docker details (full)', dockerEl.textContent);
    }} catch(e) {{
      webuiExtra = ' · WebUI check failed';
      console.warn('[Splinter] WebUI check failed', e);
    }}
    // single merged sentence — wait until Docker (slowest) completes
    const finalDetail = baseDetail + dockerExtra + webuiExtra;
    const finalTip = (orderTip + dockerExtra + webuiExtra).replace(/"/g, '&quot;');
    if (msg) msg.innerHTML = '<span style="display:inline-flex; gap:.6rem; align-items:center; flex-wrap:wrap" title="'+finalTip+'">' + tick + ' ' + (ok ? 'Ready' : 'Not ready') + ' <span style="opacity:.6">·</span> ' + finalDetail + '</span>';
    console.log('[Splinter] Ready', finalDetail);
  }} catch(e) {{
    if (msg) msg.textContent = 'load failed: ' + String(e).slice(0,120);
  }}
}}
async function createDrive() {{
  const vhdxEl = document.getElementById('setup-vhdx');
  const vhdx = vhdxEl ? vhdxEl.value.trim() : '';
  const sizeEl = document.getElementById('setup-vhdx-size');
  const size_gb = sizeEl ? parseInt(sizeEl.value, 10) : 250;
  const msg = document.getElementById('setup-msg');
  const dockerEl = document.getElementById('setup-docker');
  if (!vhdx) {{ if (msg) msg.textContent = 'select a drive first'; return; }}
  if (!confirm('Create ' + size_gb + 'GB dynamic VHDX at ' + vhdx + '?\\n\\nThis creates a sparse virtual drive (initially small, max ' + size_gb + 'GB, grows as needed) at the chosen location and formats it to ext4 on first mount. Proceed?')) return;
  if (msg) msg.textContent = 'creating VHDX at ' + vhdx + ' (' + size_gb + 'GB dynamic, sparse)…';
  if (dockerEl) dockerEl.textContent = 'calling POST /v1/setup/create-vhdx for ' + vhdx + ' (' + size_gb + 'GB)…';
  console.log('[Splinter] Create Drive', vhdx, size_gb);
  try {{
    const r = await api('/v1/setup/create-vhdx', 'POST', {{vhdx, size_gb}});
    if (msg) msg.textContent = r.already ? 'VHDX already exists at ' + vhdx : 'VHDX created at ' + vhdx + ' via ' + (r.method || 'New-VHD');
    if (dockerEl) dockerEl.textContent = 'Create result:\\n' + JSON.stringify(r, null, 1).slice(0,1200);
    console.log('[Splinter] Create result', r);
    await refreshSetup();
  }} catch(e) {{
    const txt = String(e.message || e);
    if (msg) msg.textContent = 'create failed: ' + txt.slice(0,150);
    if (dockerEl) dockerEl.textContent = 'Create failed:\\n' + txt.slice(0,800);
    console.warn('[Splinter] Create failed', txt);
  }}
}}
async function mountBare() {{
  const msg = document.getElementById('setup-msg');
  const dockerEl = document.getElementById('setup-docker');
  const vhdxEl = document.getElementById('setup-vhdx');
  const vhdx = vhdxEl ? vhdxEl.value.trim() : '';
  if (msg) msg.textContent = 'mounting VHDX as bare device (Admin UAC may pop)…';
  if (dockerEl) dockerEl.textContent = 'calling POST /v1/setup/mount-bare' + (vhdx ? ' for ' + vhdx : '') + '…';
  console.log('[Splinter] Mount bare', vhdx || '(default)');
  try {{
    const r = await api('/v1/setup/mount-bare', 'POST', vhdx ? {{vhdx}} : {{}});
    if (r.needs_elevation) {{
      if (msg) msg.textContent = 'UAC shown on host — click Yes, then Bootstrap Docker';
      if (dockerEl) dockerEl.textContent = 'UAC prompt shown on host desktop — click Yes in the Windows dialog, then click Bootstrap Docker.\\n' + (r.error || '').slice(0,500);
      console.log('[Splinter] Mount needs UAC', r);
    }} else {{
      if (msg) msg.textContent = 'bare mount OK';
      if (dockerEl) dockerEl.textContent = 'Bare mount OK:\\n' + (r.output || '').slice(0,800);
      console.log('[Splinter] Bare mount OK', r);
      await refreshSetup();
    }}
  }} catch(e) {{
    const txt = String(e.message || e);
    if (msg) msg.textContent = 'mount failed: ' + txt.slice(0,150);
    if (dockerEl) dockerEl.textContent = 'Mount failed:\\n' + txt.slice(0,800) + '\\nFix: Right-click Mount_AI_Drive.bat → Run as administrator';
    console.warn('[Splinter] Mount failed', txt);
  }}
}}
async function bootstrapDocker() {{
  const dockerEl = document.getElementById('setup-docker');
  const msg = document.getElementById('setup-msg');
  const rawEl = document.getElementById('setup-raw');
  const vhdxEl = document.getElementById('setup-vhdx');
  const vhdx = vhdxEl ? vhdxEl.value.trim() : '';
  if (msg) msg.textContent = 'bootstrapping — mounting + docker compose up…';
  if (dockerEl) dockerEl.textContent = 'calling POST /v1/setup/bootstrap' + (vhdx ? ' for ' + vhdx : '') + '…\\n';
  console.log('[Splinter] Bootstrap Docker', vhdx || '(default)');
  try {{
    const r = await api('/v1/setup/bootstrap', 'POST', vhdx ? {{vhdx}} : {{}});
    if (dockerEl) dockerEl.textContent = 'Bootstrap steps:\\n' + JSON.stringify(r.steps, null, 1).slice(0,1800) + '\\n\\nhealth: ' + JSON.stringify(r.health, null, 1).slice(0,800);
    if (msg) msg.textContent = r.ok ? 'bootstrap LINKED — docker healthy' : 'bootstrap done but docker not healthy — check docker logs dsh-compute-backend';
    if (rawEl) {{ rawEl.textContent = JSON.stringify(r, null, 1); rawEl.style.display = 'block'; }}
    console.log('[Splinter] Bootstrap result', r);
    await refreshSetup();
  }} catch(e) {{
    const txt = String(e.message || e);
    if (txt.includes('bare') || txt.includes('Administrator') || txt.includes('Mount_AI_Drive')) {{
      if (dockerEl) dockerEl.textContent = 'Needs Admin bare mount first:\\n'
        + '1) Right-click Mount_AI_Drive.bat → Run as administrator\\n'
        + '   (or: wsl --mount --vhd E:/dsh_storage.vhdx --bare)\\n'
        + '2) Then click Bootstrap Docker again (no Admin needed)\\n\\n'
        + 'Error: ' + txt.slice(0,700);
      if (msg) msg.textContent = 'needs Admin: run Mount_AI_Drive.bat as Admin, then Bootstrap again';
      console.warn('[Splinter] Bootstrap needs Admin', txt);
    }} else {{
      if (msg) msg.textContent = 'bootstrap failed: ' + txt.slice(0,150);
      if (dockerEl) dockerEl.textContent = 'Bootstrap failed:\\n' + txt.slice(0,1200);
      console.warn('[Splinter] Bootstrap failed', txt);
    }}
    if (rawEl) rawEl.style.display = 'none';
  }}
}}
async function testWebUI() {{
  // kept for compat — merged into refreshStatus (Refresh now does health + WebUI check)
  return refreshStatus();
}}
async function loadDrives() {{
  const sel = document.getElementById('setup-drive');
  const vhdxEl = document.getElementById('setup-vhdx');
  if (!sel) return;
  try {{
    const data = await api('/v1/setup/drives');
    sel.innerHTML = '';
    for (const d of data.drives) {{
      const o = document.createElement('option');
      o.value = d.mount;
      o.textContent = d.mount + ' — ' + d.free_gb + 'GB free / ' + d.total_gb + 'GB';
      sel.appendChild(o);
    }}
    // No auto-best: leave VHDX as-is (default E:/dsh_storage.vhdx) so user explicitly picks drive for LLM storage
    sel.addEventListener('change', () => {{
      const drive = sel.value;
      let base = drive;
      if (!base.endsWith('/') && !base.endsWith('\\\\')) base += '/';
      base = base.replace(/\\\\/g, '/');
      if (vhdxEl) vhdxEl.value = base + 'dsh_storage.vhdx';
    }});
  }} catch(e) {{
    sel.innerHTML = '<option>auto-detect failed</option>';
  }}
}}
setTimeout(() => {{ try {{ loadSetup(); loadDrives(); }} catch(e) {{}} }}, 900);

/* ---------------- typeahead (hub repos + local library) --------------- */
let suggestTimer = null;
function hideSuggestions() {{
  for (const id of ['sug-q', 'sug-drepo', 'sug-hfrepo'])
    document.getElementById(id).innerHTML = '';
}}

async function suggestRepos(value, targetId) {{
  if (!value || value.length < 2) {{ hideSuggestions(); return; }}
  let results = [];
  try {{
    const res = await api('/v1/models/hub?q=' + encodeURIComponent(value) + '&limit=8');
    results = res.results || [];
  }} catch (e) {{ return; }}
  const dl = document.getElementById('repo-suggestions');
  dl.innerHTML = '';
  for (const r of results) {{
    const opt = document.createElement('option');
    opt.value = r.repo;
    dl.appendChild(opt);
  }}
  const box = document.getElementById('sug-' + targetId);
  if (!box) return;
  box.innerHTML = '';
  for (const r of results.slice(0, 8)) {{
    const item = document.createElement('div');
    item.className = 'sugitem';
    const name = document.createElement('span');
    name.textContent = r.repo;
    const meta = document.createElement('span');
    meta.className = 'meta';
    meta.textContent = '↓' + r.downloads;
    item.appendChild(name);
    item.appendChild(meta);
    item.addEventListener('mousedown', ev => {{
      ev.preventDefault();
      document.getElementById(targetId).value = r.repo;
      hideSuggestions();
      if (targetId === 'q') searchHub();
      else hubFiles(r.repo).catch(() => {{}});
    }});
    box.appendChild(item);
  }}
}}

async function hubFiles(repo) {{
  const res = await api('/v1/models/hub/files/' + encodeURIComponent(repo));
  document.getElementById('drepo').value = repo;
  document.getElementById('dfile').value = res.files.length ? res.files[0].file : '';
  const lines = res.files.map(f => `${{f.file}} — ${{f.size_gb}} GB`);
  show('hub', `files in ${{repo}}:\\n${{lines.join('\\n') || '(none)'}}`);
}}

async function suggestLocal() {{
  try {{
    let l;
    if(_libCache.system && (Date.now()-_libCache.ts < 30000)) l={{models: _libCache.system}};
    else l=await api('/v1/models/local');
    const dl = document.getElementById('local-suggestions');
    dl.innerHTML = '';
    for (const m of l.models) {{
      const opt = document.createElement('option');
      opt.value = m.file;
      opt.label = `${{m.size_gb}} GB`;
      dl.appendChild(opt);
    }}
  }} catch (e) {{ /* best-effort */ }}
}}

for (const id of ['q', 'drepo', 'hfrepo']) {{
  const el = document.getElementById(id);
  if (!el) continue;
  el.addEventListener('input',
    e => {{ clearTimeout(suggestTimer);
           suggestTimer = setTimeout(() => suggestRepos(e.target.value.trim(), id), 350); }});
  el.addEventListener('blur', () => setTimeout(hideSuggestions, 150));
}}
document.getElementById('model').addEventListener('focus', suggestLocal);

/* --------------------- sessions (OpenCode-style tabs) ----------------- */
const SESS_KEY = 'splinter-console-sessions';
let sessions = {{}};
try {{ sessions = JSON.parse(localStorage.getItem(SESS_KEY) || '{{}}') || {{}}; }}
catch (e) {{ sessions = {{}}; }}
function persistSessions() {{
  localStorage.setItem(SESS_KEY, JSON.stringify(sessions));
}}
function ensureSession(id) {{
  if (!sessions[id]) sessions[id] = {{ title: null, updated: Date.now(), transcript: [] }};
  return sessions[id];
}}
function record(role, text, meta) {{
  const s = ensureSession(convId);
  s.transcript.push({{role: role, text: text, meta: meta || null}});
  if (s.transcript.length > 400) s.transcript.splice(0, s.transcript.length - 400);
  s.updated = Date.now();
  persistSessions();
}}
function restoreTranscript(id) {{
  const log = document.getElementById('chatlog');
  log.innerHTML = '';
  const s = sessions[id];
  if (!s) return;
  for (const m of s.transcript) {{
    if (m.role === 'tool') continue;   // tool steps are agent-run noise on replay
    bubble(m.role, m.text, m.meta);
  }}
}}
function tabLabel(id) {{
  const s = sessions[id];
  return (s && s.title) ? s.title : 'Session ' + id.slice(-5);
}}
function renderTabs() {{
  const wrap = document.getElementById('sess-tabs');
  wrap.innerHTML = '';
  const ids = Object.keys(sessions).sort((a, b) =>
    (sessions[b].updated || 0) - (sessions[a].updated || 0));
  for (const id of ids) {{
    const t = document.createElement('span');
    t.className = 'sesstab' + (id === convId ? ' active' : '')
      + (sessions[id].title ? '' : ' unsaved');
    const name = document.createElement('span');
    name.className = 'name';
    name.textContent = tabLabel(id);
    t.appendChild(name);
    const x = document.createElement('span');
    x.className = 'x';
    x.textContent = '\\u00d7';
    x.title = 'Close session';
    x.addEventListener('click', ev => {{ ev.stopPropagation(); closeSession(id); }});
    t.appendChild(x);
    t.addEventListener('click', () => switchSession(id));
    wrap.appendChild(t);
  }}
}}
async function switchSession(id) {{
  if (id === convId) return;
  convId = id;
  localStorage.setItem('splinter-console-conv', convId);
  restoreTranscript(id);
  renderTabs();
  applyConvProvider();
  try {{ await api('/v1/splinter/state?conversation_id=' + encodeURIComponent(id)); }} catch (e) {{}}
}}
function closeSession(id) {{
  delete sessions[id];
  persistSessions();
  api('/v1/splinter/reset', 'POST', {{conversation_id: id}}).catch(() => {{}});
  if (id !== convId) {{ renderTabs(); return; }}
  const rest = Object.keys(sessions).sort((a, b) =>
    (sessions[b].updated || 0) - (sessions[a].updated || 0));
  if (rest.length) switchSession(rest[0]);
  else newConversation();
}}
function saveSession() {{
  const cur = ensureSession(convId);
  const name = prompt('Name this session:', cur.title || '');
  if (name === null) return;
  cur.title = name.trim() || ('Session ' + new Date().toLocaleString());
  cur.updated = Date.now();
  persistSessions();
  renderTabs();
}}
/* per-conversation inference target (the header model dropdown) */
function convProviderStore() {{
  try {{ return JSON.parse(localStorage.getItem('splinter-console-convprov') || '{{}}'); }}
  catch (e) {{ return {{}}; }}
}}
function saveConvProvider(value) {{
  const per = convProviderStore();
  if (value) per[convId] = value;
  else delete per[convId];
  localStorage.setItem('splinter-console-convprov', JSON.stringify(per));
}}
function applyConvProvider() {{
  const v = convProviderStore()[convId];
  if (!v) return;
  const sel = document.getElementById('chat-provider');
  if ([...sel.options].some(o => o.value === v)) sel.value = v;
}}

/* --------------------------- afk toggle ------------------------------ */
async function toggleAfk(btn) {{
  const on = btn.dataset.afk === '1';
  const r = await api('/v1/splinter/mode', 'POST', {{afk: !on, note: on ? 'operator returned' : 'operator away'}});
  applyAfk(r.afk);
}}
function applyAfk(on) {{
  const b = document.getElementById('afkbtn');
  if (!b) return;
  b.dataset.afk = on ? '1' : '0';
  b.textContent = on ? 'AFK ON - click to end' : 'AFK';
  b.classList.toggle('afk-on', !!on);
}}
(async function initAfk() {{
  try {{ const m = await api('/v1/splinter/mode'); applyAfk(!!m.afk); }} catch (e) {{}}
}})();
/* --------------------------- research queue -------------------------- */
async function researchAdd(btn) {{
  const inp = document.getElementById('researchq');
  const q = inp.value.trim();
  if (!q) return;
  btn.disabled = true;
  try {{
    await api('/v1/research/queue', 'POST', {{question: q}});
    inp.value = '';
    loadResearchCount();
  }} finally {{ btn.disabled = false; }}
}}
async function loadResearchCount() {{
  try {{
    const r = await api('/v1/research/queue');
    const b = document.getElementById('researchcount');
    if (b) b.textContent = r.items.length ? '(' + r.items.length + ' queued)' : '';
  }} catch (e) {{}}
}}
loadResearchCount();
/* ------------------------------ chat -------------------------------- */
function bubble(role, text, meta) {{
  const log = document.getElementById('chatlog');
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  div.textContent = text;
  if (meta) {{
    const m = document.createElement('span');
    m.className = 'meta';
    m.textContent = meta;
    div.appendChild(m);
  }}
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
  // Cap DOM growth: a long-lived tab must not accumulate every bubble.
  while (log.children.length > 200) log.removeChild(log.firstChild);
  return div;
}}

function chatMode() {{
  return document.querySelector('input[name="chatmode"]:checked').value;
}}

/* ------------------------- slash commands ---------------------------- */
let chatCommands = [];

async function loadChatCommands() {{
  try {{
    chatCommands = (await api('/v1/commands')).commands;
  }} catch (e) {{ chatCommands = []; }}
}}

function chatSubmit() {{
  const input = document.getElementById('chatin');
  const text = input.value;
  if (text.trim().startsWith('/')) {{
    input.value = '';
    hideSuggestions();
    runCommand(text.trim());
    return;
  }}
  sendChat();
}}

async function runCommand(line) {{
  bubble('user', line);
  record('user', line);
  try {{
    const r = await api('/v1/commands/run', 'POST',
                        {{line: line, conversation_id: convId}});
    bubble('sys', (r.kind === 'error' ? '⚠ ' : '') + (r.text || r.kind));
    record('sys', (r.kind === 'error' ? '⚠ ' : '') + (r.text || r.kind));
    if (r.new_conversation_id) {{
      convId = r.new_conversation_id;
      localStorage.setItem('splinter-console-conv', convId);
      ensureSession(convId);
      persistSessions();
      renderTabs();
      document.getElementById('chatlog').innerHTML = '';
    }}
    if (r.mode) {{
      const radio = document.querySelector(
        `input[name="chatmode"][value="${{r.mode}}"]`);
      if (radio) radio.checked = true;
    }}
  }} catch (e) {{
    bubble('sys', 'command failed: ' + e.message);
  }}
}}

document.getElementById('chatin').addEventListener('input', e => {{
  const v = e.target.value;
  const box = document.getElementById('sug-chat');
  if (!v.startsWith('/')) {{ box.innerHTML = ''; return; }}
  const query = v.slice(1).toLowerCase();
  const matches = chatCommands.filter(c => c.name.startsWith(query));
  box.innerHTML = '';
  for (const cmd of matches) {{
    const item = document.createElement('div');
    item.className = 'sugitem';
    const name = document.createElement('span');
    name.textContent = '/' + cmd.name;
    const meta = document.createElement('span');
    meta.className = 'meta';
    meta.textContent = cmd.description + (cmd.input ? ' ' + cmd.input.hint : '');
    item.appendChild(name);
    item.appendChild(meta);
    item.addEventListener('mousedown', ev => {{
      ev.preventDefault();
      e.target.value = '/' + cmd.name + ' ';
      box.innerHTML = '';
      e.target.focus();
    }});
    box.appendChild(item);
  }}
}});

loadChatCommands();

async function sendChat() {{
  if (streamBusy) return;              /* one stream at a time */
  const input = document.getElementById('chatin');
  const query = input.value.trim();
  if (!query) return;
  input.value = '';
  if (chatMode() === 'agent') return sendAgent(query);
  bubble('user', query);
  record('user', query);
  const ai = bubble('ai', '');
  ai.classList.add('typing');
  setBusy(true);
  const ctrl = new AbortController();
  streamAbort = ctrl;
  const body = {{query: query, conversation_id: convId}};
  const provSel = document.getElementById('chat-provider').value;
  if (provSel) body.provider = provSel;
  if (Object.keys(hiveOverrides).length) body.config = hiveOverrides;
  let text = '';
  let turn = '?';
  let metaText = '';
  try {{
    const r = await fetch('/v1/splinter/stream', {{method: 'POST',
      headers: {{'content-type': 'application/json'}},
      body: JSON.stringify(body), signal: ctrl.signal}});
    if (!r.ok || !r.body) throw new Error('HTTP ' + r.status);
    const reader = r.body.getReader();
    const dec = new TextDecoder();
    let buf = '';
    while (true) {{
      const {{done, value}} = await reader.read();
      if (done) break;
      buf += dec.decode(value, {{stream: true}});
      let idx;
      while ((idx = buf.indexOf('\\n\\n')) >= 0) {{
        const frame = buf.slice(0, idx); buf = buf.slice(idx + 2);
        if (!frame.startsWith('data: ')) continue;
        const ev = JSON.parse(frame.slice(6));
        if (ev.type === 'delta') {{
          if (!text) ai.classList.remove('typing');
          text += ev.text;
          ai.textContent = text;
          log_scroll();
        }} else if (ev.type === 'meta') {{
          turn = ev.turn;
          metaText = `curated ${{ev.token_count}}/${{ev.budget}} tokens`;
        }} else if (ev.type === 'done') {{
          metaText = `splinter-curated · turn ${{turn}}`
            + (ev.tokens ? ` · ${{ev.tokens}} tok` : '')
            + (ev.tokens_per_sec ? ` · ${{ev.tokens_per_sec}} tok/s` : '')
            + (ev.stored ? '' : ' · not stored');
        }} else if (ev.type === 'error') {{
          metaText = 'error: ' + friendlyError(ev.error);
        }}
      }}
    }}
    ai.classList.remove('typing');
    if (!text && metaText) {{
      /* error-only turn: show the message itself, not an empty bubble */
      ai.textContent = metaText;
    }} else {{
      ai.textContent = text || '(empty reply)';
      const m = document.createElement('span');
      m.className = 'meta';
      m.textContent = metaText;
      ai.appendChild(m);
    }}
    if (body.inspection !== undefined && body.inspection) {{
      renderInspection(body.inspection);
    }} else {{
      fetchInspection();
    }}
  }} catch (e) {{
    ai.classList.remove('typing');
    const cancelled = e && e.name === 'AbortError';
    const msg = cancelled ? 'cancelled.'
      : 'request failed: ' + friendlyError(e.message);
    ai.textContent = msg;
    record('ai', msg);
    if (!input.value) input.value = query;   /* give the draft back */
  }} finally {{
    streamAbort = null;
    setBusy(false);
  }}
}}

let streamBusy = false;
let streamAbort = null;
function setBusy(b) {{
  streamBusy = b;
  document.getElementById('stopbtn').style.display = b ? '' : 'none';
  document.getElementById('sendbtn').disabled = b;
}}

/* Map backend/sidecar failure text to something a user can act on. The
   SSE `error` frame carries raw Python exception strings today; this is
   the client-side translation layer until the server emits clean copy. */
function friendlyError(msg) {{
  const s = String(msg || '');
  const low = s.toLowerCase();
  if (low.includes(':1234') || low.includes('max retries') ||
      low.includes('failed to establish') || low.includes('connection refused'))
    return 'No backend reachable on :1234 — start a server from the Models '
         + 'tab, or pick another provider.';
  if (low.includes('failed to fetch') || low.includes('networkerror') ||
      low.includes('load failed'))
    return 'Sidecar unreachable — is the harness process still running?';
  if (s.length > 300) return s.slice(0, 300) + '…';
  return s;
}}

function cancelStream() {{
  if (chatMode() === 'agent') return cancelAgent();
  if (streamAbort) streamAbort.abort();
}}

async function cancelAgent() {{
  try {{ await api('/v1/agent/cancel', 'POST'); }} catch (e) {{}}
}}

async function sendAgent(query) {{
  bubble('user', query);
  record('user', query);
  bubble('sys', 'dsh agent starting…');
  const ai = bubble('ai', '');
  setBusy(true);
  let text = '';
  let finish = '';
  try {{
    const r = await fetch('/v1/agent/stream', {{method: 'POST',
      headers: {{'content-type': 'application/json'}},
      body: JSON.stringify({{message: query, conversation_id: convId}})}});
    if (!r.ok || !r.body) {{
      const t = await r.text();
      throw new Error('HTTP ' + r.status + ' ' + t.slice(0, 200));
    }}
    const reader = r.body.getReader();
    const dec = new TextDecoder();
    let buf = '';
    while (true) {{
      const {{done, value}} = await reader.read();
      if (done) break;
      buf += dec.decode(value, {{stream: true}});
      let idx;
      while ((idx = buf.indexOf('\\n\\n')) >= 0) {{
        const frame = buf.slice(0, idx); buf = buf.slice(idx + 2);
        if (!frame.startsWith('data: ')) continue;
        const ev = JSON.parse(frame.slice(6));
        if (ev.type === 'assistant') {{
          text = ev.text;                       // committed per agent step
          ai.textContent = text;
          log_scroll();
        }} else if (ev.type === 'tool') {{
          /* persistent collapsible step: stays in the transcript so a long
             agent run reads like dsh's own tool view instead of toasts */
          const det = document.createElement('details');
          det.className = 'toolstep';
          const sum = document.createElement('summary');
          sum.textContent = (ev.phase === 'end' ? '\\u2713 ' : '\\u25b6 ')
            + ev.tool;
          det.appendChild(sum);
          const pre = document.createElement('pre');
          pre.textContent = JSON.stringify(ev, null, 1).slice(0, 2000);
          det.appendChild(pre);
          document.getElementById('chatlog').appendChild(det);
          record('tool', ev.phase + ': ' + ev.tool);
          log_scroll();
        }} else if (ev.type === 'lifecycle') {{
          /* turn/step boundaries stay quiet */
        }} else if (ev.type === 'done') {{
          finish = ev.finish_reason || 'end';
          if (ev.final && ev.final !== text) {{
            text = ev.final;
            ai.textContent = text;
          }}
        }} else if (ev.type === 'error') {{
          ai.textContent = 'agent error: ' + ev.error;
          return;
        }}
      }}
    }}
    ai.textContent = text || '(no response)';
    const m = document.createElement('span');
    m.className = 'meta';
    m.textContent = 'dsh agent · ' + finish;
    ai.appendChild(m);
    record('ai', text || '(no response)', 'dsh agent · ' + finish);
  }} catch (e) {{
    ai.textContent = 'agent request failed: ' + e.message;
    record('ai', 'agent request failed: ' + e.message);
  }} finally {{
    setBusy(false);
  }}
}}
function log_scroll() {{
  const log = document.getElementById('chatlog');
  log.scrollTop = log.scrollHeight;
}}

function newConversation() {{
  /* OpenCode semantics: the current session keeps its tab; this opens a
     fresh conversation id and a fresh (empty) tab beside it. */
  convId = 'console-' + crypto.randomUUID().slice(0, 8);
  localStorage.setItem('splinter-console-conv', convId);
  ensureSession(convId);
  persistSessions();
  document.getElementById('chatlog').innerHTML = '';
  bubble('ai', 'Fresh session started.'
    + (Object.keys(hiveOverrides).length
       ? ' Splinter overrides apply from the next message.' : ''));
  renderTabs();
}}

/* boot: rebuild the tab strip and the last session's transcript */
ensureSession(convId);
persistSessions();
renderTabs();
restoreTranscript(convId);
/* the model dropdown populates async from /v1/provider/config */
setTimeout(applyConvProvider, 2000);

/* --------------------------- server panel ---------------------------- */
function launchBody() {{
  return {{
    model: val('model') || null,
    hf_repo: val('hfrepo') || null,
    hf_file: val('hffile') || null,
    ctx_size: +val('ctx') || 8192,
    ngl: +val('ngl') || 999,
    api_key: val('l-apikey') || null,
    threads: num('l-threads'),
    flash_attn: document.getElementById('l-fa').checked,
    parallel_slots: num('l-parallel'),
    cache_type_k: val('l-ctk') || null,
    cache_type_v: val('l-ctv') || null,
    batch_size: num('l-batch'),
    ubatch_size: num('l-ubatch'),
    alias: val('l-alias') || null,
    mlock: document.getElementById('l-mlock').checked,
    no_mmap: document.getElementById('l-nommap').checked,
  }};
}}

async function startServer(btn) {{
  if (btn) btn.disabled = true;          /* slow POST — no double submits */
  try {{
    const r = await fetch('/v1/server/start', {{method: 'POST',
      headers: {{'content-type': 'application/json'}},
      body: JSON.stringify(launchBody())}});
    show('status', await r.text());
    refresh();
  }} finally {{ if (btn) btn.disabled = false; }}
}}

async function searchHub() {{
  show('hub', 'searching…');
  try {{
    const res = await api('/v1/models/hub?q=' +
      encodeURIComponent(val('q')) + '&limit=12');
    show('hub', res.results.map(r =>
      `${{r.repo}} · ↓${{r.downloads}} · ★${{r.likes}} · ${{r.last_modified}}`).join('\\n'));
  }} catch (e) {{ show('hub', String(e)); }}
}}

async function download(btn) {{
  if (btn) btn.disabled = true;          /* slow POST — no double submits */
  try {{
    const r = await fetch('/v1/models/hub/download', {{method: 'POST',
      headers: {{'content-type': 'application/json'}},
      body: JSON.stringify({{repo: val('drepo'), file: val('dfile')}})}});
    show('downloads', await r.text());
  }} finally {{ if (btn) btn.disabled = false; }}
}}

async function setLibraryPath() {{
  const input = document.getElementById('library-path');
  const folder = input.value.trim();
  const status = document.getElementById('import-status');
  if (!folder) {{ status.textContent = 'enter a folder path'; setTimeout(()=>status.textContent='',2000); return; }}
  status.textContent = `switching to ${{folder}}…`;
  try {{
    const r = await fetch('/v1/models/local/path', {{method: 'POST', headers: {{'content-type':'application/json'}}, body: JSON.stringify({{folder}})}});
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail || j.error || await r.text());
    status.textContent = `now using ${{j.models_dir}} (${{j.models?.length || 0}} models)`;
    document.getElementById('library-path-note').textContent = j.models_dir;
    refresh();
  }} catch (e) {{
    status.textContent = 'failed: ' + e.message;
  }} finally {{
    setTimeout(()=>status.textContent='',4000);
  }}
}}

function filterLibrary() {{
  const q = (document.getElementById('library-filter').value || '').toLowerCase().trim();
  const wrap = document.getElementById('local');
  let visible = 0;
  for (const row of wrap.children) {{
    const name = (row.dataset.file || '').toLowerCase();
    const show = !q || name.includes(q);
    row.style.display = show ? '' : 'none';
    if (show) visible++;
  }}
  const count = document.getElementById('library-filter-count');
  if (count) count.textContent = q ? `${{visible}} / ${{wrap.children.length}}` : '';
  const engSel = document.getElementById('eng-model-select');
  const launchSel = document.getElementById('launch-model-select');
  const filterSelect = (sel) => {{
    if (!sel) return;
    for (const opt of sel.options) {{
      if (!opt.value) continue;
      opt.style.display = !q || opt.value.toLowerCase().includes(q) ? '' : 'none';
      opt.hidden = !q || opt.value.toLowerCase().includes(q) ? false : true;
    }}
  }};
  filterSelect(engSel);
  filterSelect(launchSel);
  // also filter both new panes
  for (const id of ['local-system','local-linux']) {{
    const w = document.getElementById(id);
    if (!w) continue;
    let vis=0;
    for (const row of w.children) {{
      const name=(row.dataset.file||'').toLowerCase();
      const show=!q||name.includes(q);
      row.style.display=show?'':'none';
      if(show) vis++;
    }}
  }}
}}
function showToast(msg) {{
  let t=document.getElementById('toast');
  if(!t){{ t=document.createElement('div'); t.id='toast'; t.className='toast'; document.body.appendChild(t); }}
  t.textContent=msg;
  t.style.display='block';
  t.classList.remove('fade');
  t.onclick=()=>{{ t.style.display='none'; }};
  clearTimeout(t._timer);
  t._timer=setTimeout(()=>{{ t.classList.add('fade'); setTimeout(()=>{{ t.style.display='none'; }},300); }},3000);
}}
let _ctxMenu=null;
function hideContextMenu(){{ if(_ctxMenu){{ _ctxMenu.remove(); _ctxMenu=null; }} }}
function showContextMenu(x,y,file,location) {{
  hideContextMenu();
  const menu=document.createElement('div');
  menu.className='context-menu';
  _ctxMenu=menu;
  const addItem=(label, handler, danger=false)=>{{
    const it=document.createElement('div');
    it.className='context-menu-item'+(danger?' danger':'');
    it.textContent=label;
    it.onclick=async (e)=>{{ e.stopPropagation(); hideContextMenu(); await handler(); }};
    menu.appendChild(it);
  }};
  const hfUrl='https://huggingface.co/models?search=' + encodeURIComponent(file.split('/').pop().split('\\\\').pop().replace(/\\.gguf$/i,''));
  addItem('View on Hugging Face', ()=>{{ window.open(hfUrl, '_blank'); }});
  if(location==='system'){{
    addItem('Move to Linux (/mnt/dsh_storage/models)', async ()=>{{ await moveToLinux(file); }});
    addItem('Delete', async ()=>{{ await deleteModel(file,'system'); }}, true);
  }} else {{
    addItem('Send back to Windows', async ()=>{{ await moveToWindows(file); }});
    addItem('Delete', async ()=>{{ await deleteModel(file,'linux'); }}, true);
  }}
  menu.style.left=x+'px';
  menu.style.top=y+'px';
  document.body.appendChild(menu);
  // reposition if off-screen
  const r=menu.getBoundingClientRect();
  if(r.right>window.innerWidth) menu.style.left=(window.innerWidth-r.width-8)+'px';
  if(r.bottom>window.innerHeight) menu.style.top=(window.innerHeight-r.height-8)+'px';
  setTimeout(()=>{{ document.addEventListener('click', hideContextMenu, {{once:true}}); }},10);
}}
async function moveToLinux(file) {{
  if(!confirm('Move ' + file + ' to Linux (/mnt/dsh_storage/models)?\\n\\nThis copies the GGUF via WSL to the Docker volume.')) return;
  try{{
    const r=await api('/v1/models/move-to-linux','POST',{{file}});
    showToast('Moved to Linux: ' + (r.file||file));
    _libCache.ts=0; _lastRender.system=null; _lastRender.linux=null;
    // Only refresh the two library panes, not the entire app
    await Promise.all([loadLibrarySystem(), loadLibraryLinux()]);
  }} catch(e){{ alert('Move failed: '+String(e)); }}
}}
async function moveToWindows(file) {{
  if(!confirm('Send ' + file + ' back to Windows library?')) return;
  try{{
    const r=await api('/v1/models/move-to-windows','POST',{{file}});
    showToast('Sent to Windows: ' + (r.file||file));
    _libCache.ts=0; _lastRender.system=null; _lastRender.linux=null;
    await Promise.all([loadLibrarySystem(), loadLibraryLinux()]);
  }} catch(e){{ alert('Send back failed: '+String(e)); }}
}}
async function deleteModel(file, location) {{
  if(!confirm('Delete ' + file + ' from ' + location + '?')) return;
  try{{
    if(location==='system') await api('/v1/models/local?file='+encodeURIComponent(file),'DELETE');
    else await api('/v1/models/linux?file='+encodeURIComponent(file),'DELETE');
    showToast('Deleted ' + file);
    // UI-only update: remove the row, no full reload
    const wrapId=location==='system'?'local-system':'local-linux';
    const wrap=document.getElementById(wrapId);
    if(wrap) {{
      for(const row of [...wrap.children]){{
        if(row.dataset.file===file) row.remove();
      }}
      if(!wrap.children.length){{ wrap.textContent='(no .gguf files yet)'; wrap.style.color='#000'; }}
    }}
    // Update caches
    const key=location;
    if(_libCache[key]) _libCache[key]=_libCache[key].filter(m=>m.file!==file);
    if(_lastRender[key]) _lastRender[key]=_lastRender[key].filter(m=>m.file!==file);
    if(window._selectedFile===file) window._selectedFile=null;
    // Keep hidden old #local in sync
    const old=document.getElementById('local');
    if(old && location==='system') old.innerHTML=document.getElementById('local-system').innerHTML;
  }} catch(e){{ alert(String(e)); }}
}}
let _lastRender={{system: null, linux: null}};
function renderLibrary(wrapId, models, location) {{
  const wrap=document.getElementById(wrapId);
  if(!wrap) return;
  // Avoid flicker: if models are the same as last render, skip
  const cacheKey=location;
  const last=_lastRender[cacheKey];
  const same=last && last.length===models.length && last.every((m,i)=>m.file===models[i].file && m.size_gb===models[i].size_gb);
  if(same) {{
    // Just restore selection without re-rendering
    if(window._selectedFile) {{
      for(const r of wrap.children) r.classList.toggle('selected', r.dataset.file===window._selectedFile);
    }}
    return;
  }}
  _lastRender[cacheKey]=models.map(m=>({{file:m.file, size_gb:m.size_gb}}));
  wrap.innerHTML='';
  if(!models.length){{ wrap.textContent='(no .gguf files yet)'; wrap.style.color='#000'; return; }}
  for(const m of models){{
    const row=document.createElement('div');
    row.className='librow';
    row.dataset.file=m.file;
    const label=document.createElement('span');
    const short=m.file.split('/').pop().split('\\\\').pop();
    label.textContent=`${{short}} — ${{m.size_gb}} GB`;
    // tooltip
    const arch=m.architecture||m.gguf_metadata?.['general.architecture']||m.ggufMetadata?.['general.architecture']||'—';
    const quant=m.quantization||m.gguf_metadata?.quantization||m.ggufMetadata?.quantization||'—';
    const ctx=m.context_length??m.contextLength??m.gguf_metadata?.context_length??m.ggufMetadata?.context_length??(()=>{{ const g=m.gguf_metadata||m.ggufMetadata||{{}}; for(const k in g) if(k.endsWith('.context_length')) return g[k]; return '—'; }})();
    const size=m.size_gb!=null?`${{m.size_gb}} GB`:(m.sizeGb!=null?`${{m.sizeGb}} GB`:'—');
    const mod=m.modified||m.lastModified||'—';
    row.title=`${{arch}} · ${{quant}} · ${{ctx}} · ${{size}} · ${{mod}}`;
    row.addEventListener('click', (e)=>{{
      // left click selects
      if(e.button!==0) return;
      const launchSelect=document.getElementById('launch-model-select');
      if(launchSelect){{ launchSelect.value=m.file; for(const o of launchSelect.options) if(o.value===m.file){{o.hidden=false; o.style.display='';}} }}
      window._selectedFile=m.file;
      // update selection visuals for both panes
      for(const id of ['local-system','local-linux']) {{
        const w=document.getElementById(id);
        if(!w) continue;
        for(const r of w.children) r.classList.toggle('selected', r.dataset.file===m.file);
      }}
    }});
    row.addEventListener('contextmenu', (e)=>{{
      e.preventDefault();
      showContextMenu(e.clientX, e.clientY, m.file, location);
    }});
    // restore selection if this is the selected file
    if(window._selectedFile && window._selectedFile===m.file) row.classList.add('selected');
    // long-press visual
    let pt=null;
    const sp=(on)=>row.classList.toggle('pressed',on);
    row.addEventListener('mousedown',()=>{{ pt=setTimeout(()=>sp(true),400); }});
    row.addEventListener('mouseup',()=>{{ clearTimeout(pt); sp(false); }});
    row.addEventListener('mouseleave',()=>{{ clearTimeout(pt); sp(false); }});
    wrap.appendChild(row);
    row.appendChild(label);
    // keep row as flex with label only; context menu handles actions
  }}
}}
async function loadLibrarySystem() {{
  const t0=performance.now();
  const wrapSys=document.getElementById('local-system');
  const isCached = _libCache.system && (Date.now() - _libCache.ts < 30000);
  if(wrapSys && !isCached) wrapSys.textContent='loading…';
  console.log('loadLibrarySystem START', new Date().toISOString(), isCached?'cached':'fetch');
  try{{
    const ctrl=new AbortController(); const to=setTimeout(()=>ctrl.abort(), 8000);
    const l=await api('/v1/models/local', 'GET', null, ctrl.signal).catch(e=>{{ clearTimeout(to); throw e; }});
    clearTimeout(to);
    console.log('loadLibrarySystem FETCH', (performance.now()-t0).toFixed(0)+'ms', l.models?.length+' models');
    const pathInput=document.getElementById('library-path');
    const pathNote=document.getElementById('library-path-note');
    if(l.models_dir) {{
      if(pathInput && !pathInput.value) pathInput.value=l.models_dir;
      if(pathNote) pathNote.textContent=`(${{l.models_dir}})`;
    }}
    // also populate engine selects — preserve selection
    const launchSelect=document.getElementById('launch-model-select');
    const engSel=document.getElementById('eng-model');
    const prevSel=window._selectedFile || (launchSelect?launchSelect.value:'');
    if(launchSelect) launchSelect.innerHTML='<option value="">— choose local model —</option>';
    if(engSel) engSel.innerHTML='<option value="">— choose local model —</option>';
    for(const m of l.models) {{
      const short=m.file.split('/').pop().split('\\\\').pop();
      if(launchSelect) {{
        const o=document.createElement('option'); o.value=m.file; o.textContent=short; launchSelect.appendChild(o);
        if(engSel){{ const oe=document.createElement('option'); oe.value=m.file; oe.textContent=short; engSel.appendChild(oe); }}
      }}
    }}
    // restore previous selection if still valid
    if(prevSel && l.models.some(m=>m.file===prevSel)) {{
      if(launchSelect) launchSelect.value=prevSel;
      window._selectedFile=prevSel;
    }} else if(window._selectedFile && !l.models.some(m=>m.file===window._selectedFile)) {{
      // selected file no longer exists (deleted) — clear
      window._selectedFile=null;
      if(launchSelect) launchSelect.value='';
    }}
    _libCache.system=l.models; _libCache.ts=Date.now();
    renderLibrary('local-system', l.models, 'system');
    filterLibrary();
    // keep hidden old #local in sync for filter
    const old=document.getElementById('local');
    if(old){{ old.innerHTML=document.getElementById('local-system').innerHTML; }}
  }} catch(e){{ console.error('system fetch failed',e); const w=document.getElementById('local-system'); if(w) w.textContent='load failed: '+String(e).slice(0,200); }}
}}
async function loadLibraryLinux() {{
  const t0=performance.now();
  const wrap=document.getElementById('local-linux');
  const isCachedLinux = _libCache.linux && (Date.now() - _libCache.ts < 30000);
  if(wrap && !isCachedLinux) wrap.textContent='loading…';
  console.log('loadLibraryLinux START', new Date().toISOString(), isCachedLinux?'cached':'fetch');
  try{{
    const ctrl=new AbortController(); const to=setTimeout(()=>ctrl.abort(), 8000);
    const l=await api('/v1/models/linux', 'GET', null, ctrl.signal).catch(e=>{{ clearTimeout(to); throw e; }});
    clearTimeout(to);
    console.log('loadLibraryLinux FETCH', (performance.now()-t0).toFixed(0)+'ms', (l.models||[]).length+' models', l.mounted?'mounted':'not mounted');
    _libCache.linux=l.models||[]; _libCache.ts=Date.now();
    renderLibrary('local-linux', l.models||[], 'linux');
    // Keep filter in sync
    filterLibrary();
    if(l.error) console.log('linux models', l.error);
  }} catch(e){{ console.error('linux fetch failed',e); if(wrap) wrap.textContent='load failed: '+String(e).slice(0,200); }}
}}
async function loadLibrary() {{
  await Promise.all([loadLibrarySystem(), loadLibraryLinux()]);
}}

let _currentAgentPreset = null;
async function loadAgentPresets() {{
  const wrap = document.getElementById('agent-presets-list');
  const detail = document.getElementById('agent-preset-detail');
  if (wrap) wrap.textContent = 'loading…';
  try {{
    // Try DSH apiproxy first, fallback to local file list
    let presets = [];
    try {{ const r = await api('/v1/agent-presets/list'); presets = r.presets || r; }} catch(e) {{
      // Fallback: list from /v1/provider/config + shipped presets
      const r = await api('/v1/provider/config'); presets = (r.providers||[]).map(p=>({{id:p.name, name:p.displayName||p.name, trust:'user'}}));
    }}
    if (!wrap) return;
    wrap.innerHTML = '';
    if (!presets.length) {{ wrap.textContent = '(no presets)'; return; }}
    for (const p of presets) {{
      const row = document.createElement('div');
      row.className = 'librow';
      const label = document.createElement('span');
      label.textContent = `${{p.name||p.id}} — ${{p.trust||'system'}}${{p.broken?' — broken':''}}`;
      const open = document.createElement('button');
      open.textContent = p.trust==='user' ? 'Edit' : 'View';
      open.onclick = async () => {{
        _currentAgentPreset = p.id;
        document.getElementById('agent-preset-name').textContent = p.name||p.id;
        const contentEl = document.getElementById('agent-preset-content');
        try {{
          const r = await api('/v1/agent-presets/read', 'POST', {{agentPreset:p.id}});
          contentEl.textContent = r.content || JSON.stringify(r,null,2);
        }} catch(e) {{ contentEl.textContent = 'cannot read: '+e.message; }}
        detail.style.display='';
      }};
      const dup = document.createElement('button');
      dup.textContent = 'Duplicate';
      dup.onclick = async () => {{
        const nid = prompt('New preset id (a-z0-9-):', p.id+'-copy');
        if (!nid) return;
        try {{ await api('/v1/agent-presets/copy', 'POST', {{from:p.id, agentPreset:nid}}); loadAgentPresets(); }} catch(e){{ alert(String(e)); }}
      }};
      row.appendChild(label);
      row.appendChild(open);
      row.appendChild(dup);
      if (p.trust==='user') {{
        const del = document.createElement('button'); del.textContent='Delete'; del.onclick=async()=>{{ if(!confirm('Delete '+p.id+'?'))return; try{{ await api('/v1/agent-presets/remove','POST',{{agentPreset:p.id}}); loadAgentPresets();}}catch(e){{alert(String(e));}} }}; row.appendChild(del);
      }}
      if (p.description) row.title = p.description;
      wrap.appendChild(row);
    }}
  }} catch(e) {{ if(wrap) wrap.textContent='load failed: '+e.message; }}
}}
async function createAgentPreset() {{
  const id = prompt('New preset id (a-z0-9-):', 'my-preset');
  if (!id) return;
  const from = prompt('Copy from preset id (blank for empty):', 'standard') || 'standard';
  try {{ await api('/v1/agent-presets/copy','POST',{{from, agentPreset:id}}); loadAgentPresets(); }} catch(e){{ alert(String(e)); }}
}}
async function saveAgentPreset() {{
  const contentEl = document.getElementById('agent-preset-content');
  const msg = document.getElementById('agent-preset-msg');
  if (!_currentAgentPreset || !contentEl) return;
  msg.textContent='saving…';
  try {{
    // For now, just show that editing is via files; direct save not yet wired in Studio
    msg.textContent='editing via files: save agent.cordis.yml in ~/.dsh/.agent-presets/'+_currentAgentPreset+'/';
    // Future: POST to /v1/agent-presets/write
  }} catch(e){{ msg.textContent='save failed: '+e.message; }}
  setTimeout(()=>{{msg.textContent='';}},3000);
}}
function closeAgentPreset() {{
  document.getElementById('agent-preset-detail').style.display='none';
  _currentAgentPreset=null;
}}

/* --------------------------- engines tab ----------------------------- */
let engines = [];
let engDefault = '';

function samplingToForm(s) {{
  const set = (id, v) => document.getElementById(id).value =
    (v === undefined || v === null) ? '' : v;
  set('s-temp', s.temperature); set('s-topp', s.top_p); set('s-topk', s.top_k);
  set('s-minp', s.min_p); set('s-rep', s.repeat_penalty);
  set('s-pres', s.presence_penalty); set('s-freq', s.frequency_penalty);
  set('s-seed', s.seed); set('s-miro', s.mirostat);
  set('s-tau', s.mirostat_tau); set('s-eta', s.mirostat_eta);
  set('s-stop', Array.isArray(s.stop) ? s.stop.join(',') : (s.stop || ''));
}}

function formToSampling() {{
  const out = {{}};
  const put = (key, id, parse) => {{
    const raw = val(id);
    if (raw !== '') out[key] = parse ? +raw : raw;
  }};
  put('temperature', 's-temp', true); put('top_p', 's-topp', true);
  put('top_k', 's-topk', true); put('min_p', 's-minp', true);
  put('repeat_penalty', 's-rep', true); put('presence_penalty', 's-pres', true);
  put('frequency_penalty', 's-freq', true); put('seed', 's-seed', true);
  put('mirostat', 's-miro', true); put('mirostat_tau', 's-tau', true);
  put('mirostat_eta', 's-eta', true);
  const stop = val('s-stop');
  if (stop) out.stop = stop.includes(',') ? stop.split(',').map(x => x.trim())
                                          : stop;
  return out;
}}

function currentEngineIndex() {{
  return +document.getElementById('eng-select').value;
}}

function engineSelected() {{
  const e = engines[currentEngineIndex()];
  if (!e) return;
  document.getElementById('eng-name').value = e.name;
  document.getElementById('eng-kind').value = e.kind;
  document.getElementById('eng-url').value = e.base_url || '';
  document.getElementById('eng-default').checked =
    e.name.toLowerCase() === engDefault.toLowerCase();
  samplingToForm(e.sampling || {{}});
  show('eng-loadopts', e.load_options && Object.keys(e.load_options).length
    ? e.load_options : '(none recorded)');
  engDirty = false;
  
  // merged grid: populate uniform fields
  const mSel = document.getElementById('eng-model');
  if (mSel && e.model) mSel.value = e.model;
  const ctxLenInput = document.getElementById('eng-ctxlen');
  if (ctxLenInput) ctxLenInput.value = String(e.load_options?.context ?? e.load_options?.contextLength ?? 8192);
  const fitCtx = document.getElementById('fit-ctx');
  if (fitCtx) fitCtx.value = String(e.load_options?.context ?? 8192);
  const thr = document.getElementById('eng-threads');
  if (thr) thr.value = e.load_options?.threads ?? '';
  const gpu = document.getElementById('eng-gpu');
  if (gpu) gpu.value = String(e.load_options?.gpu_layers ?? 999);
  const flash = document.getElementById('eng-flash');
  if (flash) flash.value = e.load_options?.flash_attn ? 'on' : '';
  const par = document.getElementById('eng-parallel');
  if (par) par.value = e.load_options?.parallel_slots ?? '';
  const bat = document.getElementById('eng-batch');
  if (bat) bat.value = e.load_options?.batch_size ?? '';
  const ubat = document.getElementById('eng-ubatch');
  if (ubat) ubat.value = e.load_options?.ubatch_size ?? '';
  const ctk = document.getElementById('eng-ctk');
  if (ctk) ctk.value = e.load_options?.cache_type_k ?? '';
  const ctv = document.getElementById('eng-ctv');
  if (ctv) ctv.value = e.load_options?.cache_type_v ?? '';
  const alias = document.getElementById('eng-alias');
  if (alias) alias.value = e.load_options?.alias ?? '';
  const disp = document.getElementById('eng-displayName');
  if (disp) disp.value = e.displayName || e.name || '';
  const apik = document.getElementById('eng-apikey');
  if (apik) apik.value = '';
document.getElementById('eng-msg').textContent = '';
}}

function engineAdd() {{
  engines.push({{name: 'engine-' + (engines.length + 1), kind: 'llama_cpp',
                base_url: '', load_options: {{}}, capabilities: ['streaming'],
                sampling: {{}}}});
  renderEngineSelect(engines.length - 1);
  engineSelected();
  engDirty = true;
}}

function renderEngineSelect(selectIndex) {{
  const sel = document.getElementById('eng-select');
  sel.innerHTML = '';
  engines.forEach((e, i) => {{
    const o = document.createElement('option');
    o.value = i;
    o.textContent = e.name + (e.name.toLowerCase() === engDefault.toLowerCase()
                              ? '  (default)' : '');
    sel.appendChild(o);
  }});
  sel.value = selectIndex;
  refreshAbSelects();
}}

function refreshAbSelects() {{
  const ids = ['ab-select-a','ab-select-b','ab-profile-a','ab-profile-b'];
  for (const id of ids) {{
    const sel = document.getElementById(id);
    if (!sel) continue;
    const cur = sel.value;
    sel.innerHTML = '';
    engines.forEach(e => {{
      const o = document.createElement('option');
      o.value = e.name;
      o.textContent = e.name;
      sel.appendChild(o);
    }});
    if (cur && [...sel.options].some(o => o.value === cur)) sel.value = cur;
    else if (sel.options.length) {{
      if (id.endsWith('-a') && sel.options.length >= 1) sel.value = sel.options[0].value;
      if (id.endsWith('-b') && sel.options.length >= 2) sel.value = sel.options[1].value;
      else if (id.endsWith('-b') && sel.options.length >= 1) sel.value = sel.options[0].value;
    }}
  }}
}}

async function benchAb() {{
  const a = document.getElementById('ab-select-a')?.value || document.getElementById('ab-profile-a')?.value || '';
  const b = document.getElementById('ab-select-b')?.value || document.getElementById('ab-profile-b')?.value || '';
  const basePort = parseInt(document.getElementById('ab-baseport')?.value || '1234', 10);
  const btn = document.getElementById('ab-bench-btn');
  const winnerEl = document.getElementById('ab-winner');
  const badgeEl = document.getElementById('ab-badge');
  const ra = document.getElementById('ab-result-a');
  const rb = document.getElementById('ab-result-b');
  if (!a || !b) {{ if (winnerEl) winnerEl.textContent = 'select two profiles'; return; }}
  if (btn) btn.disabled = true;
  if (winnerEl) winnerEl.textContent = 'benching…';
  if (badgeEl) badgeEl.textContent = 'benching…';
  if (ra) ra.textContent = '';
  if (rb) rb.textContent = '';
  try {{
    let r;
    try {{
      r = await api('/v1/engines/ab/bench', 'POST', {{profile_a: a, profile_b: b, basePort: basePort, base_port: basePort}});
    }} catch (e) {{
      // fallback aliases
      try {{ r = await api('/v1/engines/bench', 'POST', {{profile_a: a, profile_b: b, basePort: basePort}}); }}
      catch (e2) {{ r = await api('/v1/ab/bench', 'POST', {{a: a, b: b, basePort: basePort}}); }}
    }}
    const winner = r.winner || (r.a_tok_per_sec > r.b_tok_per_sec ? 'A' : r.b_tok_per_sec > r.a_tok_per_sec ? 'B' : 'tie');
    const atps = (r.a_tok_per_sec ?? r.a?.tok_per_sec ?? r.tokens_per_sec_a ?? 0).toFixed(1);
    const btps = (r.b_tok_per_sec ?? r.b?.tok_per_sec ?? r.tokens_per_sec_b ?? 0).toFixed(1);
    const txt = `winner: ${{winner}} — A ${{atps}} tok/s vs B ${{btps}} tok/s`;
    if (winnerEl) {{ winnerEl.textContent = txt; winnerEl.className = 'winner-badge winner-' + winner.toLowerCase(); }}
    if (badgeEl) {{ badgeEl.textContent = txt; badgeEl.style.display = ''; badgeEl.className = 'winner-badge'; }}
    if (ra) ra.textContent = `A (${{a}}) ${{atps}} tok/s`;
    if (rb) rb.textContent = `B (${{b}}) ${{btps}} tok/s`;
  }} catch (e) {{
    const msg = 'error: ' + (e.message || String(e));
    if (winnerEl) winnerEl.textContent = msg;
    if (badgeEl) {{ badgeEl.textContent = msg; badgeEl.style.display = ''; }}
  }} finally {{
    if (btn) btn.disabled = false;
  }}
}}

function collectEngines() {{
  const e = engines[currentEngineIndex()];
  if (e) {{
    e.name = val('eng-name') || e.name;
    e.kind = document.getElementById('eng-kind').value;
    e.base_url = val('eng-url');
    e.sampling = formToSampling();
  }}
  return engines;
}}

async function loadEngines() {{
  try {{
    const data = await api('/v1/engines');
    engines = data.engines.map(e => ({{...e}}));
    engDefault = data.default || '';
    engRevision = data.revision ?? 0;
    // also fetch provider revision for atomic guard
    try {{ const p = await api('/v1/provider/config'); provRevision = p.revision ?? 0; provDefault = p.default || provDefault; }} catch(e) {{}}
    renderEngineSelect(0);
    engineSelected();
  }} catch (e) {{ show('eng-loadopts', String(e)); }}
}}

// unified Engine profiles grid — single form edits both namespaces atomically (ctx.models lifecycle + ctx.llm provider profile)
let engRevision = 0;
let provRevision = 0;
let engDirty = false;

// settings façade mirroring dsh-settings: mutate(ns, payload, expectedRevision)
const settings = {{
  mutate: async (ns, payload, expectedRevision) => {{
    const path = ns === 'llm' ? '/v1/provider/config' : '/v1/engines';
    const body = ns === 'llm'
      ? {{ providers: payload.providers, default: payload.default, persist: true, expectedRevision }}
      : {{ engines: payload.engines, default: payload.default, persist: true, expectedRevision }};
    const r = await api(path, 'POST', body);
    if (ns === 'llm') provRevision = r.revision ?? provRevision + 1;
    else engRevision = r.revision ?? engRevision + 1;
    return r;
  }}
}};

async function requestLoad(opts) {{
  // ctx.models lifecycle: spawn /v1/server/start with the profile's load_options
  return api('/v1/server/start', 'POST', opts);
}}

async function saveEngines() {{
  // legacy alias — now delegates to the unified saver
  return saveEngineProfile();
}}

async function saveEngineProfile() {{
  const form = document.getElementById('engine-profile-form');
  if (!form) return saveEnginesLegacy();
  const idx = currentEngineIndex();
  const eng = engines[idx];
  if (!eng) return;
  // collect Endpoint + Model + Context into the two namespaces
  eng.name = val('eng-name') || eng.name;
  eng.kind = document.getElementById('eng-kind').value;
  eng.base_url = val('eng-url');
  eng.displayName = val('eng-displayName') || eng.name;
  // Model dropdown from local library
  const selModel = val('eng-model') || document.getElementById('launch-model-select')?.value || '';
  eng.model = selModel;
  // Context
  const ctxLen = parseInt(val('eng-ctxlen') || document.getElementById('fit-ctx')?.value || '8192', 10);
  eng.load_options = eng.load_options || {{}};
  eng.load_options.context = ctxLen;
  eng.load_options.contextLength = ctxLen;
  // Load grid
  const threadsVal = val('eng-threads'); if (threadsVal) eng.load_options.threads = parseInt(threadsVal,10); else delete eng.load_options.threads;
  const gpuVal = val('eng-gpu'); if (gpuVal) eng.load_options.gpu_layers = parseInt(gpuVal,10);
  const flashVal = document.getElementById('eng-flash')?.value; if (flashVal) eng.load_options.flash_attn = flashVal === 'on'; else delete eng.load_options.flash_attn;
  const parVal = val('eng-parallel'); if (parVal) eng.load_options.parallel_slots = parseInt(parVal,10); else delete eng.load_options.parallel_slots;
  const batchVal = val('eng-batch'); if (batchVal) eng.load_options.batch_size = parseInt(batchVal,10); else delete eng.load_options.batch_size;
  const ubatchVal = val('eng-ubatch'); if (ubatchVal) eng.load_options.ubatch_size = parseInt(ubatchVal,10); else delete eng.load_options.ubatch_size;
  eng.load_options.cache_type_k = val('eng-ctk') || undefined;
  eng.load_options.cache_type_v = val('eng-ctv') || undefined;
  eng.load_options.alias = val('eng-alias') || undefined;
  if (val('eng-alias')) eng.load_options.alias = val('eng-alias');
  // Sampling 4x2
  eng.sampling = formToSampling();
  // Advanced (mirostat etc already in sampling)
  // persist expectedRevision guard — 409 conflict if stale
  const msgEl = document.getElementById('eng-msg');
  const saveBtn = document.getElementById('eng-save');
  if (saveBtn) saveBtn.disabled = true;
  if (msgEl) msgEl.textContent = 'saving…';
  // build llm provider profile (ctx.llm) from Endpoint section
  const llmPayload = {{
    providers: [{{ name: eng.name, base_url: eng.base_url, api_key: val('eng-apikey') || '', model: selModel || eng.model || '', displayName: eng.displayName || eng.name }}],
    default: document.getElementById('eng-default')?.checked ? eng.name : provDefault,
  }};
  const modelsPayload = {{
    engines: engines,
    default: document.getElementById('eng-default')?.checked ? eng.name : engDefault,
  }};
  const loadOpts = {{
    model: selModel || eng.model || null,
    ctx_size: ctxLen,
    ngl: eng.load_options.gpu_layers ?? 999,
    auto_fit: document.getElementById('eng-autofit')?.checked ?? false,
    visible_devices: val('eng-devices') || null,
    threads: eng.load_options.threads ?? null,
    flash_attn: eng.load_options.flash_attn ?? false,
    parallel_slots: eng.load_options.parallel_slots ?? null,
    cache_type_k: eng.load_options.cache_type_k ?? null,
    cache_type_v: eng.load_options.cache_type_v ?? null,
    batch_size: eng.load_options.batch_size ?? null,
    ubatch_size: eng.load_options.ubatch_size ?? null,
    alias: eng.load_options.alias ?? null,
    mlock: eng.load_options.mlock ?? false,
    no_mmap: eng.load_options.no_mmap ?? false,
  }};
  try {{
    // single Save that atomically mutates both namespaces + lifecycle
    // ctx.models lifecycle + ctx.llm provider profile
    await Promise.all([

    // single Save: Promise.all([settings.mutate(llm), settings.mutate(models), requestLoad]) with expectedRevision guard      settings.mutate('llm', llmPayload, provRevision),
      settings.mutate('models', modelsPayload, engRevision),
      requestLoad(loadOpts),
    ]);
    if (msgEl) msgEl.textContent = 'saved + loaded';
    engDirty = false;
  }} catch (e) {{
    const txt = String(e.message || e);
    if (txt.includes('409') || txt.toLowerCase().includes('revision')) {{
      if (msgEl) msgEl.textContent = 'conflict: stale revision — reload and retry (expectedRevision guard)';
    }} else {{
      if (msgEl) msgEl.textContent = 'save failed: ' + txt;
    }}
    throw e;
  }} finally {{
    if (saveBtn) saveBtn.disabled = false;
    setTimeout(() => {{ if (msgEl && msgEl.textContent.startsWith('saved')) msgEl.textContent = ''; }}, 3000);
  }}
}}

async function saveEnginesLegacy() {{

  const list = collectEngines();
  const defName = document.getElementById('eng-default').checked
    ? (val('eng-name') || (engines[currentEngineIndex()] || {{}}).name || '')
    : engDefault;
  try {{
    const r = await api('/v1/engines', 'POST',
      {{engines: list, default: defName, persist: true}});
    engDefault = r.default;
    renderEngineSelect(currentEngineIndex());
    document.getElementById('eng-msg').textContent =
      'saved to ' + r.persisted_to;
    engDirty = false;
  }} catch (e) {{ document.getElementById('eng-msg').textContent = String(e); }}
}}

/* ---------------- Auto presets: hardware + model size + gguf-metadata → load_options ---------------- */
function computeAutoPreset(size_gb, hw, file, ggufMeta) {{
  const avail = Number(hw.available_gb ?? hw.vram_gb ?? hw.total_ram_gb ?? 8) || 8;
  const name = (file || '').toLowerCase();
  const meta = ggufMeta || {{}};
  // gguf-metadata: prefer declared parameter count / block_count when available
  const paramsB = Number(meta.parameter_count) ? Number(meta.parameter_count)/1e9
    : Number(meta['general.parameter_count']) ? Number(meta['general.parameter_count'])/1e9
    : null;
  const blockCount = Number(meta.block_count ?? meta['llama.block_count'] ?? meta['qwen.block_count']) || null;
  // Heuristic layers for known families when metadata missing
  let estLayers = blockCount;
  if (!estLayers) {{
    if (name.includes('32b') || name.includes('30b') || (paramsB && paramsB>=30) || size_gb>15) estLayers = 62;
    else if (name.includes('14b') || name.includes('13b') || (paramsB && paramsB>=13)) estLayers = 40;
    else if (name.includes('7b') || name.includes('8b') || (paramsB && paramsB>=7)) estLayers = 32;
    else if (name.includes('4b') || name.includes('3b') || (paramsB && paramsB>=3)) estLayers = 36;
    else if (name.includes('qwen3-4b')) estLayers = 36;
    else if (name.includes('qwen3-32b')) estLayers = 64;
    else estLayers = 32;
  }}
  let gpu_layers, ctx, threads, flash_attn, cache_k, cache_v;
  const low = name;
  const is32 = low.includes('32b') || low.includes('30b') || (paramsB && paramsB>=30) || size_gb>15;
  const is4 = low.includes('4b') || low.includes('3b') || low.includes('qwen3-4b') || (paramsB && paramsB>=3 && paramsB<6) || (size_gb>=2 && size_gb<6);
  if (is32) {{
    // qwen3-32b example: on 8GB → offload 12 + 4k
    if (avail <= 9) {{ gpu_layers = 12; ctx = 4096; }}
    else if (avail <= 16) {{ gpu_layers = 20; ctx = 8192; }}
    else if (avail <= 24) {{ gpu_layers = Math.min(40, estLayers); ctx = 8192; }}
    else {{ gpu_layers = 999; ctx = 16384; }}
  }} else if (is4) {{
    // qwen3-4b example: on 8GB → gpu_layers 28 + 8k ctx
    if (avail <= 9) {{ gpu_layers = 28; ctx = 8192; }}
    else if (avail <= 16) {{ gpu_layers = 999; ctx = 16384; }}
    else {{ gpu_layers = 999; ctx = 32768; }}
  }} else {{
    if (size_gb + 2.0 <= avail * 0.9) {{ gpu_layers = 999; ctx = 8192; }}
    else if (size_gb + 1.0 <= avail * 1.1) {{ gpu_layers = Math.min(28, estLayers); ctx = 8192; }}
    else {{ gpu_layers = Math.min(12, estLayers); ctx = 4096; }}
  }}
  // Clamp to real layer count
  if (gpu_layers !== 999) gpu_layers = Math.min(gpu_layers, estLayers);
  // Other load options tuned with context
  threads = null;
  flash_attn = ctx >= 8192;
  cache_k = null; cache_v = null;
  if (ctx > 16384) {{ cache_k = 'q8_0'; cache_v = 'q8_0'; }}
  return {{ gpu_layers, context: ctx, ctx_size: ctx, threads, flash_attn, cache_type_k: cache_k, cache_type_v: cache_v, block_count: estLayers, gguf_metadata: meta }};
}}
async function fillGpuDevices() {{
  const sel = document.getElementById('eng-devices');
  if (!sel || sel.dataset.filled === '1') return;
  try {{
    const st = await api('/v1/server/status');
    const devs = (st.hardware && st.hardware.devices) || [];
    for (const d of devs) {{
      const opt = document.createElement('option');
      opt.value = String(d.index);
      opt.textContent = `${{d.index}}: ${{d.name || d.bdf || 'gpu'}} ${{d.free_gb ?? '?'}}GB free${{d.display ? ' (display)' : ''}}`;
      sel.appendChild(opt);
    }}
    sel.dataset.filled = '1';
  }} catch (e) {{}}
}}
async function engineAuto() {{
  const btn = document.getElementById('eng-auto');
  const msg = document.getElementById('eng-auto-msg');
  const selFile = window._selectedFile || (document.getElementById('launch-model-select') ? document.getElementById('launch-model-select').value : '');
  if (!selFile) {{
    if (msg) msg.textContent = 'select a model first';
    setTimeout(()=>{{ if(msg) msg.textContent=''; }}, 2500);
    return;
  }}
  if (btn) btn.disabled = true;
  if (msg) msg.textContent = 'computing preset…';
  try {{
    // Prefer server-side preset API, fallback to client-side hardware/size_gb/gguf-metadata
    let preset = null;
    try {{
      const data = await api('/v1/engines/preset?file=' + encodeURIComponent(selFile));
      preset = data.load_options || data.preset || data;
      if (preset.gpu_layers == null && preset.gpu_layers !== 0) throw new Error('no preset');
    }} catch (_e) {{
      // Client-side: use cached _libCache if fresh, else fetch
      let local;
      if(_libCache.system && (Date.now()-_libCache.ts < 30000)) local={{models: _libCache.system}};
      else local=await api('/v1/models/local');
      const status=await api('/v1/server/status');
      const hardware = status.hardware || status;
      const entry = (local.models||[]).find(m=>m.file===selFile);
      const size_gb = entry ? Number(entry.size_gb)||0 : 0;
      const gguf_metadata = entry ? (entry.gguf_metadata || entry.metadata || {{}}) : {{}};
      preset = computeAutoPreset(size_gb, hardware, selFile, gguf_metadata);
    }}
    const ctxVal = preset.context ?? preset.ctx_size ?? 8192;
    const nglVal = preset.gpu_layers ?? 999;
    // Update Engine profile load_options and display
    const idx = currentEngineIndex();
    if (engines[idx]) {{
      engines[idx].load_options = {{ ...(engines[idx].load_options||{{}}), context: ctxVal, gpu_layers: nglVal }};
      if (preset.threads) engines[idx].load_options.threads = preset.threads;
      if (preset.flash_attn) engines[idx].load_options.flash_attn = true;
      if (preset.cache_type_k) engines[idx].load_options.cache_type_k = preset.cache_type_k;
      if (preset.cache_type_v) engines[idx].load_options.cache_type_v = preset.cache_type_v;
      show('eng-loadopts', engines[idx].load_options);
    }}
    // Directly set values in the form (Launch panel + Engine fit slider) and then save
    const ctxInput = document.getElementById('ctx');
    const nglInput = document.getElementById('ngl');
    if (ctxInput) ctxInput.value = String(ctxVal);
    if (nglInput) nglInput.value = String(nglVal);
    const fitCtx = document.getElementById('fit-ctx');
    if (fitCtx) {{ fitCtx.value = String(ctxVal); try{{ updateFit(); }}catch(e){{}} }}
    // Optional: reflect other load options in Launch panel
    if (preset.threads != null) {{ const el=document.getElementById('l-threads'); if(el) el.value=preset.threads; }}
    if (preset.flash_attn != null) {{ const el=document.getElementById('l-fa'); if(el) el.checked=!!preset.flash_attn; }}
    if (preset.cache_type_k) {{ const el=document.getElementById('l-ctk'); if(el) el.value=preset.cache_type_k; }}
    if (preset.cache_type_v) {{ const el=document.getElementById('l-ctv'); if(el) el.value=preset.cache_type_v; }}
    engDirty = true;
    if (msg) msg.textContent = `preset: ${{nglVal}} layers / ${{ctxVal}} ctx — saving…`;
    await saveEngines();
    if (msg) msg.textContent = `Auto: ${{nglVal}} layers / ${{ctxVal}} ctx — saved`;
  }} catch (e) {{
    if (msg) msg.textContent = 'auto failed: ' + e.message;
  }} finally {{
    if (btn) btn.disabled = false;
    setTimeout(()=>{{ if(msg && msg.textContent.startsWith('Auto:')) msg.textContent=''; }}, 4000);
  }}
}}

async function exportEngineProfiles() {{
  const msg = document.getElementById('import-export-msg');
  try {{
    const data = {{ engines, exported_at: new Date().toISOString(), version: 1 }};
    const blob = new Blob([JSON.stringify(data, null, 2)], {{type: 'application/json'}});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'engine-profiles.json';
    a.click();
    URL.revokeObjectURL(url);
    if (msg) msg.textContent = `exported ${{engines.length}} profiles`;
  }} catch (e) {{
    if (msg) msg.textContent = 'export failed: ' + e.message;
  }} finally {{
    setTimeout(()=>{{ if(msg) msg.textContent=''; }}, 3000);
  }}
}}

async function importEngineProfiles(input) {{
  const file = input.files[0];
  if (!file) return;
  const msg = document.getElementById('import-export-msg');
  try {{
    const text = await file.text();
    const data = JSON.parse(text);
    const incoming = data.engines || data.profiles || data;
    if (!Array.isArray(incoming)) throw new Error('invalid bundle: missing engines array');
    for (const eng of incoming) {{
      if (!eng.name) throw new Error('engine missing name');
      const idx = engines.findIndex(e => e.name === eng.name);
      if (idx >= 0) engines[idx] = eng;
      else engines.push(eng);
    }}
    renderEngineSelect(0);
    engineSelected();
    if (msg) msg.textContent = `imported ${{incoming.length}} profiles — save to persist`;
  }} catch (e) {{
    if (msg) msg.textContent = 'import failed: ' + e.message;
  }} finally {{
    input.value = '';
    setTimeout(()=>{{ if(msg) msg.textContent=''; }}, 4000);
  }}
}}

/* ---------------- Fit estimator S2: VRAM needs vs has + context slider ---------------- */
let _fitHardware = null;
let _fitHardwarePromise = null;
let _fitModelsCache = null;
async function getFitHardware() {{
  if (_fitHardware) return _fitHardware;
  if (_fitHardwarePromise) return _fitHardwarePromise;
  _fitHardwarePromise = api('/v1/server/status').then(s => {{
    const hw = s.hardware || s;
    const has = hw.available_gb ?? hw.available_ram_gb ?? hw.total_ram_gb ?? hw.totalRamGb ?? 8;
    const total = hw.total_ram_gb ?? hw.totalRamGb ?? has;
    return {{ available_gb: Number(has) || 8, total_ram_gb: Number(total) || 8, vram_gb: hw.vram_gb ?? null, devices: hw.devices || [] }};
  }}).catch(()=>({{available_gb:8, total_ram_gb:8, vram_gb:null, devices:[]}})).then(h=>{{_fitHardware=h; return h;}});
  return _fitHardwarePromise;
}}
async function getFitModels() {{
  if (_fitModelsCache && Date.now()-_fitModelsCache.ts < 5000) return _fitModelsCache.data;
  // Use _libCache if fresh (avoid extra fetch)
  if(_libCache.system && (Date.now()-_libCache.ts < 30000)) {{
    const data={{models_dir: document.getElementById('library-path')?.value||'', models: _libCache.system}};
    _fitModelsCache={{data, ts: Date.now()}};
    return data;
  }}
  const data = await api('/v1/models/local');
  _fitModelsCache = {{data, ts: Date.now()}};
  return data;
}}
function formatCtx(v) {{
  const n = Number(v);
  if (n>=1024) return (n/1024)|0 + 'k';
  return String(n);
}}
async function updateFit() {{
  const selFile = window._selectedFile || (document.getElementById('launch-model-select') ? document.getElementById('launch-model-select').value : '');
  const box = document.getElementById('eng-fit');
  const ctxEl = document.getElementById('fit-ctx');
  if (!box || !ctxEl) return;
  if (!selFile) {{ box.style.display='none'; return; }}
  box.style.display='';
  let size_gb = 0;
  try {{
    const local = await getFitModels();
    const entry = (local.models||[]).find(m=>m.file===selFile);
    if (entry && entry.size_gb!=null) size_gb = Number(entry.size_gb)||0;
  }} catch(e) {{}}
  const ctx = parseInt(ctxEl.value,10)||8192;
  const label = document.getElementById('fit-ctx-label');
  if (label) label.textContent = formatCtx(ctx);
  let hw;
  try {{ hw = await getFitHardware(); }} catch(e) {{ hw = {{available_gb:8}}; }}
  const has_gb = Number(hw.available_gb)||8;
  const kv_gb = (ctx/1024)*0.25;
  const needs_gb = size_gb + kv_gb;
  const ratio = has_gb>0 ? needs_gb/has_gb : 1;
  const needsEl = document.getElementById('fit-needs');
  const hasEl = document.getElementById('fit-has');
  const needsVal = document.getElementById('fit-needs-val');
  const pctEl = document.getElementById('fit-pct');
  const fill = document.getElementById('fit-fill');
  const btn = document.getElementById('eng-load-fit');
  if (needsEl) needsEl.textContent = `needs ${{needs_gb.toFixed(1)}}GB`;
  if (hasEl) hasEl.textContent = `/ ${{has_gb.toFixed(1)}}GB`;
  if (needsVal) needsVal.textContent = `${{needs_gb.toFixed(1)}}GB`;
  if (pctEl) {{
    const fits = needs_gb <= has_gb;
    pctEl.textContent = fits ? (ratio>0.9 ? 'Tight' : 'Fits') : 'Too large';
    pctEl.className = 'fit-pct ' + (fits ? (ratio>0.7 ? 'warn' : 'ok') : 'bad');
  }}
  if (fill) {{
    fill.style.width = Math.min(100, ratio*100).toFixed(1)+'%';
    fill.className = 'fit-fill ' + (ratio>1 ? 'red' : ratio>0.7 ? 'amber' : 'green');
  }}
  if (btn) {{
    const fits = needs_gb <= has_gb;
    btn.disabled = !fits;
    btn.title = fits ? `Fits — load ${{selFile.split('/').pop().split('\\\\').pop()}} at ${{formatCtx(ctx)}}` : `needs ${{needs_gb.toFixed(1)}}GB but you have ${{has_gb.toFixed(1)}}GB — reduce context or free VRAM`;
  }}
}}
function engineLoadFromFit() {{
  const selFile = window._selectedFile || (document.getElementById('launch-model-select') ? document.getElementById('launch-model-select').value : '');
  const ctx = parseInt(document.getElementById('fit-ctx')?.value||'8192',10);
  if (!selFile) return;
  const modelInput = document.getElementById('model');
  const ctxInput = document.getElementById('ctx');
  if (modelInput) modelInput.value = selFile;
  if (ctxInput) ctxInput.value = String(ctx);
  const sel = document.getElementById('launch-model-select');
  if (sel) sel.value = selFile;
  startServer(document.querySelector('button[onclick="startServer(this)"]') || null);
}}
(function initFitHooks(){{
  const ctxEl = document.getElementById('fit-ctx');
  if (ctxEl) ctxEl.addEventListener('input', ()=>{{ updateFit(); }});
  let lastSel = null;
  setInterval(()=>{{
    const cur = window._selectedFile || (document.getElementById('launch-model-select') ? document.getElementById('launch-model-select').value : '');
    if (cur !== lastSel) {{ lastSel = cur; updateFit(); }}
  }}, 400);
  const launchSel = document.getElementById('launch-model-select');
  if (launchSel) launchSel.addEventListener('change', ()=>{{ setTimeout(updateFit, 50); }});
  setTimeout(()=>{{ getFitHardware().then(()=>updateFit()); }}, 600);
}})();

/* ----------------------------- splinter tab ------------------------------ */
const SPLINTER_NUMERIC = [['max_context', 'h-maxctx'], ['max_tokens', 'h-maxtok'],
  ['stale_threshold', 'h-stale'], ['dedup_threshold', 'h-dedup'],
  ['drift_threshold', 'h-drift'], ['remembrance_threshold', 'h-remem'],
  ['vocab_boost', 'h-vocab']];

function hiveToForm(cfg) {{
  for (const [key, id] of SPLINTER_NUMERIC)
    document.getElementById(id).value =
      (cfg[key] === undefined || cfg[key] === null) ? '' : cfg[key];
  document.getElementById('h-conf').value = cfg.confidence_mode || 'off';
  document.getElementById('h-sanitize').checked = !!cfg.sanitize_context;
  document.getElementById('h-hedge').checked = !!cfg.filter_hedge_replies;
  document.getElementById('h-medium').checked = !!cfg.enable_medium;
  document.getElementById('h-comb').checked = !!cfg.comb_enabled;
  document.getElementById('h-combk').value = cfg.comb_top_k ?? '';
  document.getElementById('h-combgate').value = cfg.comb_gate_threshold ?? '';
  document.getElementById('h-combmax').value = cfg.comb_max_records ?? '';
  document.getElementById('h-combrel').checked = !!cfg.comb_relevant_only;
}}

function collectHiveOverrides() {{
  const out = {{}};
  const defaults = window.__hiveDefaults || {{}};
  const changed = (key, value) => defaults[key] === undefined
    || JSON.stringify(defaults[key]) !== JSON.stringify(value);
  for (const [key, id] of SPLINTER_NUMERIC) {{
    const v = num(id);
    if (v !== null && changed(key, v)) out[key] = v;
  }}
  const conf = document.getElementById('h-conf').value;
  if (changed('confidence_mode', conf)) out.confidence_mode = conf;
  const checks = [['h-sanitize', 'sanitize_context'],
    ['h-hedge', 'filter_hedge_replies'], ['h-medium', 'enable_medium']];
  for (const [id, key] of checks) {{
    const v = document.getElementById(id).checked;
    if (changed(key, v)) out[key] = v;
  }}
  if (document.getElementById('h-comb').checked) {{
    out.comb_enabled = true;
    const k = num('h-combk'); if (k !== null) out.comb_top_k = k;
    const g = num('h-combgate'); if (g !== null) out.comb_gate_threshold = g;
    const m = num('h-combmax'); if (m !== null) out.comb_max_records = m;
    out.comb_relevant_only = document.getElementById('h-combrel').checked;
    out.comb_dir = 'harness_comb';
  }}
  return out;
}}

async function loadHiveDefaults() {{
  try {{
    const cfg = await api('/v1/splinter/defaults');
    window.__hiveDefaults = cfg;
    hiveToForm(cfg);
    hiveOverrides = collectHiveOverrides();
    document.getElementById('splinter-msg').textContent =
      Object.keys(hiveOverrides).length + ' override(s) active for new conversations';
  }} catch (e) {{ document.getElementById('splinter-msg').textContent = String(e); }}
}}

function resetHiveDefaults() {{
  if (window.__hiveDefaults) hiveToForm(window.__hiveDefaults);
  hiveOverrides = collectHiveOverrides();
  document.getElementById('splinter-msg').textContent = 'defaults restored';
}}

for (const key of ['h-maxctx','h-maxtok','h-stale','h-dedup','h-drift',
                   'h-remem','h-vocab','h-conf','h-sanitize','h-hedge',
                   'h-medium','h-comb','h-combk','h-combgate','h-combmax',
                   'h-combrel']) {{
  const el = document.getElementById(key);
  if (el) el.addEventListener('change', () => {{
    hiveOverrides = collectHiveOverrides();
    document.getElementById('splinter-msg').textContent =
      Object.keys(hiveOverrides).length + ' override(s) active for new conversations';
  }});
}}

/* --------------------------- providers tab ---------------------------- */
let providers = [];
let provDefault = '';

async function loadProviders() {{
  try {{
    const data = await api('/v1/provider/config');
    providers = data.providers.map(p => ({{...p}}));
    provDefault = data.default || '';
    renderProviders();
  }} catch (e) {{ document.getElementById('prov-msg').textContent = String(e); }}
}}

function renderProviders() {{
  const wrap = document.getElementById('prov-list');
  wrap.innerHTML = '';
  providers.forEach((p, i) => {{
    const row = document.createElement('div');
    row.className = 'row';
    row.style.borderTop = '1px solid #e8edf2';
    row.style.paddingTop = '.4rem';
    const mk = (placeholder, value, size, onChange, title) => {{
      const inp = document.createElement('input');
      inp.placeholder = placeholder; inp.size = size; inp.value = value || '';
      if (title) inp.title = title;
      inp.addEventListener('input', onChange);
      return inp;
    }};
    const name = mk('name', p.name, 12, v => p.name = v, 'Provider id, e.g. anthropic or openai; key for model routing');
    const url = mk('base_url', p.base_url, 34, v => p.base_url = v, 'OpenAI-compatible endpoint root, e.g. https://api.anthropic.com/v1');
    const model = mk('model', p.model, 18, v => p.model = v, 'Default model id for this provider, e.g. claude-3-5-sonnet-20241022');
    const key = mk('api_key', p.api_key, 16, v => p.api_key = v, 'API key for this provider; stored masked as *** and sent as Bearer token');
    const def = document.createElement('input');
    def.type = 'radio'; def.name = 'prov-default'; def.checked =
      p.name.toLowerCase() === provDefault.toLowerCase();
    def.title = 'default provider';
    def.addEventListener('change', () => provDefault = p.name);
    const del = document.createElement('button');
    del.textContent = '✕'; del.title = 'remove';
    del.addEventListener('click', () => {{
      providers.splice(i, 1); renderProviders(); }});
    const lbl = document.createElement('label');
    lbl.className = 'inline'; lbl.title = 'default';
    lbl.appendChild(def);
    row.appendChild(lbl);
    row.appendChild(name); row.appendChild(url); row.appendChild(model);
    row.appendChild(key); row.appendChild(del);
    wrap.appendChild(row);
  }});
  if (!providers.length)
    wrap.innerHTML = '<div class="note">(none — add one)</div>';
}}

function providerAdd() {{
  providers.push({{name: '', base_url: 'http://', api_key: '', model: '',
                  headers: {{}}}});
  renderProviders();
}}

async function saveProviders() {{
  const list = providers
    .filter(p => p.name && p.base_url)
    .map(p => ({{name: p.name, base_url: p.base_url,
                api_key: p.api_key || '', model: p.model || '',
                headers: p.headers || {{}}}}));
  try {{
    const r = await api('/v1/provider/config', 'POST',
      {{providers: list, default: provDefault, persist: true}});
    providers = r.providers.map(p => ({{...p}}));
    provDefault = r.default;
    renderProviders();
    document.getElementById('prov-msg').textContent =
      'saved to ' + (r.persisted_to || 'memory');
  }} catch (e) {{ document.getElementById('prov-msg').textContent = String(e); }}
}}

/* --------------------------- inspector ------------------------------- */
for (const btn of document.querySelectorAll('[data-rtab]')) {{
  btn.addEventListener('click', () => {{
    for (const b of document.querySelectorAll('[data-rtab]')) b.classList.remove('active');
    for (const p of document.querySelectorAll('[data-rtab] + .tabpane, .tabpane[data-rtab]')) {{}}
    document.querySelectorAll('.col .tabpane').forEach(p => p.style.display = 'none');
    btn.classList.add('active');
    document.getElementById(btn.dataset.rtab).style.display = '';
    if (btn.dataset.rtab === 'rtab-inspect') fetchInspection();
  }});
}}

async function fetchInspection() {{
  try {{
    const data = await api('/v1/splinter/inspect/' + encodeURIComponent(convId));
    renderInspection(data);
  }} catch (e) {{
    document.getElementById('inspect-summary').textContent =
      'no inspection data: ' + e.message;
  }}
}}

function renderInspection(d) {{
  const kv = (label, v) => `<div class='kv'><span>${{label}}</span><b>${{v}}</b></div>`;
  document.getElementById('inspect-summary').innerHTML =
    kv('Turn', d.turn) + kv('Route', d.routing?.route_to || '-') +
    kv('Budget', `${{d.budget?.used ?? 0}} / ${{d.budget?.total ?? 0}}`) +
    kv('Utilization', `${{((d.budget?.utilization ?? 0) * 100).toFixed(1)}}%`) +
    kv('Top score', d.top_raw_score) +
    kv('Store chunks', d.store_chunks) +
    kv('Drift', d.drift_detected ? '⚠ yes' : 'no');
  const selWrap = document.getElementById('inspect-selected');
  selWrap.innerHTML = '';
  for (const c of d.selected_chunks || []) {{
    selWrap.insertAdjacentHTML('beforeend',
      `<div class='chunkrow sel'><span class='score'>${{c.raw_score}}</span>${{c.id}}` +
      `<div class='preview'>${{c.preview}}</div></div>`);
  }}
  if (!(d.selected_chunks || []).length)
    selWrap.textContent = '(none selected)';
  const dropWrap = document.getElementById('inspect-dropped');
  dropWrap.innerHTML = '';
  for (const c of d.dropped_chunks || []) {{
    dropWrap.insertAdjacentHTML('beforeend',
      `<div class='chunkrow drop'><span class='score'>${{c.raw_score}}</span>${{c.id}}` +
      `<div class='preview'>${{c.preview}}</div></div>`);
  }}
  document.getElementById('inspect-dropped-n').textContent =
    d.dropped_count ? `(${{d.dropped_count}} total)` : '';
  if (!(d.dropped_chunks || []).length)
    dropWrap.textContent = '(none dropped)';
  document.getElementById('inspect-assembled').textContent =
    d.assembled_preview || '(empty)';
  document.getElementById('inspect-timings').textContent =
    JSON.stringify(d.timings ?? {{}}, null, 1);
}}

/* --------------------------- status poll ------------------------------ */
async function refresh() {{
  api('/v1/server/status').then(s => {{
    show('status', s);
    const title = document.getElementById('chat-title');
    if (s.running && s.healthy) {{
      title.textContent = s.instances.length > 1
        ? `${{s.instances.length}} models loaded`
        : 'Loaded: ' + (s.model || 'model');
      title.className = 'ok';
    }} else {{
      title.textContent = s.running ? 'Loading…' : 'No model loaded';
      title.className = 'bad';
    }}
    const topStatus = document.getElementById('top-right-status');
    if (topStatus) {{
      if (s.running && s.healthy) {{
        topStatus.textContent = s.instances.length > 1
          ? `Launch: ${{s.instances.length}} models loaded`
          : 'Launch: Loaded ' + (s.model || 'model');
      }} else {{
        topStatus.textContent = s.running ? 'Launch: Loading…' : 'Launch: No model loaded';
      }}
    }}
    // Loaded-instance cards: per-model unload, port shown.
    const wrap = document.getElementById('instances');
    wrap.innerHTML = '';
    for (const inst of s.instances || []) {{
      const row = document.createElement('div');
      row.className = 'librow' + (inst.healthy ? ' loaded' : '');
      const label = document.createElement('span');
      label.textContent = `${{inst.key}} — port ${{inst.port}}`
        + `${{inst.adopted ? ' (adopted)' : ''}}`;
      const unload = document.createElement('button');
      unload.textContent = 'unload';
      unload.title = 'stop this model server';
      unload.addEventListener('click', async () => {{
        if (!confirm('Unload ' + inst.key + '?')) return;
        try {{
          await api('/v1/server/unload', 'POST', {{key: inst.key}});
          refresh();
        }} catch (e) {{ alert(String(e)); }}
      }});
      row.appendChild(label);
      row.appendChild(unload);
      wrap.appendChild(row);
    }}
    // Chat model picker: every loaded instance's provider + remote providers.
    api('/v1/provider/config').then(pc => {{
      const sel = document.getElementById('chat-provider');
      const current = sel.value;
      sel.innerHTML = '';
      const locals = (s.instances || []).map(i => 'local-' + i.key);
      for (const p of pc.providers) {{
        if (!locals.includes(p.name) && p.name.startsWith('local-')) continue;
        const o = document.createElement('option');
        o.value = p.name;
        o.textContent = p.name + (p.model ? ' — ' + p.model.split('\\\\').pop().split('/').pop() : '');
        sel.appendChild(o);
      }}
      sel.value = pc.default && locals.concat(
        pc.providers.filter(p => !p.name.startsWith('local-')).map(p => p.name)
      ).includes(pc.default) ? pc.default : (sel.firstElementChild ? sel.firstElementChild.value : '');
      if (current && [...sel.options].some(o => o.value === current)) sel.value = current;
    }}).catch(() => {{}});
  }}).catch(e => show('status', String(e)));
  api('/v1/server/log?tail=30').then(l => {{
    if (l.lines.length) show('srvlog', l.lines.join('\\n'));
  }}).catch(() => {{}});
  // Library is loaded on demand when tab-library is clicked, not on every refresh
  // Hugging Face models downloaded on Windows (via HF API) are regular GGUFs in the System library.
  // They appear in System. Right-click → Move to Linux copies them to /mnt/dsh_storage/models via WSL
  // (sudo cp /mnt/c/... → /mnt/dsh_storage/models) so Docker can use them. Linux models are listed
  // separately and can be sent back or deleted. Both locations are tracked.
  api('/v1/models/hub/downloads').then(d => {{
    const lines = d.downloads.map(j => `${{j.filename}}: ${{j.state}} (${{j.elapsed_s}}s)`);
    if (lines.length) show('downloads', lines.join('\\n'));
  }}).catch(() => {{}});
}}

/* --------------------- process manager (S4) --------------------- */
async function refreshProcesses() {{
  const msg = document.getElementById('proc-msg');
  const tbody = document.getElementById('proc-table');
  const sidecarEl = document.getElementById('proc-sidecar');
  try {{
    const data = await api('/v1/processes');
    tbody.innerHTML = '';
    if (sidecarEl) sidecarEl.textContent = `sidecar ${{data.sidecar.pid}} · ${{data.sidecar.memory_mb}} MB · ${{data.sidecar.cpu_percent}}% CPU`;
    for (const p of data.processes || []) {{
      const tr = document.createElement('tr');
      const td = (t) => {{ const c=document.createElement('td'); c.textContent=t; return c; }};
      tr.appendChild(td(String(p.pid)));
      tr.appendChild(td(p.kind || ''));
      const nameTd = document.createElement('td');
      nameTd.title = (p.cmdline || []).join(' ');
      nameTd.textContent = (p.name || '') + (p.key ? ' ('+p.key+')' : '');
      tr.appendChild(nameTd);
      tr.appendChild(td(String(p.cpu_percent ?? '')));
      tr.appendChild(td(String(p.memory_mb ?? '')));
      tr.appendChild(td(p.port ? String(p.port) : ''));
      tr.appendChild(td(p.status || ''));
      const killTd = document.createElement('td');
      const btn = document.createElement('button');
      btn.textContent = 'kill';
      btn.title = p.key ? 'unload '+p.key : 'terminate pid '+p.pid;
      btn.addEventListener('click', async () => {{
        if (!confirm('Kill ' + (p.key || p.pid) + '?')) return;
        btn.disabled = true;
        try {{
          const body = p.key ? {{key: p.key}} : {{pid: p.pid}};
          await api('/v1/processes/kill','POST', body);
          if (msg) msg.textContent = 'killed ' + (p.key || p.pid);
          refreshProcesses(); refresh();
        }} catch(e) {{ alert(String(e)); btn.disabled=false; }}
      }});
      killTd.appendChild(btn);
      tr.appendChild(killTd);
      tbody.appendChild(tr);
    }}
    if (!(data.processes||[]).length) {{
      const tr=document.createElement('tr'); const td=document.createElement('td');
      td.colSpan=8; td.className='note'; td.textContent='(no managed processes)'; tr.appendChild(td); tbody.appendChild(tr);
    }}
    if (msg) {{ msg.textContent=''; setTimeout(()=>{{ if(msg) msg.textContent=''; }},2000); }}
  }} catch(e) {{
    if (tbody) tbody.innerHTML = '<tr><td colspan="8" class="note">load failed: '+String(e).slice(0,200)+'</td></tr>';
    if (msg) msg.textContent = String(e).slice(0,120);
  }}
}}
refreshProcesses();
setInterval(refreshProcesses, 8000);
setTimeout(()=>{{ try{{ refresh(); }}catch(e){{ console.error(e); }} }}, 800);
setInterval(refresh, 15000);
loadHiveDefaults();
// Library tab loads on demand — not on initial refresh to keep UI responsive
console.log('studio ready, tabs active');
document.getElementById('chatin').addEventListener('blur',
  () => setTimeout(() => {{ document.getElementById('sug-chat').innerHTML = ''; }}, 150));
document.getElementById('chatin').focus();
// Refresh status when tab becomes visible again after idle (library only on click)
document.addEventListener('visibilitychange', ()=>{{
  if(document.visibilityState==='visible'){{
    refresh().catch(e=>console.error(e));
    refreshProcesses().catch(e=>console.error(e));
  }}
}});
</script>
</body></html>"""
