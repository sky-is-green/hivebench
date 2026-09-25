"""Stack tab fragment — builder, per-tier Auto preset, residency strip (§B3).

Pure renderers: they take fixture data and return an HTML fragment string, so
they can be tested standalone (no app, no network).  Dynamic values are
HTML-escaped.  ``css()`` returns the sibling ``stack_tab.css`` for the page
shell to inline.

The residency strip consumes the ``StackManager.status()`` shape (§B3 /
``manager.TierRuntime``): ``{"ok", "stack", "tiers": [{"role", "key", "port",
"ctx", "model", "backend", "resident", "vram_gb", "tok_s", "per_card"}],
"warnings"}``.

Frozen at T34 (ADR-L8).  T41 implements it (plus ``stack_tab.css``); T44 mounts
the fragment.
"""

from __future__ import annotations

import html
import json
from dataclasses import fields as _dataclass_fields
from typing import Any, Mapping, Optional, Sequence

from harness.stack.schema import ROLES, Tier

#: Sibling stylesheet read by :func:`css`.
CSS_NAME = "stack_tab.css"

# §5 task row: "css is a module symbol — ``CSS_NAME``" — so the stylesheet is
# *this constant*, not a shipped file; :func:`css` hands it to the page shell,
# which inlines it next to ``harness/studio.css`` (read-only, not re-styled here).
CSS = """
/* stack_tab — Local Stack Control fragment (T41). Scoped .stk-*; the Studio
   palette (honeycomb canvas, black containers, #FFDD00 text inside) comes from
   harness/studio.css, so only fragment-local layout is declared here. */
.stk-tab .stk-head { display: flex; align-items: baseline; gap: .5rem; flex-wrap: wrap; }
.stk-tab .stk-head h2 { margin-top: 0; }
.stk-tab .stk-note { font-size: .82rem; color: rgba(255,221,0,.72); }
.stk-tab .stk-empty { color: rgba(255,221,0,.72); font-size: .86rem; margin: .4rem 0; }
.stk-tab .stk-actions { display: flex; gap: .4rem; align-items: center; flex-wrap: wrap; margin: .7rem 0 .3rem; }
.stk-tab .stk-actions button { font-weight: 700; }
.stk-tab .stk-btn-apply { background: #157a3e; color: #fff; border-color: #000; }
.stk-tab .stk-btn-apply:hover { background: #1a8a4a; }
.stk-tab .stk-btn-unload { background: #b3372c; color: #fff; border-color: #000; }
.stk-tab .stk-btn-unload:hover { background: #cd4437; }

/* --- saved stacks --- */
.stk-saved { display: flex; flex-wrap: wrap; gap: .45rem; margin: .5rem 0; }
.stk-saved-row { display: inline-flex; align-items: center; gap: .4rem; background: #000;
  border: 1.5px solid #000; border-radius: 6px; padding: .3rem .55rem; font-size: .86rem; color: #FFDD00; }
.stk-saved-row .stk-name { font-weight: 700; }
.stk-saved-row .stk-roles { color: rgba(255,221,0,.72); font-size: .78rem; }
.stk-saved-row button { padding: .05rem .5rem; margin: 0; }

/* --- tier builder --- */
.stk-builder { display: flex; flex-direction: column; gap: .55rem; margin: .6rem 0; }
.stk-tier-card { border: 1px solid rgba(255,221,0,.18); border-radius: 8px;
  padding: .6rem .75rem; background: rgba(255,255,255,.04); }
.stk-tier-card > .stk-auto-head { display: flex; align-items: center; gap: .5rem; flex-wrap: wrap; margin-bottom: .45rem; }
.stk-role { font-weight: 700; font-size: .9rem; background: #FFB703; color: #000;
  border: 1.5px solid #000; border-radius: 999px; padding: .05rem .6rem; }
.stk-role-hint { color: rgba(255,221,0,.72); font-size: .78rem; }
.stk-row { display: flex; gap: .45rem; align-items: center; flex-wrap: wrap; }
.stk-row label.inline { color: rgba(255,221,0,.72); font-size: .8rem; }
.stk-row input[type="number"] { width: 6.5rem; }
.stk-row select { max-width: 22rem; }
.stk-btn-auto { background: #157a3e; color: #fff; border-color: #000; font-weight: 700; }
.stk-btn-auto:hover { background: #1a8a4a; }
.stk-msg { font-size: .8rem; color: #FFDD00; }
.stk-advanced { margin-top: .45rem; }
.stk-advanced > summary { color: rgba(255,221,0,.72); font-size: .78rem; }
.stk-advanced .stk-row { margin-top: .35rem; }

/* --- residency strip --- */
.stk-residency { margin-top: .3rem; }
.stk-res-head { display: flex; align-items: center; gap: .5rem; flex-wrap: wrap; margin-bottom: .5rem; }
.stk-res-stack { font-weight: 700; }
.stk-pill { font-size: .76rem; font-weight: 700; border-radius: 999px; padding: .05rem .55rem;
  border: 1.5px solid #000; }
.stk-pill.ok { background: #157a3e; color: #fff; }
.stk-pill.bad { background: #b3372c; color: #fff; }
.stk-pill.warn { background: #FFB703; color: #000; }
.stk-tiers { display: flex; flex-direction: column; gap: .55rem; }
.stk-tier { border: 1.5px solid #000; border-radius: 8px; padding: .55rem .7rem; background: rgba(255,255,255,.04); }
.stk-tier-head { display: flex; align-items: center; gap: .5rem; flex-wrap: wrap; }
.stk-tier-model { font-weight: 600; word-break: break-word; }
.stk-tier-key { color: rgba(255,221,0,.72); font-size: .76rem; }
.stk-metrics { display: flex; gap: .8rem; flex-wrap: wrap; margin: .4rem 0 .2rem; font-size: .82rem; }
.stk-metric b { color: rgba(255,221,0,.72); font-weight: 600; margin-right: .2rem; }
.stk-metric .stk-num { font-variant-numeric: tabular-nums; color: #FFDD00; }
.stk-cards { width: 100%; border-collapse: collapse; margin-top: .35rem; font-size: .78rem; }
.stk-cards th, .stk-cards td { text-align: left; padding: .12rem .4rem .12rem 0; white-space: nowrap; }
.stk-cards th { color: rgba(255,221,0,.72); font-weight: 600; }
.stk-cards td { font-variant-numeric: tabular-nums; color: #FFDD00; }
.stk-bar { height: 8px; border: 1px solid #000; border-radius: 999px; overflow: hidden;
  background: rgba(255,221,0,.14); min-width: 90px; }
.stk-bar > i { display: block; height: 100%; width: 0; }
.stk-bar > i.green { background: #157a3e; }
.stk-bar > i.amber { background: #FFB703; }
.stk-bar > i.red { background: #b3372c; }
.stk-warnings { margin: .5rem 0 0; padding-left: 1.1rem; color: #FFB703; font-size: .8rem; }
"""

# ---------------------------------------------------------------------------
# T41 implementation notes (the frozen signatures above are the contract; this
# block is implementation).
#
# * No JavaScript is emitted.  Every control carries a ``data-action`` (plus
#   ``data-*`` context) and the page shell (T44) wires one delegated listener —
#   so the fragment is inert and error-free standalone, and the action names
#   below are the whole seam T44 has to bind:
#       stack-load · stack-delete · stack-auto · stack-validate · stack-apply
#       stack-unload
# * The residency strip mirrors the §B3 payload one-for-one onto ``data-*``
#   attributes (``data-stack-status`` on the root, ``data-tier``/``data-role``/
#   ``data-key``/``data-port``/``data-ctx``/``data-model``/``data-backend``/
#   ``data-resident``/``data-vram-gb``/``data-tok-s`` per tier, and
#   ``data-card``/``data-weights``/``data-kv``/``data-total``/``data-budget``
#   per card row), so a caller can assert the rendered shape without scraping
#   text.  A ``None`` measurement is the empty string, ``bool`` is
#   ``true``/``false`` — both round-trip through ``json.loads``/``float()``.
# * Tier defaults come from :class:`~harness.stack.schema.Tier` so the pickers
#   cannot drift from the data model.
# ---------------------------------------------------------------------------

#: Tier defaults lifted from the data model (``Tier(ctx=8192, ngl=99, ...)``).
_TIER_DEFAULTS: dict[str, Any] = {f.name: f.default for f in _dataclass_fields(Tier)}
_DEFAULT_CTX = int(_TIER_DEFAULTS["ctx"])
_DEFAULT_NGL = int(_TIER_DEFAULTS["ngl"])

#: Tier role → what the seat must be good at (LOCAL-STACKS §2).  ADR-L1: roles,
#: not sizes — the hint is shown next to every tier card.
ROLE_HINTS: dict[str, str] = {
    "face": "generalist breadth + depth, instruction following, tool calls",
    "worker": "tool-call reliability, instruction following, speed, long context",
    "agency": "sustained tool loops, query reformulation, enough-evidence judgment",
    "mechanics": "latency and cost — single-shot transforms, bulk text munging",
}

#: KV cache types offered per tier (LOCAL-STACKS §4: never Q4_0).
_CACHE_TYPES: tuple[str, ...] = ("q8_0", "q5_0", "q5_1", "f16")

#: Placeholder for a measurement the box could not report.
_DASH = "—"


def _raw(value: Any) -> str:
    """Scalar → attribute string: ``None`` empty, ``bool`` ``true``/``false``."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _esc(value: Any) -> str:
    """Text-node escaping; ``None`` renders as the empty string."""
    if value is None:
        return ""
    return html.escape(str(value), quote=False)


def _attr(name: str, value: Any) -> str:
    """One `` name="value"`` attribute, escaped for a double-quoted context."""
    return ' {}="{}"'.format(name, html.escape(_raw(value), quote=True))


def _json_attr(obj: Any) -> str:
    """A JSON payload as an escaped attribute value."""
    return html.escape(json.dumps(obj, default=str), quote=True)


def _count(value: Any) -> int:
    """Tier count that tolerates both shapes ``list_stacks`` may hand back."""
    if isinstance(value, (list, tuple)):
        return len(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _gb(value: Any) -> str:
    """GiB measurement with two decimals, or the dash placeholder."""
    try:
        return "{:.2f} GiB".format(float(value))
    except (TypeError, ValueError):
        return _DASH


def _rate(value: Any) -> str:
    """tokens/second with one decimal, or the dash placeholder."""
    try:
        return "{:.1f}".format(float(value))
    except (TypeError, ValueError):
        return _DASH


def _int(value: Any) -> str:
    """Thousand-separated integer, or the dash placeholder."""
    try:
        return "{:,}".format(int(value))
    except (TypeError, ValueError):
        return _DASH


def _slug(value: Any) -> str:
    """Attribute-id-safe slug for a tier role (runs of punctuation collapse)."""
    out: list[str] = []
    for char in str(value or ""):
        if char.isalnum():
            out.append(char)
        elif out and out[-1] != "-":
            out.append("-")
    return "".join(out).strip("-") or "tier"


def _field(label: str, control: str) -> str:
    """One ``label.inline`` + control pair."""
    return '<label class="inline">{} {}</label>'.format(_esc(label), control)


def _number_input(ident: str, value: Any, *, low: Any = None, step: Any = None) -> str:
    """A ``<input type=number>`` with an optional minimum and step."""
    attrs = _attr("id", ident) + _attr("min", low) + _attr("step", step)
    return '<input{} type="number" value="{}">'.format(attrs, _raw(value))


def _select(ident: str, options: Sequence[tuple[str, str, bool]], *, extra: str = "") -> str:
    """A ``<select>`` from ``(value, label, selected)`` triples."""
    body = "".join(
        '<option{}{}>{}</option>'.format(
            ' selected' if selected else "", _attr("value", value), _esc(label)
        )
        for value, label, selected in options
    )
    return '<select{}{}>{}</select>'.format(_attr("id", ident), extra, body)


def _pct_of_bar(total: Any, budget: Any) -> tuple[float, str]:
    """``(percent, tone)`` for a card's planned residency against its budget."""
    try:
        total_f, budget_f = float(total), float(budget)
    except (TypeError, ValueError):
        return 0.0, "amber"
    if budget_f <= 0:
        return 0.0, "amber"
    ratio = total_f / budget_f
    if ratio > 1.0:
        return 100.0, "red"
    return max(0.0, min(100.0, ratio * 100.0)), "green" if ratio < 0.9 else "amber"


def _render_per_card(per_card: Any) -> str:
    """The planned per-card residency table of one tier (§B3 ``per_card``)."""
    rows = [c for c in (per_card or []) if isinstance(c, Mapping)]
    if not rows:
        return ""
    head = (
        "<thead><tr><th>card</th><th>weights</th><th>kv</th><th>total</th>"
        "<th>budget</th><th>fit</th></tr></thead>"
    )
    body = []
    for card in rows:
        pct, tone = _pct_of_bar(card.get("total"), card.get("budget"))
        body.append(
            '<tr{}>'
            "<td>{card}</td><td>{weights}</td><td>{kv}</td><td>{total}</td><td>{budget}</td>"
            '<td><span class="stk-bar"><i class="{tone}" style="width:{pct:.1f}%"></i></span></td>'
            "</tr>".format(
                _attr("data-card", card.get("card"))
                + _attr("data-weights", card.get("weights"))
                + _attr("data-kv", card.get("kv"))
                + _attr("data-total", card.get("total"))
                + _attr("data-budget", card.get("budget")),
                card=_esc(card.get("card")),
                weights=_gb(card.get("weights")),
                kv=_gb(card.get("kv")),
                total=_gb(card.get("total")),
                budget=_gb(card.get("budget")),
                tone=tone,
                pct=pct,
            )
        )
    return '<table class="stk-cards">{}<tbody>{}</tbody></table>'.format(
        head, "".join(body)
    )


def _render_tier_row(tier: Mapping[str, Any]) -> str:
    """One tier of the residency strip — every ``TierRuntime`` field mirrored."""
    resident = bool(tier.get("resident"))
    model = tier.get("model") or _DASH
    return (
        '<div class="stk-tier"{}>'
        '<div class="stk-tier-head">'
        '<span class="stk-role">{}</span>'
        '<span class="stk-tier-model">{}</span>'
        '<span class="stk-tier-key">{}</span>'
        '<span class="stk-pill {}">{}</span>'
        "</div>"
        '<div class="stk-metrics">'
        '<span class="stk-metric"><b>port</b><span class="stk-num">{}</span></span>'
        '<span class="stk-metric"><b>ctx</b><span class="stk-num">{}</span></span>'
        '<span class="stk-metric"><b>vram</b><span class="stk-num">{}</span></span>'
        '<span class="stk-metric"><b>tok/s</b><span class="stk-num">{}</span></span>'
        '<span class="stk-metric"><b>backend</b><span class="stk-num">{}</span></span>'
        "</div>"
        "{}"
        "</div>".format(
            _attr("data-tier", json.dumps(tier, default=str))
            + _attr("data-role", tier.get("role"))
            + _attr("data-key", tier.get("key"))
            + _attr("data-port", tier.get("port"))
            + _attr("data-ctx", tier.get("ctx"))
            + _attr("data-model", tier.get("model"))
            + _attr("data-backend", tier.get("backend"))
            + _attr("data-resident", resident)
            + _attr("data-vram_gb", tier.get("vram_gb"))
            + _attr("data-tok_s", tier.get("tok_s")),
            _esc(tier.get("role") or _DASH),
            _esc(model),
            _esc(tier.get("key") or ""),
            "ok" if resident else "bad",
            "resident" if resident else "not resident",
            _int(tier.get("port")),
            _int(tier.get("ctx")),
            _gb(tier.get("vram_gb")),
            _rate(tier.get("tok_s")),
            _esc(tier.get("backend") or _DASH),
            _render_per_card(tier.get("per_card")),
        )
    )


def css() -> str:
    """Contents of the sibling ``stack_tab.css`` (re-read per call)."""
    return CSS


def render_residency_strip(status: Mapping[str, Any]) -> str:
    """The per-tier residency strip for one ``/v1/stacks/status`` payload.

    ``status`` is the ``StackManager.status()`` shape; an empty/``None``-ish
    payload renders the unapplied state (no tiers).  Never raises on a missing
    key.
    """
    payload: Mapping[str, Any] = status if isinstance(status, Mapping) else {}
    tiers = [t for t in (payload.get("tiers") or []) if isinstance(t, Mapping)]
    stack = payload.get("stack")
    applied = bool(tiers)
    tone = "ok" if applied and payload.get("ok") is not False else "bad"
    label = "applied" if applied else "not applied"
    if applied and payload.get("ok") is False:
        label = "applied · degraded"

    body = (
        "".join(_render_tier_row(tier) for tier in tiers)
        or '<p class="stk-empty">No stack applied — build one and hit Apply.</p>'
    )
    warnings = [w for w in (payload.get("warnings") or []) if w not in (None, "")]
    warn_html = (
        '<ul class="stk-warnings">{}</ul>'.format(
            "".join("<li>{}</li>".format(_esc(w)) for w in warnings)
        )
        if warnings
        else ""
    )
    return (
        '<div class="stk-residency"{}>'
        '<div class="stk-res-head">'
        '<span class="stk-res-stack">{}</span>'
        '<span class="stk-pill {}">{}</span>'
        "</div>"
        '<div class="stk-tiers">{}</div>'
        "{}"
        "</div>".format(
            _attr("data-stack-status", json.dumps(payload, default=str)),
            _esc(stack or "no stack"),
            tone,
            _esc(label),
            body,
            warn_html,
        )
    )


def _hardware_note(hardware: Optional[Mapping[str, Any]]) -> str:
    """The VRAM line under an Auto control, from the ``_hardware_summary`` shape."""
    if not isinstance(hardware, Mapping):
        return (
            '<div class="stk-note">VRAM budget unknown — the shell passes the '
            "hardware summary at mount.</div>"
        )
    cards = [d for d in (hardware.get("devices") or []) if isinstance(d, Mapping)]
    budget = hardware.get("available_gb")
    if budget is None:
        budget = hardware.get("vram_free_gb")
    if budget is None:
        budget = hardware.get("vram_gb")
    try:
        total = "{:.2f} GiB".format(float(budget))
    except (TypeError, ValueError):
        total = _DASH
    return '<div class="stk-note">VRAM budget {} · {} card(s) · source {}</div>'.format(
        _esc(total), _esc(len(cards)), _esc(hardware.get("vram_source") or "unknown")
    )


def _model_options(models: Sequence[Mapping[str, Any]]) -> list[tuple[str, str, bool]]:
    """``(value, label, selected)`` triples for the local-GGUF picker."""
    options: list[tuple[str, str, bool]] = []
    for index, entry in enumerate(models):
        if not isinstance(entry, Mapping):
            continue
        value = entry.get("file") or entry.get("name") or ""
        name = entry.get("name") or value
        size = entry.get("size_gb")
        label = "{} — {}".format(name, _gb(size)) if size is not None else str(name)
        options.append((str(value), label, index == 0))
    return options


def render_auto_preset(
    role: str,
    *,
    models: Sequence[Mapping[str, Any]] = (),
    hardware: Optional[Mapping[str, Any]] = None,
) -> str:
    """The per-tier **Auto** preset control.

    ``role`` is a tier role (LOCAL-STACKS §2); ``models`` is one page of
    ``models_manager.list_local()`` rows; ``hardware`` is the hardware summary
    used by the preset math.  Renders a control that fills a tier's ``ctx`` /
    ``ngl`` from the chosen local GGUF.
    """
    models = [m for m in (models or []) if isinstance(m, Mapping)]
    options = _model_options(models)
    slug = _slug(role)
    picker = _select("stk-{}-model".format(slug), options) if options else ""
    ctx = _number_input("stk-{}-ctx".format(slug), _DEFAULT_CTX, low=2048, step=1024)
    ngl = _number_input("stk-{}-ngl".format(slug), _DEFAULT_NGL, low=0, step=1)
    return (
        '<div class="stk-auto"{role}>'
        '<div class="stk-auto-head">'
        '<span class="stk-role">{label}</span><span class="stk-role-hint">{hint}</span>'
        "</div>"
        '<div class="stk-row">{model} {ctx} {ngl} {auto} {msg}</div>'
        "{note}"
        "</div>".format(
            role=_attr("data-role", role) + _attr("data-hardware", json.dumps(hardware, default=str)),
            label=_esc(role),
            hint=_esc(ROLE_HINTS.get(str(role), "")),
            model=_field("model", picker or '<span class="stk-note">no local GGUFs</span>'),
            ctx=_field("ctx", ctx),
            ngl=_field("ngl", ngl),
            auto='<button type="button"{} class="stk-btn-auto">Auto</button>'.format(
                _attr("id", "stk-{}-auto".format(slug)) + _attr("data-action", "stack-auto")
            ),
            msg='<span class="stk-msg"{}></span>'.format(
                _attr("id", "stk-{}-msg".format(slug))
            ),
            note=_hardware_note(hardware),
        )
    )


def _render_tier_advanced(role: str) -> str:
    """The §B3 launch-config fields the Auto control does not own."""
    slug = _slug(role)
    caches = [(c, c, c == "q8_0") for c in _CACHE_TYPES]
    backends = [(b, b, b == "vulkan") for b in ("vulkan", "cuda", "hip", "cpu")]

    def text_field(label: str, suffix: str, placeholder: str) -> str:
        return _field(
            label,
            '<input{} type="text" placeholder="{}">'.format(
                _attr("id", "stk-{}-{}".format(slug, suffix)), placeholder
            ),
        )

    fields = [
        _field("cache_k", _select("stk-{}-cache-k".format(slug), caches)),
        _field("cache_v", _select("stk-{}-cache-v".format(slug), caches)),
        _field("backend", _select("stk-{}-backend".format(slug), backends)),
        text_field("pin", "pin", "HIP_VISIBLE_DEVICES=1"),
        text_field("ts", "ts", "1,1"),
        text_field("mmproj", "mmproj", "mmproj-F16.gguf"),
        _field("spec n_max", _number_input("stk-{}-spec-n".format(slug), 3, low=0, step=1)),
    ]
    return (
        '<details class="stk-advanced"><summary>launch config — {}</summary>'
        '<div class="stk-row">{}</div>'
        "</details>".format(_esc(role), "".join(fields))
    )


def _render_saved(stacks: Sequence[Mapping[str, Any]]) -> str:
    """The saved-stack list (``schema.list_stacks()`` summaries)."""
    rows = [s for s in (stacks or []) if isinstance(s, Mapping)]
    if not rows:
        return '<p class="stk-empty">No saved stacks yet.</p>'
    items = []
    for row in rows:
        name = row.get("name") or ""
        roles = [r for r in (row.get("roles") or []) if r]
        tier_count = _count(row.get("tiers")) or len(roles)
        summary = " · ".join(str(r) for r in roles) or "{} tier(s)".format(tier_count)
        items.append(
            '<div class="stk-saved-row"{row}>'
            '<span class="stk-name">{name}</span>'
            "<span class=\"stk-roles\">{summary} · {count} tier(s)</span>"
            '<button type="button"{load_attrs} class="stk-btn-load">Load</button>'
            '<button type="button"{del_attrs} class="stk-btn-unload">Delete</button>'
            "</div>".format(
                row=_attr("data-stack", name),
                name=_esc(name),
                summary=_esc(summary),
                count=tier_count,
                load_attrs=_attr("data-action", "stack-load") + _attr("data-stack", name),
                del_attrs=_attr("data-action", "stack-delete") + _attr("data-stack", name),
            )
        )
    return '<div class="stk-saved">{}</div>'.format("".join(items))


def render_stack_tab(
    *,
    stacks: Sequence[Mapping[str, Any]] = (),
    models: Sequence[Mapping[str, Any]] = (),
    status: Optional[Mapping[str, Any]] = None,
) -> str:
    """The whole Stack tab: saved-stack builder + Auto preset + residency strip.

    ``stacks`` are ``schema.list_stacks()`` summaries; ``models`` are local
    library rows for the tier pickers; ``status`` is the live
    ``/v1/stacks/status`` payload.
    """
    tiers = "".join(
        '{}{}'.format(
            render_auto_preset(role, models=models),
            _render_tier_advanced(role),
        )
        for role in ROLES
    )
    name_input = '<input{} type="text" placeholder="peer-2tier">'.format(
        _attr("id", "stk-name")
    )
    return (
        '<div class="tabpane stk-tab"{}>'
        "<section>"
        '<div class="stk-head"><h2>Stacks</h2>'
        '<span class="stk-note">roles, not sizes — one local GGUF per tier; face loads first</span>'
        "</div>"
        "{}"
        '<div class="stk-row">{} {}</div>'
        '<h3 style="margin:.7rem 0 .3rem; font-size:.95rem">Tiers</h3>'
        '<div class="stk-builder">{}</div>'
        '<div class="stk-actions">'
        '<button type="button"{} class="stk-btn-validate">Validate</button>'
        '<button type="button"{} class="stk-btn-apply">Apply</button>'
        '<button type="button"{} class="stk-btn-unload">Unload</button>'
        '<span class="stk-msg"{}></span>'
        "</div>"
        "</section>"
        "<section>"
        '<div class="stk-head"><h2 style="margin-top:0">Residency</h2>'
        '<span class="stk-note">GET /v1/stacks/status — per tier: port, ctx, VRAM, tok/s</span>'
        "</div>"
        "{}"
        "</section>"
        "</div>".format(
            _attr("id", "tab-stacks") + _attr("data-fragment", "stack_tab")
            + _attr("data-css-name", CSS_NAME),
            _render_saved(stacks),
            _field("stack name", name_input),
            _field(
                "repo",
                '<input{} type="text" placeholder="unsloth/Qwen3.8-27B-GGUF">'.format(
                    _attr("id", "stk-repo")
                ),
            ),
            tiers,
            _attr("data-action", "stack-validate"),
            _attr("data-action", "stack-apply"),
            _attr("data-action", "stack-unload"),
            _attr("id", "stk-msg-main"),
            render_residency_strip(status if isinstance(status, Mapping) else {}),
        )
    )
