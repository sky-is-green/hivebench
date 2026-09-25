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

from typing import Any, Mapping, Optional, Sequence

#: Sibling stylesheet read by :func:`css`.
CSS_NAME = "stack_tab.css"


def css() -> str:
    """Contents of the sibling ``stack_tab.css`` (re-read per call)."""
    raise NotImplementedError


def render_residency_strip(status: Mapping[str, Any]) -> str:
    """The per-tier residency strip for one ``/v1/stacks/status`` payload.

    ``status`` is the ``StackManager.status()`` shape; an empty/``None``-ish
    payload renders the unapplied state (no tiers).  Never raises on a missing
    key.
    """
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError
