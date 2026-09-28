"""Make the suite (`experiments/`, `testing/`, `tests/`), sidecar (`harness/`)
and vendored dsh SDK (`vendor/deepseek_harness`) importable, plus the system
under test from the sibling splinter-memory checkout, so flat names like
`cortex`, `sieve`, `experiments`, `harness` resolve regardless of how pytest
is invoked (editable install makes this unnecessary, but from-source runs
stay supported).

The sibling checkout is resolved in this order (see `resolve_splinter_home`):

1. an explicit ``$SPLINTER_HOME`` (legacy ``$STRATA_HOME`` still accepted);
2. walking up from ``ROOT`` for a ``splinter-memory`` / ``strata-memory`` dir;
3. the primary checkout's sibling, recovered from the git worktree marker.

HIVE-OPS worktrees live at ``<project>/worktrees/<repo>/hivebench-<Task_ID>``,
so ``ROOT/../splinter-memory`` does not exist from a worktree. Rather than
silently run against that wrong tree, resolution fails loudly and prints the
paths it searched.
"""

import importlib
import importlib.util
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

# Sibling checkout names in preference order: current, then pre-rename.
SIBLING_NAMES = ("splinter-memory", "strata-memory")


class SplinterHomeNotFound(RuntimeError):
    """Raised when no sibling splinter/strata checkout can be located."""


def _is_dir(path):
    """Filesystem probe, split out so tests can force the not-found path."""
    return os.path.isdir(path)


def _primary_checkout(root):
    """Return the primary checkout root when `root` is a linked git worktree.

    In a linked worktree ``.git`` is a file ``gitdir: <primary>/.git/worktrees/<n>``;
    two levels above that gitdir is the primary checkout. Returns ``None`` for a
    normal (or non-git) checkout.
    """
    marker = os.path.join(root, ".git")
    if not os.path.isfile(marker):
        return None
    try:
        with open(marker, encoding="utf-8") as fh:
            text = fh.read().strip()
    except OSError:
        return None
    if not text.startswith("gitdir:"):
        return None
    gitdir = text.split(":", 1)[1].strip()
    if not os.path.isabs(gitdir):
        gitdir = os.path.normpath(os.path.join(root, gitdir))
    # <primary>/.git/worktrees/<name> -> <primary>
    common = os.path.normpath(os.path.join(gitdir, "..", ".."))
    primary = os.path.dirname(common)
    return primary or None


def _walk_up_candidates(root):
    """Yield ``<ancestor>/<sibling>`` for root and each ancestor, nearest first."""
    cur = os.path.abspath(root)
    while True:
        for name in SIBLING_NAMES:
            yield os.path.normpath(os.path.join(cur, name))
        parent = os.path.dirname(cur)
        if parent == cur:
            return
        cur = parent


def resolve_splinter_home(root, env=None):
    """Resolve the sibling splinter-memory checkout for `root`.

    Order: ``$SPLINTER_HOME``/``$STRATA_HOME`` -> walk up from `root` -> the
    primary checkout's sibling. Raises `SplinterHomeNotFound` (printing the
    path it resolved to) instead of returning a directory that does not exist.
    """
    env = os.environ if env is None else env

    # 1) Explicit override — honor it, but never silently accept a bad path.
    for var in ("SPLINTER_HOME", "STRATA_HOME"):
        override = env.get(var)
        if override:
            resolved = os.path.normpath(os.path.abspath(os.path.expanduser(override)))
            if _is_dir(resolved):
                return resolved
            raise SplinterHomeNotFound(
                f"${var} is set to {resolved!r}, but that directory does not "
                "exist. Point it at a splinter-memory/strata-memory checkout "
                "or unset it."
            )

    # 2) Walk up from ROOT (covers the worktree layout via a shared parent).
    for candidate in _walk_up_candidates(root):
        if _is_dir(candidate):
            return candidate

    # 3) Primary checkout's sibling (when the worktree is not under the primary).
    primary = _primary_checkout(root)
    if primary:
        for name in SIBLING_NAMES:
            candidate = os.path.normpath(os.path.join(os.path.dirname(primary), name))
            if _is_dir(candidate):
                return candidate

    # 4) Fail loudly, printing every path we resolved/tried.
    searched = list(_walk_up_candidates(root))
    fallback = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(root)), SIBLING_NAMES[0])
    )
    tried = "\n".join(f"    - {p}" for p in searched)
    raise SplinterHomeNotFound(
        "Could not locate the sibling memory checkout "
        f"({' / '.join(SIBLING_NAMES)}).\n"
        f"  worktree root     : {os.path.abspath(root)}\n"
        f"  primary checkout  : {primary or 'unknown'}\n"
        f"  SPLINTER_HOME     : {env.get('SPLINTER_HOME', '<unset>')}\n"
        f"  paths searched    :\n{tried}\n"
        f"  fallback path     : {fallback}\n"
        "Set SPLINTER_HOME=/path/to/splinter-memory (or STRATA_HOME) and re-run."
    )


def resolve_package_name(home):
    """The sibling's python package dir: current ``splinter`` or legacy ``strata``.

    The two layouts are mutually exclusive in practice; the current name is the
    default so a checkout missing both surfaces as the usual ImportError rather
    than a resolution error.
    """
    for name in ("splinter", "strata"):
        if _is_dir(os.path.join(home, name)):
            return name
    return "splinter"


SPLINTER = resolve_splinter_home(ROOT)

# The sibling was renamed `strata` -> `splinter`.  Whichever package directory
# the resolved checkout ships, expose it under both names for the process
# lifetime: legacy `strata` consumers and current `splinter` consumers hit the
# same module objects (T23).  `SPLINTER/<pkg>` is also what makes the
# checkout's flat names (cortex, sieve, retention, ...) importable.
PACKAGE_NAME = resolve_package_name(SPLINTER)
LEGACY_PACKAGE_NAME = "strata" if PACKAGE_NAME == "splinter" else "splinter"


class _AliasLoader:
    """Loader seam: hand the import system the already-imported real module."""

    def __init__(self, module):
        self.module = module

    def create_module(self, spec):
        return self.module

    def exec_module(self, module):
        pass


class LegacyAliasFinder:
    """Resolve ``alias[.sub]`` imports to the ``real[.sub]`` modules.

    The alias imports the real module first and returns the *same* object, so
    no module is ever executed twice (duplicating singletons like the splinter
    conversation store).  It is inserted at the front of ``sys.meta_path`` so
    the remap also covers package internals, while a real on-disk package
    still wins for its own name (the alias names a package that does not
    exist on disk).
    """

    def __init__(self, alias, real):
        self.alias = alias
        self.real = real

    def find_spec(self, fullname, path=None, target=None):
        if fullname != self.alias and not fullname.startswith(self.alias + "."):
            return None
        real_name = self.real + fullname[len(self.alias):]
        module = importlib.import_module(real_name)
        sys.modules[fullname] = module
        return importlib.util.spec_from_loader(
            fullname, _AliasLoader(module), is_package=hasattr(module, "__path__")
        )

# Export the resolved checkout so every downstream consumer in the process
# (e.g. tests/run_hive_tests.py, subprocess test runners) agrees on one path
# instead of re-deriving a stale ROOT/../splinter-memory from a worktree.
os.environ.setdefault("SPLINTER_HOME", SPLINTER)

# experiments/, testing/, tests/ and the sidecar package harness/ live at the
# repo root (the old double-nested harness/harness/ was flattened 2026-09-12).
sys.path.insert(0, ROOT)
# The vendored dsh Python SDK (deepseek_harness) — the agent bridge imports it.
sys.path.insert(0, os.path.join(ROOT, "vendor"))
# The checkout root (so `import splinter` works) and the package root (so flat
# names like `cortex`, `sieve` work).  The package dir is current (`splinter`)
# or legacy (`strata`) depending on what the checkout ships.
sys.path.insert(0, SPLINTER)
sys.path.insert(0, os.path.join(SPLINTER, PACKAGE_NAME))

# Legacy `strata.*` imports resolve to the `splinter.*` modules (and vice
# versa on a pre-rename checkout) — see LegacyAliasFinder.  The finder runs
# before PathFinder: package internals use absolute `splinter.*` imports, so on
# a legacy tree they must also be remapped or the package would be loaded
# twice under two names.
if PACKAGE_NAME == "splinter":
    # Current layout: the alias stays lazy; nothing imports the old name.
    sys.meta_path.insert(0, LegacyAliasFinder("strata", "splinter"))
else:
    # Pre-rename checkout: hivebench imports the new name, and the package's
    # own modules use it internally too.  Load the real package once under its
    # real name, then alias `splinter` onto that exact object.
    _real_package = importlib.import_module(PACKAGE_NAME)
    sys.modules.setdefault(LEGACY_PACKAGE_NAME, _real_package)
    sys.meta_path.insert(0, LegacyAliasFinder(LEGACY_PACKAGE_NAME, PACKAGE_NAME))
