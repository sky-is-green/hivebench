"""HiveBench Studio harness — the FastAPI sidecar over the strata research stack.

Exposes the Strata context-curation pipeline and the HiveBench measurement layer
as a local HTTP service (default 127.0.0.1:8765) for the dsh shell (M2) and the
web UI. See HARNESS-SPEC.md §3.3 for the endpoint contract.

Run::

    python -m harness                 # live (LM Studio / provider backends)
    python -m harness --mock          # offline (fake drone + mock backend)
"""

from harness.app import create_app

__all__ = ["create_app"]
