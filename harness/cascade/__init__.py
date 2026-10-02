"""Cascade pilot harness — the deterministic layer of the role cascade.

The pilot brief is ``CASCADE-PILOT.md`` (research workspace, steer-exp): a
consumer-hardware serving system that assigns each request to the cheapest
model good at it and spreads work over every compute unit on the box.  It is
a mixture of *models by role*, not a fused model.

The dividing line (brief §"Harness vs model vs training") is:

- **harness (here)** — deterministic policy: role taxonomy, model registry,
  path definitions P0–P6, scheduling, async judgement plumbing + cancellation,
  batching, escalation thresholds, telemetry.
- **model** — learned judgment: routers, decision models, generators, drafters,
  encoders.
- **offline** — measurement and training: the per-role oracle, calibration,
  autotune.

``harness.cascade`` is a skeleton: interfaces and deterministic behaviour with
models stubbed.  Measured quality/latency land in
:class:`harness.cascade.oracle.CandidateMeasurement` records and feed the
per-role frontiers; nothing here loads a model or spawns a server — the
existing :mod:`harness.stack` layer owns launches once a resident set is
chosen.
"""

from harness.cascade.judgement import (
    Judgement,
    JudgementBroker,
    JudgementResult,
    JudgementState,
    result_for,
)
from harness.cascade.oracle import CandidateMeasurement, frontier_by_role, pareto_frontier
from harness.cascade.paths import (
    ESCALATION_TRIGGERS,
    PATHS,
    EscalationTrigger,
    Path,
    PathStep,
    ROOT_PATHS,
    get_path,
    paths_for_bucket,
    resolve_steps,
)
from harness.cascade.policy import Policy
from harness.cascade.registry import (
    DEFAULT_CANDIDATES,
    Candidate,
    CandidateRegistry,
    ResidentSet,
    default_registry,
    select_resident_set,
)
from harness.cascade.roles import (
    ROLES,
    Device,
    Kind,
    Role,
    get_role,
    roles_for_kind,
    roles_on,
)
from harness.cascade.scheduler import (
    LatencyModel,
    Plan,
    RoleCost,
    RouteDecision,
    StageEstimate,
    plan_request,
)
from harness.cascade.telemetry import RoleRecord, TelemetryLog

__all__ = [
    # roles
    "Device",
    "Kind",
    "Role",
    "ROLES",
    "get_role",
    "roles_for_kind",
    "roles_on",
    # registry
    "Candidate",
    "CandidateRegistry",
    "DEFAULT_CANDIDATES",
    "ResidentSet",
    "default_registry",
    "select_resident_set",
    # paths
    "EscalationTrigger",
    "ESCALATION_TRIGGERS",
    "Path",
    "PathStep",
    "PATHS",
    "ROOT_PATHS",
    "get_path",
    "paths_for_bucket",
    "resolve_steps",
    # policy
    "Policy",
    # judgements
    "Judgement",
    "JudgementBroker",
    "JudgementResult",
    "JudgementState",
    "result_for",
    # scheduling
    "LatencyModel",
    "RoleCost",
    "RouteDecision",
    "StageEstimate",
    "Plan",
    "plan_request",
    # telemetry
    "RoleRecord",
    "TelemetryLog",
    # oracle
    "CandidateMeasurement",
    "pareto_frontier",
    "frontier_by_role",
]
