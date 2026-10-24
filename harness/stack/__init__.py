"""``harness.stack`` — Local Stack Control (LSC) public surface.

The skeletons in this package are the wire contract for T35–T43 (ADR-L8).
Consumers import from here, not from the submodules::

    from harness.stack import Stack, Tier, load_stack, save_stack
    from harness.stack import plan_residency, ResidencyPlan
    from harness.stack import StackManager, create_router

Modules: :mod:`schema` (data model + file I/O + shape validation),
:mod:`residency` (per-card plan), :mod:`manager` (apply/status/unload),
:mod:`api` (router factory).
"""

from harness.stack.api import PREFIX, create_router
from harness.stack.deepinfer import DeepInferLauncher
from harness.stack.endpoint import endpoint_base, endpoint_port, probe_endpoint
from harness.stack.manager import (
    StackManager,
    TierRuntime,
    tier_env,
    tier_extra_args,
    tier_load_options,
)
from harness.stack.residency import CardPlan, ResidencyPlan, plan_residency
from harness.stack.schema import (
    DEFAULT_ROUTING,
    LAUNCH_ARGS_DENIED,
    LAUNCH_KINDS,
    ROLE_AGENCY,
    ROLE_FACE,
    ROLE_MECHANICS,
    ROLE_WORKER,
    ROLES,
    STACK_VERSION,
    STACKS_DIRNAME,
    Stack,
    Tier,
    delete_stack,
    list_stacks,
    load_stack,
    save_stack,
    stack_path,
    stacks_dir,
    validate_launch,
    validate_shape,
)
from harness.stack.strata import StrataLauncher, merge_engine_args

__all__ = [
    # schema
    "DEFAULT_ROUTING",
    "LAUNCH_ARGS_DENIED",
    "LAUNCH_KINDS",
    "ROLE_AGENCY",
    "ROLE_FACE",
    "ROLE_MECHANICS",
    "ROLE_WORKER",
    "ROLES",
    "STACK_VERSION",
    "STACKS_DIRNAME",
    "Stack",
    "Tier",
    "delete_stack",
    "list_stacks",
    "load_stack",
    "save_stack",
    "stack_path",
    "stacks_dir",
    "validate_launch",
    "validate_shape",
    # residency
    "CardPlan",
    "ResidencyPlan",
    "plan_residency",
    # manager / endpoint helper
    "StackManager",
    "TierRuntime",
    "endpoint_base",
    "endpoint_port",
    "probe_endpoint",
    "tier_env",
    "tier_extra_args",
    "tier_load_options",
    # strata launcher
    "StrataLauncher",
    "merge_engine_args",
    # deepinfer launcher
    "DeepInferLauncher",
    # api
    "PREFIX",
    "create_router",
]
