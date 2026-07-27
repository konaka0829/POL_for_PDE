"""Import-safe authoritative Paper 1 protocol identities.

No scientific module should duplicate these strings.  A protocol is part of
the compute identity, not merely presentation metadata.
"""

E0_SCHEMA_VERSION = "paper1-e0-v3"
E0_REQUIRED_CHECKS = frozenset(
    {
        "resampling",
        "fourier_projector",
        "reference_spatial_convergence",
        "reference_temporal_convergence",
        "reference_joint_convergence",
        "finite_data_interface",
        "no_high_frequency_leak",
        "target_coefficient_consistency",
        "model1_full_observation_identity",
        "model1_bandlimited_reduced_j",
        "model1_aliasing_counterexample",
    }
)
MASTER_DATASET_SCHEMA_VERSION = "paper1-master-dataset-v1"
E1_SCHEMA_VERSION = "paper1-e1-v2"
E2_SCHEMA_VERSION = "paper1-e2-v3"

RUN_PLAN_SCHEMA_VERSION = "paper1-run-plan-v2"
RUN_MANIFEST_SCHEMA_VERSION = "paper1-run-manifest-v2"
RUN_ARTIFACT_MANIFEST_SCHEMA_VERSION = "paper1-run-artifact-manifest-v1"

MATRIX_SCHEMA_VERSION = "paper1-matrix-spec-v1"
E1_MATRIX_PLUGIN_PROTOCOL = "paper1-e1-resolution-matrix-v3"

BASELINE_SCHEMA_VERSION = "paper1-phase1-scientific-baseline-v4"
MATRIX_BASELINE_SCHEMA_VERSION = "paper1-e1-matrix-smoke-baseline-v3"
# Schema versions describe record shape.  The comparison policy describes
# field taxonomy/tolerances, while the generator version identifies the
# artifact-only extraction algorithm.  They evolve independently.
COMPARISON_POLICY_VERSION = "paper1-scientific-comparison-v2"
BASELINE_GENERATOR_VERSION = "paper1-scientific-baseline-generator-v4"


def recipe_protocols(kind: str) -> tuple[str, ...]:
    """Return the ordered protocol dependency chain for a scalar run."""
    protocols = {
        "e0": (E0_SCHEMA_VERSION,),
        "e1": (E0_SCHEMA_VERSION, E1_SCHEMA_VERSION),
        "e2": (
            E0_SCHEMA_VERSION,
            MASTER_DATASET_SCHEMA_VERSION,
            E2_SCHEMA_VERSION,
        ),
    }
    try:
        return protocols[kind]
    except KeyError as exc:
        raise ValueError(f"unknown Paper 1 experiment kind: {kind}") from exc
