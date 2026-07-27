"""Field-aware cross-runtime comparison for saved scientific baselines."""
from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Mapping


COMPARISON_POLICY_VERSION = "paper1-scientific-comparison-v2"
_CROSS_RUNTIME_IGNORED_FIELDS = frozenset(
    {"selected_models_content_hash", "state_key"}
)


@dataclass(frozen=True)
class NumericTolerance:
    """Relative and absolute tolerance for one numeric category."""

    rtol: float
    atol: float


@dataclass(frozen=True)
class ScientificComparisonPolicy:
    """Explicit numeric category assignment for every expected float path."""

    version: str
    numeric_paths: Mapping[str, str]
    tolerances: Mapping[str, NumericTolerance]


@dataclass(frozen=True)
class ScientificDifference:
    """One readable structural, exact, or numeric mismatch."""

    path: str
    expected: Any
    actual: Any
    reason: str


DEFAULT_TOLERANCES = {
    # These remain well below the experiment acceptance thresholds.  Relative
    # tolerance absorbs BLAS/FFT implementation drift away from zero.
    "scientific_float64": NumericTolerance(rtol=1e-7, atol=1e-12),
    "scientific_float32": NumericTolerance(rtol=2e-6, atol=2e-7),
    # Roundoff diagnostics are judged by dtype-scale absolute error because a
    # relative tolerance is meaningless near theoretical zero.
    "roundoff_float64": NumericTolerance(rtol=0.0, atol=5e-11),
    "roundoff_float32": NumericTolerance(rtol=0.0, atol=2e-6),
    # Selection identities are exact elsewhere; the metrics which support the
    # choice use a tighter rule to expose ranking-boundary changes.
    "selection_metric": NumericTolerance(rtol=1e-8, atol=1e-12),
    # Random draws are intentionally not exact across PyTorch versions.  Only
    # saved aggregate metrics enter the portable record; identities, seeds,
    # shapes, and selected candidates remain exact structural fields.
    "stochastic_aggregate": NumericTolerance(rtol=5e-5, atol=2e-7),
    # Raw response-matrix/eigenspectrum diagnostics are useful regression
    # evidence but can amplify small LAPACK differences in ill-conditioned
    # directions. Discrete rank/selection/stability identities remain exact.
    "condition_sensitive": NumericTolerance(rtol=2e-5, atol=5e-10),
}


def normalized_numeric_path(path: str) -> str:
    """Collapse table/list indices while retaining every named field."""
    return re.sub(r"\[\d+\]", "[*]", path)


def _path(parent: str, component: object) -> str:
    if isinstance(component, int):
        return f"{parent}[{component}]"
    return f"{parent}.{component}" if parent != "$" else f"$.{component}"


def compare_scientific_record(
    actual: Any,
    expected: Any,
    policy: ScientificComparisonPolicy,
) -> list[ScientificDifference]:
    """Return path-aware differences without a permissive numeric fallback."""
    differences: list[ScientificDifference] = []

    def compare(left: Any, right: Any, path: str) -> None:
        if isinstance(right, bool) or right is None or isinstance(right, str):
            if type(left) is not type(right) or left != right:
                differences.append(
                    ScientificDifference(path, right, left, "exact mismatch")
                )
            return
        if isinstance(right, int) and not isinstance(right, bool):
            if type(left) is not int or left != right:
                differences.append(
                    ScientificDifference(path, right, left, "integer mismatch")
                )
            return
        if isinstance(right, float):
            if not isinstance(left, (int, float)) or isinstance(left, bool):
                differences.append(
                    ScientificDifference(path, right, left, "numeric type mismatch")
                )
                return
            if not math.isfinite(right) or not math.isfinite(float(left)):
                differences.append(
                    ScientificDifference(path, right, left, "non-finite numeric")
                )
                return
            category = policy.numeric_paths.get(path)
            if category is None:
                category = policy.numeric_paths.get(normalized_numeric_path(path))
            if category is None:
                differences.append(
                    ScientificDifference(
                        path, right, left, "numeric path absent from policy"
                    )
                )
                return
            if category == "exact_numeric":
                if float(left) != right:
                    differences.append(
                        ScientificDifference(
                            path, right, left, "exact numeric mismatch"
                        )
                    )
                return
            tolerance = policy.tolerances.get(category)
            if tolerance is None:
                differences.append(
                    ScientificDifference(
                        path, right, left, f"unknown numeric category {category}"
                    )
                )
                return
            error = abs(float(left) - right)
            limit = tolerance.atol + tolerance.rtol * abs(right)
            if error > limit:
                differences.append(
                    ScientificDifference(
                        path,
                        right,
                        left,
                        f"{category} error {error:.6g} exceeds {limit:.6g}",
                    )
                )
            return
        if isinstance(right, dict):
            if not isinstance(left, dict):
                differences.append(
                    ScientificDifference(path, right, left, "object type mismatch")
                )
                return
            left_keys = set(left) - _CROSS_RUNTIME_IGNORED_FIELDS
            right_keys = set(right) - _CROSS_RUNTIME_IGNORED_FIELDS
            if left_keys != right_keys:
                differences.append(
                    ScientificDifference(
                        path,
                        sorted(right_keys),
                        sorted(left_keys),
                        "object key mismatch",
                    )
                )
                return
            for key in sorted(right_keys):
                compare(left[key], right[key], _path(path, key))
            return
        if isinstance(right, list):
            if not isinstance(left, list) or len(left) != len(right):
                differences.append(
                    ScientificDifference(
                        path,
                        len(right),
                        len(left) if isinstance(left, list) else left,
                        "array length/type mismatch",
                    )
                )
                return
            for index, item in enumerate(right):
                compare(left[index], item, _path(path, index))
            return
        differences.append(
            ScientificDifference(path, type(right).__name__, type(left).__name__, "unsupported expected type")
        )

    compare(actual, expected, "$")
    return differences


def assert_scientific_record_matches(
    actual: Any,
    expected: Any,
    policy: ScientificComparisonPolicy,
    *,
    limit: int = 12,
) -> None:
    """Raise an assertion with the first readable scientific differences."""
    differences = compare_scientific_record(actual, expected, policy)
    if not differences:
        return
    lines = [
        f"{item.path}: {item.reason}; expected={item.expected!r}, actual={item.actual!r}"
        for item in differences[:limit]
    ]
    remaining = len(differences) - len(lines)
    if remaining:
        lines.append(f"... and {remaining} more difference(s)")
    raise AssertionError("scientific baseline mismatch:\n" + "\n".join(lines))


def policy_from_baseline(
    baseline: Mapping[str, Any], *, section: str | None = None
) -> ScientificComparisonPolicy:
    """Load the explicit policy embedded in a v3/v2 scientific fixture."""
    raw = baseline.get("comparison_policy")
    if not isinstance(raw, dict):
        raise ValueError("scientific baseline comparison_policy is missing")
    numeric_paths = raw.get("numeric_paths")
    if not isinstance(numeric_paths, dict):
        raise ValueError("scientific baseline numeric_paths is missing")
    tolerances = {
        name: NumericTolerance(float(value["rtol"]), float(value["atol"]))
        for name, value in raw.get("tolerances", {}).items()
    }
    paths = {str(key): str(value) for key, value in numeric_paths.items()}
    if section is not None:
        prefix = f"$.{section}"
        paths = {
            "$" + path[len(prefix):]: category
            for path, category in paths.items()
            if path == prefix or path.startswith(prefix + ".") or path.startswith(prefix + "[")
        }
    return ScientificComparisonPolicy(
        version=str(raw.get("version")),
        numeric_paths=paths,
        tolerances=tolerances,
    )
