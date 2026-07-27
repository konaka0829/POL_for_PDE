"""Authoritative saved-artifact validation used by scalar reuse."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Protocol

from pol.runtime.artifacts import exact_artifact_tree
from pol.runtime.hashing import stable_object_hash
from pol.runtime.io import file_sha256


@dataclass(frozen=True)
class ArtifactIdentity:
    """Validated content identity of a complete recipe output."""

    step_name: str
    protocol_version: str
    content_hash: str


class StepArtifactContract(Protocol):
    """Recipe-owned validation boundary required before scalar reuse."""

    step_name: str
    protocol_version: str

    def validate_complete(
        self,
        output_dir: Path,
        *,
        expected_config_identity: str | None = None,
        expected_config_path: Path | None = None,
    ) -> ArtifactIdentity: ...


def canonical_config_identity(path: Path) -> str:
    """Hash the strict canonical scientific configuration at ``path``."""
    from .config import canonical_config_json, load_config_json

    return stable_object_hash(canonical_config_json(load_config_json(path)))


def _validate_expected_config(
    saved_path: Path,
    expected_config_identity: str | None,
    expected_config_path: Path | None = None,
) -> None:
    if expected_config_identity is None:
        return
    if canonical_config_identity(saved_path) == expected_config_identity:
        return
    # E1 resolves its foundation-selected reference settings into the saved
    # configuration.  Bind the experiment-owned section and all independent
    # resolution/readout dimensions to the request; prerequisite validation
    # separately owns the selected reference solver/time fields.
    if expected_config_path is not None:
        from .config import load_config_json

        saved = load_config_json(saved_path)
        requested = load_config_json(expected_config_path)
        if (
            saved.e1 is not None
            and requested.e1 is not None
            and saved.e1 == requested.e1
            and saved.domain == requested.domain
            and saved.data == requested.data
            and saved.spatial.target_data_nx == requested.spatial.target_data_nx
            and saved.spatial.surrogate_internal_nx
            == requested.spatial.surrogate_internal_nx
            and saved.spatial.observation_dim
            == requested.spatial.observation_dim
            and saved.spatial.target_output_dim
            == requested.spatial.target_output_dim
        ):
            return
    raise ValueError(
        f"saved artifact config does not match current request: {saved_path}"
    )


def _flat_regular_files(output_dir: Path) -> set[str]:
    if output_dir.is_symlink() or not output_dir.is_dir():
        raise ValueError(f"unsafe artifact directory: {output_dir}")
    files: set[str] = set()
    for child in output_dir.iterdir():
        if child.is_symlink():
            raise ValueError(f"artifact directory contains symlink: {child}")
        if child.is_dir():
            raise ValueError(f"artifact directory contains unexpected directory: {child}")
        if child.is_file():
            files.add(child.name)
    return files


def _identity(
    step_name: str, protocol_version: str, output_dir: Path, names: set[str]
) -> ArtifactIdentity:
    from pol.runtime.io import file_sha256

    return ArtifactIdentity(
        step_name,
        protocol_version,
        stable_object_hash(
            {
                name: {
                    "size_bytes": (output_dir / name).stat().st_size,
                    "sha256": file_sha256(output_dir / name),
                }
                for name in sorted(names)
            }
        ),
    )


class E0ArtifactContract:
    step_name = "e0"

    @property
    def protocol_version(self) -> str:
        from .e0 import E0_SCHEMA_VERSION

        return E0_SCHEMA_VERSION

    def validate_complete(
        self, output_dir: Path, *, expected_config_identity: str | None = None,
        expected_config_path: Path | None = None,
    ) -> ArtifactIdentity:
        import torch

        from .config import canonical_config_json, load_config_json
        from .e0 import load_master_initial_conditions
        from .protocols import E0_REQUIRED_CHECKS
        from .recipes.foundation_validation import ARTIFACTS

        names = set(ARTIFACTS)
        exact_artifact_tree(output_dir, names)
        artifact_manifest = json.loads(
            (output_dir / "artifact_manifest.json").read_text(encoding="utf-8")
        )
        if (
            artifact_manifest.get("schema_version")
            != "paper1-e0-artifact-manifest-v1"
            or artifact_manifest.get("recipe_protocol") != self.protocol_version
        ):
            raise ValueError("E0 artifact manifest protocol mismatch")
        records = artifact_manifest.get("artifacts")
        if not isinstance(records, list):
            raise ValueError("E0 artifact manifest records must be a list")
        recorded_paths = [
            record.get("relative_path") for record in records
            if isinstance(record, dict)
        ]
        expected_paths = names - {"artifact_manifest.json"}
        if (
            len(recorded_paths) != len(records)
            or len(recorded_paths) != len(set(recorded_paths))
            or set(recorded_paths) != expected_paths
        ):
            raise ValueError("E0 artifact manifest record set mismatch")
        for record in records:
            relative = record["relative_path"]
            if (
                not isinstance(relative, str)
                or Path(relative).is_absolute()
                or Path(relative).parts != (relative,)
            ):
                raise ValueError(f"unsafe E0 artifact manifest path: {relative!r}")
            path = output_dir / relative
            if (
                path.is_symlink()
                or not path.is_file()
                or path.stat().st_size != record.get("size_bytes")
                or file_sha256(path) != record.get("sha256")
            ):
                raise ValueError(f"E0 artifact byte integrity mismatch: {relative}")
        summary = json.loads((output_dir / "e0_summary.json").read_text())
        _validate_expected_config(
            output_dir / "resolved_config.json", expected_config_identity,
            expected_config_path,
        )
        if summary.get("schema_version") != self.protocol_version:
            raise ValueError("E0 summary protocol mismatch")
        if summary.get("status") != "pass":
            raise ValueError("E0 summary status is not pass")
        checks = summary.get("required_checks")
        if not isinstance(checks, dict) or set(checks) != E0_REQUIRED_CHECKS or any(
            value != "pass" for value in checks.values()
        ):
            raise ValueError("E0 required checks are not all pass")
        convergence = json.loads(
            (output_dir / "reference_convergence.json").read_text(encoding="utf-8")
        )
        if (
            convergence.get("schema_version") != self.protocol_version
            or convergence.get("joint_status") != "pass"
        ):
            raise ValueError("E0 reference convergence is not a protocol pass")
        selected_space = convergence.get("selected_spatial")
        selected_time = convergence.get("selected_temporal")
        if not isinstance(selected_space, dict) or not isinstance(selected_time, dict):
            raise ValueError("E0 selected reference records are missing")
        selected = summary.get("selected_reference")
        expected_selected = {
            "reference_nx": selected_space.get("candidate_nx"),
            "solver": selected_time.get("solver"),
            "requested_dt": selected_time.get("requested_dt"),
            "requested_fine_dt": selected_time.get("requested_fine_dt"),
            "effective_inner_step": selected_time.get("effective_inner_step"),
            "joint_status": "pass",
        }
        if selected != expected_selected:
            raise ValueError("E0 summary/reference selection mismatch")
        accepted = load_config_json(
            output_dir / "accepted_production_config.json"
        )
        resolved = load_config_json(output_dir / "resolved_config.json")
        if (
            accepted.spatial.reference_nx != selected["reference_nx"]
            or accepted.target.solver != selected["solver"]
            or accepted.target.dt != selected["requested_dt"]
            or accepted.target.fine_dt != selected["requested_fine_dt"]
        ):
            raise ValueError("E0 accepted config/reference selection mismatch")
        # E0 may only clear its gate section and replace the selected reference
        # resolution/time integrator.  Reconstruct that authorized transform.
        from dataclasses import replace
        authorized = replace(
            resolved,
            e0=None,
            spatial=replace(
                resolved.spatial, reference_nx=int(selected["reference_nx"])
            ),
            target=replace(
                resolved.target,
                solver=str(selected["solver"]),
                dt=float(selected["requested_dt"]),
                fine_dt=selected["requested_fine_dt"],
            ),
        )
        if canonical_config_json(accepted) != canonical_config_json(authorized):
            raise ValueError("E0 accepted config contains unauthorized changes")
        load_master_initial_conditions(
            output_dir / "master_initial_conditions.pt",
            load_config_json(output_dir / "accepted_production_config.json"),
        )
        manifest = json.loads((output_dir / "master_manifest.json").read_text())
        archive = torch.load(
            output_dir / "master_initial_conditions.pt",
            map_location="cpu",
            weights_only=False,
        )
        if manifest.get("tensor_hash") != archive["metadata"].get("tensor_hash"):
            raise ValueError("E0 master manifest tensor hash mismatch")
        for name in (
            "reference_convergence.json",
            "resampling_checks.json",
            "input_interface_checks.json",
            "model1_identity.json",
        ):
            payload = json.loads((output_dir / name).read_text())
            if not isinstance(payload, dict):
                raise ValueError(f"invalid E0 JSON artifact: {name}")
        return _identity(self.step_name, self.protocol_version, output_dir, names)

    def validate_failure(self, output_dir: Path) -> ArtifactIdentity:
        names = {
            "e0_summary.json",
            "environment.json",
            "input_interface_checks.json",
            "master_initial_conditions.pt",
            "master_manifest.json",
            "model1_identity.json",
            "reference_convergence.csv",
            "reference_convergence.json",
            "resampling_checks.json",
            "resolved_config.json",
        }
        exact_artifact_tree(output_dir, names)
        summary = json.loads((output_dir / "e0_summary.json").read_text())
        if (
            summary.get("schema_version") != self.protocol_version
            or summary.get("status") != "fail"
            or not summary.get("failure_reasons")
        ):
            raise ValueError("E0 failure summary contract mismatch")
        return _identity(self.step_name, self.protocol_version, output_dir, names)


class MasterDatasetArtifactContract:
    step_name = "master_dataset"
    from .protocols import MASTER_DATASET_SCHEMA_VERSION as protocol_version

    def validate_complete(
        self, output_dir: Path, *, expected_config_identity: str | None = None,
        expected_config_path: Path | None = None,
    ) -> ArtifactIdentity:
        from .datasets import load_master_dataset

        names = {"master_dataset.pt", "manifest.json", "resolved_config.json"}
        exact_artifact_tree(output_dir, names)
        _validate_expected_config(
            output_dir / "resolved_config.json", expected_config_identity,
            expected_config_path,
        )
        load_master_dataset(output_dir)
        return _identity(self.step_name, self.protocol_version, output_dir, names)


class E1ArtifactContract:
    step_name = "e1"

    @property
    def protocol_version(self) -> str:
        from .e1 import E1_SCHEMA_VERSION

        return E1_SCHEMA_VERSION

    def validate_complete(
        self, output_dir: Path, *, expected_config_identity: str | None = None,
        expected_config_path: Path | None = None,
    ) -> ArtifactIdentity:
        from .config import load_config_json
        from .e1_qa import (
            expected_artifacts,
            validate_artifact_set,
            validate_plots,
            validate_saved_numeric_artifacts,
            verify_artifact_manifest,
        )

        config = load_config_json(output_dir / "resolved_config.json")
        _validate_expected_config(
            output_dir / "resolved_config.json", expected_config_identity,
            expected_config_path,
        )
        plot_manifest = json.loads((output_dir / "plot_manifest.json").read_text())
        skip_plots = plot_manifest.get("status") == "skipped"
        plot_names = validate_plots(output_dir, skip_plots=skip_plots)
        names = expected_artifacts(plot_names)
        validate_saved_numeric_artifacts(output_dir, config)
        verify_artifact_manifest(output_dir, names)
        validate_artifact_set(output_dir, names)
        exact_artifact_tree(output_dir, names)
        summary = json.loads((output_dir / "e1_summary.json").read_text())
        if (
            summary.get("schema_version") != self.protocol_version
            or summary.get("status") != "pass"
        ):
            raise ValueError("E1 summary is not a complete pass")
        return _identity(self.step_name, self.protocol_version, output_dir, names)

    def validate_failure(self, output_dir: Path) -> ArtifactIdentity:
        names = {"e1_summary.json", "environment.json", "failed_runs.json"}
        exact_artifact_tree(output_dir, names)
        summary = json.loads((output_dir / "e1_summary.json").read_text())
        if (
            summary.get("schema_version") != self.protocol_version
            or summary.get("status") != "fail"
            or not summary.get("failure_reason")
        ):
            raise ValueError("E1 failure summary contract mismatch")
        failures = json.loads((output_dir / "failed_runs.json").read_text())
        if not isinstance(failures, list) or not failures:
            raise ValueError("E1 failure log contract mismatch")
        return _identity(self.step_name, self.protocol_version, output_dir, names)


class E2ArtifactContract:
    step_name = "e2"

    @property
    def protocol_version(self) -> str:
        from .e2 import E2_SCHEMA_VERSION

        return E2_SCHEMA_VERSION

    def validate_complete(
        self, output_dir: Path, *, expected_config_identity: str | None = None,
        expected_config_path: Path | None = None,
    ) -> ArtifactIdentity:
        from .e2_qa import validate_resume_output

        if not validate_resume_output(output_dir):
            raise ValueError("E2 output is not a complete pass")
        _validate_expected_config(
            output_dir / "resolved_config.json", expected_config_identity,
            expected_config_path,
        )
        names = {
            path.name
            for path in output_dir.iterdir()
            if path.is_file() and not path.is_symlink()
        }
        if any(path.is_symlink() for path in output_dir.iterdir()):
            raise ValueError("E2 output contains symlink")
        return _identity(self.step_name, self.protocol_version, output_dir, names)

    def validate_failure(self, output_dir: Path) -> ArtifactIdentity:
        from .e2_qa import validate_artifact_contract

        summary = json.loads((output_dir / "e2_summary.json").read_text())
        if (
            summary.get("schema_version") != self.protocol_version
            or summary.get("status") != "fail"
        ):
            raise ValueError("E2 failure summary contract mismatch")
        plot = json.loads((output_dir / "plot_manifest.json").read_text())
        validate_artifact_contract(
            output_dir,
            status="fail",
            skip_plots=plot.get("status") == "skipped",
            test_evaluated=False,
            failure_kind=summary.get("failure_kind"),
        )
        names = {
            path.name for path in output_dir.iterdir()
            if path.is_file() and not path.is_symlink()
        }
        return _identity(self.step_name, self.protocol_version, output_dir, names)


_CONTRACTS: dict[str, StepArtifactContract] = {
    "e0": E0ArtifactContract(),
    "master_dataset": MasterDatasetArtifactContract(),
    "e1": E1ArtifactContract(),
    "e2": E2ArtifactContract(),
}


def validate_step_artifacts(
    step_name: str,
    output_dir: Path,
    *,
    expected_config_identity: str | None = None,
    expected_config_path: Path | None = None,
) -> ArtifactIdentity:
    try:
        contract = _CONTRACTS[step_name]
    except KeyError as exc:
        raise ValueError(f"unknown artifact contract: {step_name}") from exc
    return contract.validate_complete(
        output_dir,
        expected_config_identity=expected_config_identity,
        expected_config_path=expected_config_path,
    )
