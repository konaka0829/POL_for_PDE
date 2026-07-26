"""Immutable public types for artifact-only plotting."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping


@dataclass(frozen=True)
class PlotTaskSpec:
    """One requested registered plot recipe and its canonical settings."""

    recipe_id: str
    settings: Mapping[str, Any]


@dataclass(frozen=True)
class PlotContext:
    """Paths and settings passed to one plot renderer."""

    input_dir: Path
    output_dir: Path
    settings: Mapping[str, Any]


@dataclass(frozen=True)
class PlotResult:
    """Files and logical metadata produced by one renderer."""

    outputs: tuple[Mapping[str, Any], ...]


class PlotRenderError(RuntimeError):
    """A renderer failure carrying successful outputs and format failures."""

    def __init__(
        self,
        message: str,
        *,
        outputs: tuple[Mapping[str, Any], ...] = (),
        failures: tuple[Mapping[str, Any], ...] = (),
    ) -> None:
        super().__init__(message)
        self.outputs = outputs
        self.failures = failures


@dataclass(frozen=True)
class PlotRecipe:
    """Registered artifact-only plot implementation."""

    recipe_id: str
    version: str
    supported_experiment_kinds: tuple[str, ...]
    required_input_files: tuple[str, ...]
    render: Callable[[PlotContext], PlotResult]
    validate_settings: Callable[[Mapping[str, Any]], Mapping[str, Any]]
