from pathlib import Path

import pytest

from scripts.run_model123_ttilde_sweep import (
    build_ttilde_range,
    parse_models,
    parse_ttilde_values,
    save_combined_plot,
)


def test_build_ttilde_range_default_like_grid():
    values = build_ttilde_range(0.5, 1.5, 0.05)
    assert len(values) == 21
    assert values[0] == pytest.approx(0.5)
    assert values[1] == pytest.approx(0.55)
    assert values[-1] == pytest.approx(1.5)


def test_parse_ttilde_values_explicit_list():
    values = parse_ttilde_values("0.6,1.0,1.4", 0.5, 1.5, 0.05)
    assert values == [0.6, 1.0, 1.4]


def test_parse_models_rejects_invalid_name():
    with pytest.raises(ValueError, match="Unsupported model"):
        parse_models("model1,invalid")


def test_build_ttilde_range_rejects_bad_step():
    with pytest.raises(ValueError, match="ttilde-step"):
        build_ttilde_range(0.5, 1.5, 0.0)


def test_save_combined_plot_logy_writes_outputs(tmp_path: Path):
    rows = {
        "model1": [
            {"Ttilde": 0.5, "test_relL2": 1.0e-1, "test_relL2_plot": 1.0e-1},
            {"Ttilde": 1.0, "test_relL2": 1.0e-2, "test_relL2_plot": 1.0e-2},
        ],
        "model2": [
            {"Ttilde": 0.5, "test_relL2": 5.0e-2, "test_relL2_plot": 5.0e-2},
            {"Ttilde": 1.0, "test_relL2": 5.0e-3, "test_relL2_plot": 5.0e-3},
        ],
    }

    out_path = tmp_path / "ttilde_vs_error_all_models_logy"
    save_combined_plot(rows, out_path, log_y=True)

    for ext in ("png", "pdf", "svg"):
        assert out_path.with_suffix("." + ext).exists()
