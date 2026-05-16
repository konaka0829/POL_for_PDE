import importlib


def test_core_modules_import_without_side_effects():
    modules = [
        "pol.burgers_spectral_1d",
        "pol.reservoir_1d",
        "pol.ridge",
        "pol.elm",
        "pol.features_1d",
        "pol.model123_1d.metrics",
        "pol.model123_1d.predictors",
        "pol.model123_1d.error_decomposition",
        "model123_burgers_1d",
        "model1_error_decomposition_1d",
        "scripts.run_model123_param_sweep",
    ]
    for name in modules:
        importlib.import_module(name)
