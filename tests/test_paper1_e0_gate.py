from pol.paper1.e0 import build_required_checks


def _artifacts():
    return (
        {"status": "pass"}, {"status": "pass"},
        {"spatial_status": "pass", "temporal_status": "pass", "joint_status": "pass"},
        {"finite_data_interface": {"status": "pass"}, "no_high_frequency_leak": {"status": "pass"}, "target_coefficient_consistency": {"status": "pass"}},
        {"full_observation": {"status": "pass"}, "bandlimited_reduced_j": {"status": "pass"}, "aliasing_counterexample": {"status": "pass"}},
    )


def test_every_computed_check_is_required():
    required = build_required_checks(*_artifacts())
    assert len(required) == 11 and all(v == "pass" for v in required.values())


def test_coefficient_and_alias_failures_reach_top_gate():
    items = list(_artifacts())
    items[3]["target_coefficient_consistency"]["status"] = "fail"
    assert build_required_checks(*items)["target_coefficient_consistency"] == "fail"
    items = list(_artifacts())
    items[4]["aliasing_counterexample"]["status"] = "fail"
    assert build_required_checks(*items)["model1_aliasing_counterexample"] == "fail"
