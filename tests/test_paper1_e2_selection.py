from pol.paper1.e2 import select_first_with_tolerance


def test_validation_selection_uses_config_order_and_not_test():
    rows=[
        {"validation_field_relative_l2_mean":.2,"test":.01,"parameter":1},
        {"validation_field_relative_l2_mean":.1,"test":.9,"parameter":2},
        {"validation_field_relative_l2_mean":.1000001,"test":.001,"parameter":3},
    ]
    assert select_first_with_tolerance(rows,"validation_field_relative_l2_mean",1e-8)["parameter"]==2
    assert "test" not in select_first_with_tolerance.__code__.co_varnames
