from benchmarks.action_benchmark import _score_jev, _score_standard


def _action(**overrides):
    value = {
        "region": "bow",
        "operation": "outward",
        "magnitude_level": 2,
        "magnitude_value": None,
        "longitudinal_extent": None,
        "vertical_extent": None,
        "symmetry": True,
        "constraints": {},
    }
    value.update(overrides)
    return value


def test_standard_score_uses_common_action_projection():
    target = _action()
    score = _score_standard({"actions": [{"region": "bow", "operation": "outward"}]}, [target])
    assert score["turn_exact"]


def test_missing_predicted_action_counts_as_incorrect_field_values():
    score = _score_standard({"actions": []}, [_action()])
    assert all(value == 1 for value in score["field_totals"].values())
    assert all(value == 0 for value in score["field_correct"].values())


def test_jev_uses_the_same_projection_as_standard():
    target = _action()
    score = _score_jev({"actions": [{"region": "bow", "operation": "outward"}]}, [target])
    assert score["turn_exact"]
