from benchmarks.unified_action import (
    actions_from_kev,
    aggregate_action_results,
    score_actions,
    target_actions_from_kev,
)


def _record():
    return {
        "questions": {
            "action_count": {"label": "two"},
            "region_1": {"label": "bow"},
            "operation_1": {"label": "outward"},
            "region_2": {"label": "stern"},
            "operation_2": {"label": "inward"},
        }
    }


def test_kev_projection_and_target_share_canonical_action_shape():
    record = _record()
    probabilities = {
        "action_count": {"one": 0.1, "two": 0.9, "three": 0.0},
        "region_1": {"bow": 1.0},
        "operation_1": {"outward": 1.0},
        "region_2": {"stern": 1.0},
        "operation_2": {"inward": 1.0},
    }
    prediction = actions_from_kev(record, probabilities)
    target = target_actions_from_kev(record)
    assert prediction == {"actions": target}
    assert score_actions(prediction, target)["turn_exact"]


def test_predicted_extra_actions_are_a_failed_prediction():
    record = _record()
    probabilities = {
        "action_count": {"one": 0.1, "two": 0.1, "three": 0.8},
        "region_1": {"bow": 1.0},
        "operation_1": {"outward": 1.0},
        "region_2": {"stern": 1.0},
        "operation_2": {"inward": 1.0},
    }
    try:
        actions_from_kev(record, probabilities)
    except ValueError as error:
        assert "exceeds" in str(error)
    else:
        raise AssertionError("expected an invalid predicted action count")


def test_failed_prediction_is_in_accuracy_denominator():
    summary = aggregate_action_results([{
        "status": "failed",
        "target_actions": [{"region": "bow", "operation": "outward"}],
    }])
    assert summary["requested_turns"] == 1
    assert summary["failed_turns"] == 1
    assert summary["turn_exact_rate"] == 0.0
    assert summary["field_totals"] == {"region": 1, "operation": 1}
