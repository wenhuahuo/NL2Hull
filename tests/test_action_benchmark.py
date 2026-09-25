from benchmarks.action_benchmark import _answer_score, _score_jev, _score_standard


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


def test_jev_score_uses_most_probable_integer_level():
    answers = {
        "magnitude_1": {
            "score": 1.8,
            "probabilities": {"1": 0.2, "2": 0.7, "3": 0.1},
        }
    }
    assert _answer_score(answers, "magnitude_1") == 2


def test_standard_score_accepts_exact_explicit_values():
    target = _action(
        magnitude_level=None,
        magnitude_value=0.02,
        longitudinal_extent=[0.65, 0.9],
        vertical_extent=[0.1, 0.5],
    )
    score = _score_standard({"actions": [target]}, [target])
    assert score["turn_exact"]


def test_missing_predicted_action_counts_as_incorrect_field_values():
    score = _score_standard({"actions": []}, [_action()])
    assert all(value == 1 for value in score["field_totals"].values())
    assert all(value == 0 for value in score["field_correct"].values())


def test_jev_explicit_mode_is_part_of_semantic_score():
    target = _action(
        magnitude_level=None,
        magnitude_value=0.02,
        longitudinal_extent=[0.65, 0.9],
        vertical_extent=[0.1, 0.5],
    )
    prediction = {
        "actions": [
            {
                "region": "bow",
                "operation": "outward",
                "magnitude_mode": "qualitative",
                "magnitude_level": None,
                "constraints": {},
            }
        ]
    }
    score = _score_jev(prediction, [target], numeric_supported=False)
    assert not score["semantic_turn_exact"]
