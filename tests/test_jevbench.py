from copy import deepcopy
from types import SimpleNamespace

import pytest

from dataset.jevbench import jevbench_probabilities, kev_record


def task(qtype, criteria, labels, expected):
    return SimpleNamespace(id="public-example", family="policy", state={"document": "unchanged"},
                           question={"type": qtype, "instructions": "unchanged", "criteria": criteria},
                           labels=labels, expected=expected)


def test_noul_bridge_preserves_state_and_criteria_without_answer_leakage():
    item = task("noul", {"false": "not permitted", "true": "permitted"}, ["no", "yes"], "yes")
    original = deepcopy(item)
    record = kev_record(item)
    assert record["state"] == original.state
    assert record["questions"]["decision"]["label"] is True
    assert record["questions"]["decision"]["criteria"] == original.question["criteria"]
    assert "expected" not in record and "provenance" not in record
    assert item.question == original.question
    assert jevbench_probabilities(item, {"probabilities": {"decision": {"false": 0.3, "true": 0.7}}}) == {"no": 0.3, "yes": 0.7}


def test_choice_keeps_criteria_order_even_when_labels_have_different_order():
    item = task("choice", {"z": "last", "a": "first"}, ["a", "z"], "z")
    record = kev_record(item)
    assert list(record["questions"]["decision"]["criteria"]) == ["z", "a"]
    assert record["questions"]["decision"]["label"] == "z"
    pred = {"probabilities": {"decision": {"z": 0.8, "a": 0.2}}}
    assert jevbench_probabilities(item, pred) == pred["probabilities"]["decision"]


def test_score_keeps_integer_ground_truth_and_ordered_levels():
    item = task("score", ["low", "medium", "high"], ["0", "1", "2"], 2)
    assert kev_record(item)["questions"]["decision"]["label"] == 2


def test_mismatched_choice_labels_are_not_repaired():
    item = task("choice", {"a": "A"}, ["a", "b"], "a")
    with pytest.raises(ValueError, match="criteria differ"):
        kev_record(item)


def test_invalid_answer_ids_are_not_ignored():
    item = task("noul", None, ["no", "yes"], "no")
    with pytest.raises(ValueError, match="answer IDs"):
        jevbench_probabilities(item, {"probabilities": {"wrong": {"false": 1, "true": 0}}})


def test_invalid_binary_keys_are_not_repaired():
    item = task("noul", None, ["no", "yes"], "no")
    with pytest.raises(ValueError, match="noul probability keys"):
        jevbench_probabilities(item, {"probabilities": {"decision": {"no": 1, "yes": 0}}})
