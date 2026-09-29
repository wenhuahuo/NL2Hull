import pytest

from dataset.prepare_kev_ship_data import _validated_splits


def _source():
    return {
        "action_00001": {
            "sample_id": "action_00001",
            "hull_id": "DTC",
            "actions": [
                {
                    "region": "bow",
                    "operation": "outward",
                    "magnitude_level": 2,
                    "magnitude_value": None,
                    "longitudinal_extent": None,
                    "vertical_extent": None,
                    "constraints": {},
                }
            ],
            "turns": [{"output_actions": [0]}],
        }
    }


def _sample(task_key, *, split_hull="DTC", category=None, text="same"):
    sample = {
        "task_key": task_key,
        "sample_id": "action_00001",
        "hull_id": split_hull,
        "turn_index": 0,
        "action_indices": [0],
        "text": text,
        "model": "model-a",
        "variant_index": 0,
    }
    if category is not None:
        sample["augmentation_category"] = category
    return sample


def test_validation_excludes_invalid_rollback_and_deduplicates_text():
    samples = [
        _sample("train", text="same"),
        _sample("duplicate", text="same"),
        _sample("rollback", category="undo_previous_step", text="rollback"),
    ]
    splits, counts = _validated_splits(
        samples,
        _source(),
        exclude_invalid_rollback=True,
        deduplicate_text=True,
    )
    assert [row["task_key"] for row in splits["train"]] == ["train"]
    assert counts["duplicate_removed_train"] == 1
    assert counts["excluded_invalid_rollback"] == 1


def test_validation_rejects_hull_metadata_mismatch():
    with pytest.raises(ValueError, match="hull mismatch"):
        _validated_splits(
            [_sample("bad", split_hull="wigley_hull")],
            _source(),
            exclude_invalid_rollback=False,
            deduplicate_text=False,
        )
