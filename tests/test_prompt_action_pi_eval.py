from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_spec = spec_from_file_location(
    "run_prompt_action_pi_eval",
    Path(__file__).parents[1] / "scripts/jev/run_prompt_action_pi_eval.py",
)
_module = module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_module)


def test_kev_record_uses_the_same_direct_action_target():
    record = {
        "state": "把船艏向外扩。",
        "questions": {
            "action_count": {"label": "one"},
            "region_1": {"label": "bow"},
            "operation_1": {"label": "outward"},
        },
        "_meta": {
            "task_key": "sample/turn_0/variant_0",
            "sample_id": "sample",
            "hull_id": "work_boat",
        },
    }

    normalized = _module.evaluation_record(record)

    assert normalized["text"] == record["state"]
    assert normalized["target_actions"] == [
        {"region": "bow", "operation": "outward"}
    ]
