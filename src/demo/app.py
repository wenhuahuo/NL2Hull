from __future__ import annotations

import json
import os
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Any

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benchmarks.unified_action import actions_from_kev
from end2end.pipeline import run_turns
from end2end.state import load_hull_state
from end2end.actions import executable_actions
from kev.checkpoint import LoadOptions
from kev.predictors import LocalPredictor
from kev.suite import SERVING_CONTEXT


MODEL_ID = "wenhuahuo/chip-0.8b"
HULL_STATE = Path(__file__).parent / "assets" / "KVLCC2.json"
DEVICE = os.environ.get("NL2HULL_DEVICE", "cpu")

REGIONS = OrderedDict(
    [
        ("bow", "艏部"),
        ("stern", "艉部"),
        ("bulb", "球鼻艏"),
        ("midbody", "中体"),
        ("deck", "甲板"),
        ("bilge", "舭部"),
        ("global", "全船"),
    ]
)
OPERATIONS = OrderedDict(
    [
        ("outward", "外扩"),
        ("inward", "内收"),
        ("upward", "上抬"),
        ("downward", "下压"),
        ("forward", "向艏移动"),
        ("aftward", "向艉移动"),
        ("increase_fullness", "增加丰满度"),
        ("decrease_fullness", "降低丰满度"),
        ("increase_flare", "增加外飘"),
        ("change_bulb_length", "改变球鼻艏长度"),
        ("increase_length", "增加全船长度"),
        ("decrease_length", "缩短全船长度"),
        ("increase_breadth", "增加全船宽度"),
        ("decrease_breadth", "减小全船宽度"),
    ]
)
MAGNITUDES = ["none", "slight", "small", "moderate", "large", "very_large"]

PREDICTOR = LocalPredictor(
    MODEL_ID,
    DEVICE,
    LoadOptions.from_env(),
    context=SERVING_CONTEXT,
)
BASE_WATERLINES, _ = load_hull_state(HULL_STATE)


def _question(
    question_type: str,
    instructions: str,
    criteria: dict[str, str] | list[str] | None = None,
    label: Any = None,
) -> dict[str, Any]:
    question: dict[str, Any] = {
        "type": question_type,
        "instructions": instructions,
        "label": label,
        "src": "nl2hull_demo",
    }
    if criteria is not None:
        question["criteria"] = criteria
    return question


def _record(request: str, action_slots: int) -> dict[str, Any]:
    questions: OrderedDict[str, dict[str, Any]] = OrderedDict()
    questions["action_count"] = _question(
        "choice",
        "当前输入包含几个船型变形动作？",
        {"one": "包含 1 个动作", "two": "包含 2 个动作", "three": "包含 3 个动作"},
        "one",
    )
    for index in range(1, action_slots + 1):
        questions[f"region_{index}"] = _question(
            "choice",
            f"第 {index} 个动作发生在哪个船体区域？",
            REGIONS,
            next(iter(REGIONS)),
        )
        questions[f"operation_{index}"] = _question(
            "choice",
            f"第 {index} 个动作的变形操作是什么？",
            OPERATIONS,
            next(iter(OPERATIONS)),
        )
        questions[f"mode_{index}"] = _question(
            "choice",
            f"第 {index} 个动作使用定性强度还是显式数值？",
            {
                "qualitative": "使用略微、适度、明显等定性表达",
                "explicit": "包含变形量或作用范围等显式数值",
            },
            "qualitative",
        )
        questions[f"magnitude_{index}"] = _question(
            "score",
            f"第 {index} 个动作的定性变化程度是多少？显式数值动作选择 none。",
            MAGNITUDES,
            0,
        )
        questions[f"preserve_displacement_{index}"] = _question(
            "noul",
            f"第 {index} 个动作是否要求保持排水量不变？",
            {
                "true": "用户明确要求保持排水量或排水体积",
                "false": "用户没有提出保持排水量的要求",
            },
            False,
        )
        questions[f"preserve_deck_line_{index}"] = _question(
            "noul",
            f"第 {index} 个动作是否要求保持甲板线不变？",
            {
                "true": "用户明确要求保持甲板线、甲板边线或甲板型线",
                "false": "用户没有提出保持甲板线的要求",
            },
            False,
        )
    return {"state": request, "questions": questions}


def _predict(record: dict[str, Any]) -> dict[str, Any]:
    return PREDICTOR(record)


def _argmax(values: dict[str, float]) -> str:
    return max(values, key=values.get)


def _plot_waterlines(before: list[Any], after: list[Any], path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for axis, waterlines, title in (
        (axes[0], before, "Before"),
        (axes[1], after, "After"),
    ):
        for waterline in waterlines:
            axis.plot(waterline.x, waterline.width, color="#1769aa", linewidth=0.8)
            axis.plot(waterline.x, -waterline.width, color="#1769aa", linewidth=0.8)
        axis.set_title(title)
        axis.set_xlabel("x")
        axis.set_ylabel("half-breadth")
        axis.set_aspect("equal", adjustable="datalim")
        axis.grid(alpha=0.25)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def run_demo(request: str) -> tuple[str | None, str, str | None]:
    request = request.strip()
    if not request:
        raise gr.Error("请输入船型设计请求。")

    first_record = _record(request, 0)
    first_prediction = _predict(first_record)
    count_key = _argmax(first_prediction["probabilities"]["action_count"])
    action_count = {"one": 1, "two": 2, "three": 3}[count_key]

    record = _record(request, action_count)
    prediction = _predict(record)
    actions = executable_actions(record, prediction["probabilities"])
    result = run_turns(BASE_WATERLINES, [actions])

    summary = {
        "hull": "KVLCC2",
        "model": MODEL_ID,
        "request": request,
        "predicted_action_count": action_count,
        "actions": [
            {
                "region": action.region,
                "operation": action.operation,
                "magnitude_level": action.magnitude_level,
                "constraints": dict(action.constraints),
            }
            for action in actions
        ],
        "status": result["status"],
        "geometry_valid": result.get("geometry_valid"),
        "constraints_satisfied": result.get("constraints_satisfied"),
        "final_metrics": result.get("final_metrics"),
    }

    if result["status"] != "completed":
        return None, json.dumps(summary, ensure_ascii=False, indent=2), None

    output_dir = Path(tempfile.mkdtemp(prefix="nl2hull_demo_"))
    image_path = output_dir / "kvlcc2_before_after.png"
    result_path = output_dir / "nl2hull_result.json"
    _plot_waterlines(BASE_WATERLINES, result["final_waterlines"], image_path)
    result_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(image_path), json.dumps(summary, ensure_ascii=False, indent=2), str(result_path)


with gr.Blocks(title="NL2Hull Demo") as demo:
    gr.Markdown(
        "# NL2Hull Demo\n"
        "输入中文船型设计请求，使用 Chip-0.8B 预测类型化决策，并通过 NURBS/FFD 引擎生成 KVLCC2 船型变形结果。\n\n"
        "[论文（arXiv:2610.09896）](https://arxiv.org/abs/2610.09896) · "
        "[模型](https://huggingface.co/wenhuahuo/chip-0.8b) · "
        "[数据集](https://huggingface.co/datasets/wenhuahuo/ship-design-decisions)"
    )
    request = gr.Textbox(
        label="自然语言船型设计请求",
        value="将舭部区域大幅上抬。",
        lines=3,
    )
    run_button = gr.Button("运行 NL2Hull", variant="primary")
    image = gr.Image(label="KVLCC2 水线变形前后对比", type="filepath")
    result = gr.Code(label="推理与几何评估结果", language="json")
    download = gr.File(label="下载 JSON 结果")
    run_button.click(run_demo, inputs=request, outputs=[image, result, download])


if __name__ == "__main__":
    demo.launch()
