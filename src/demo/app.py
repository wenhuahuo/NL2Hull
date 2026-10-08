from __future__ import annotations

import json
import os
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
import plotly.graph_objects as go
from scipy.spatial import ConvexHull

from end2end.actions import executable_actions
from end2end.pipeline import run_turns
from end2end.state import load_hull_state
from kev.checkpoint import LoadOptions
from kev.predictors import LocalPredictor
from kev.suite import SERVING_CONTEXT
from nurbs_ship_reconstruction.core.geometry import skin_waterlines


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
BASE_MESH = skin_waterlines(BASE_WATERLINES, samples=48)


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


def _mesh_trace(mesh: Any, name: str, color: str, opacity: float) -> go.Mesh3d:
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    return go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        name=name,
        color=color,
        opacity=opacity,
        flatshading=True,
        lighting={"ambient": 0.55, "diffuse": 0.8, "specular": 0.25, "roughness": 0.65},
        hoverinfo="skip",
    )


def _mesh_figure(mesh: Any, title: str, original: Any | None = None) -> go.Figure:
    figure = go.Figure()
    if original is not None:
        figure.add_trace(_mesh_trace(original, "Original", "#64748b", 0.28))
    figure.add_trace(_mesh_trace(mesh, title, "#1769aa", 0.82 if original is not None else 0.95))
    figure.update_layout(
        height=360,
        margin={"l": 0, "r": 0, "t": 32, "b": 0},
        title=title,
        showlegend=original is not None,
        scene={
            "xaxis_title": "x",
            "yaxis_title": "y",
            "zaxis_title": "z",
            "aspectmode": "data",
            "camera": {"eye": {"x": 1.45, "y": 1.45, "z": 0.95}},
            "dragmode": "orbit",
        },
    )
    return figure


def _projection_polygon(mesh: Any, axes: tuple[int, int]) -> np.ndarray:
    points = np.asarray(mesh.vertices)[:, list(axes)]
    boundary = points[ConvexHull(points).vertices]
    return np.vstack([boundary, boundary[0]])


def _projection_trace(
    mesh: Any,
    axes: tuple[int, int],
    name: str,
    color: str,
    *,
    fill: str,
    dash: str = "solid",
    opacity: float = 1.0,
) -> go.Scatter:
    polygon = _projection_polygon(mesh, axes)
    return go.Scatter(
        x=polygon[:, 0],
        y=polygon[:, 1],
        mode="lines",
        name=name,
        line={"color": color, "width": 2, "dash": dash},
        fill="toself" if fill != "none" else None,
        fillcolor=color if fill != "none" else None,
        opacity=opacity,
        hoverinfo="skip",
    )


def _projection_figure(
    mesh: Any,
    title: str,
    axes: tuple[int, int],
    axis_titles: tuple[str, str],
    original: Any | None = None,
) -> go.Figure:
    figure = go.Figure()
    if original is not None:
        figure.add_trace(
            _projection_trace(
                original,
                axes,
                "Original",
                "#64748b",
                fill="none",
                dash="dash",
                opacity=0.95,
            )
        )
    figure.add_trace(
        _projection_trace(
            mesh,
            axes,
            title,
            "#1769aa",
            fill="solid",
            opacity=0.48 if original is not None else 0.8,
        )
    )
    figure.update_layout(
        height=280,
        margin={"l": 48, "r": 12, "t": 32, "b": 42},
        title=title,
        showlegend=original is not None,
        xaxis={"title": axis_titles[0], "scaleanchor": "y", "scaleratio": 1},
        yaxis={"title": axis_titles[1]},
    )
    return figure


def _render_views(base_mesh: Any, modified_mesh: Any, show_diff: bool) -> tuple[go.Figure, ...]:
    original = base_mesh if show_diff else None
    return (
        _mesh_figure(base_mesh, "Original 3D model"),
        _mesh_figure(modified_mesh, "Modified 3D model", original),
        _projection_figure(base_mesh, "Original side view", (0, 2), ("x", "z")),
        _projection_figure(modified_mesh, "Modified side view", (0, 2), ("x", "z"), original),
        _projection_figure(base_mesh, "Original top view", (0, 1), ("x", "y")),
        _projection_figure(modified_mesh, "Modified top view", (0, 1), ("x", "y"), original),
    )


def _update_diff(view_state: dict[str, Any] | None, show_diff: bool) -> tuple[go.Figure | None, ...]:
    if view_state is None:
        return (None, None, None)
    return (
        _mesh_figure(view_state["modified_mesh"], "Modified 3D model", view_state["base_mesh"] if show_diff else None),
        _projection_figure(
            view_state["modified_mesh"],
            "Modified side view",
            (0, 2),
            ("x", "z"),
            view_state["base_mesh"] if show_diff else None,
        ),
        _projection_figure(
            view_state["modified_mesh"],
            "Modified top view",
            (0, 1),
            ("x", "y"),
            view_state["base_mesh"] if show_diff else None,
        ),
    )


def run_demo(request: str, show_diff: bool) -> tuple[Any, ...]:
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
    summary_text = json.dumps(summary, ensure_ascii=False, indent=2)

    if result["status"] != "completed":
        return (None, None, None, None, None, None, summary_text, None, None)

    modified_mesh = skin_waterlines(result["final_waterlines"], samples=48)
    views = _render_views(BASE_MESH, modified_mesh, show_diff)
    output_dir = Path(tempfile.mkdtemp(prefix="nl2hull_demo_"))
    result_path = output_dir / "nl2hull_result.json"
    result_path.write_text(summary_text, encoding="utf-8")
    view_state = {"base_mesh": BASE_MESH, "modified_mesh": modified_mesh}
    return (*views, summary_text, str(result_path), view_state)


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
    gr.Examples(
        examples=[
            ["将球鼻艏长度非常明显地增加"],
            ["将甲板稍微后移"],
            ["将全船长度非常大幅地增加"],
        ],
        example_labels=[
            "1. 将球鼻艏长度非常明显地增加",
            "2. 将甲板稍微后移",
            "3. 将全船长度非常大幅地增加",
        ],
        inputs=request,
        label="自然语言候选（点击即可填入）",
        examples_per_page=3,
    )
    show_diff = gr.Checkbox(
        label="在修改后的三维模型、侧视图和顶视图中叠加原始船型",
        value=True,
    )
    run_button = gr.Button("运行 NL2Hull", variant="primary")
    view_state = gr.State()

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 原始船型 · 三维模型")
            original_3d = gr.Plot(show_label=False)
        with gr.Column():
            gr.Markdown("### 修改后船型 · 三维模型")
            modified_3d = gr.Plot(show_label=False)
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 原始船型 · 侧视图")
            original_side = gr.Plot(show_label=False)
        with gr.Column():
            gr.Markdown("### 修改后船型 · 侧视图")
            modified_side = gr.Plot(show_label=False)
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 原始船型 · 顶视图")
            original_top = gr.Plot(show_label=False)
        with gr.Column():
            gr.Markdown("### 修改后船型 · 顶视图")
            modified_top = gr.Plot(show_label=False)

    result = gr.Code(label="推理与几何评估结果", language="json")
    download = gr.File(label="下载 JSON 结果")
    outputs = [
        original_3d,
        modified_3d,
        original_side,
        modified_side,
        original_top,
        modified_top,
        result,
        download,
        view_state,
    ]
    run_button.click(run_demo, inputs=[request, show_diff], outputs=outputs)
    show_diff.change(
        _update_diff,
        inputs=[view_state, show_diff],
        outputs=[modified_3d, modified_side, modified_top],
    )


if __name__ == "__main__":
    demo.launch()
