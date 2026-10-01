"""Build the editable Draw.io version of the NL2Hull paper framework figure."""

from __future__ import annotations

import argparse
import base64
from html import escape
from urllib.parse import quote
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, tostring


BLUE = "#2b80d7"
DARK = "#17243a"
PURPLE = "#5c3ca2"
ORANGE = "#f28c28"
GREEN = "#55a868"


def _data_uri(path: Path, mime: str) -> str:
    raw = path.read_bytes()
    if mime == "image/svg+xml":
        text = raw.decode("utf-8").replace("#2c2c2c", "#ff7b38").replace("#13227a", "#6d46b8")
        return "data:image/svg+xml," + quote(text, safe="")
    encoded = base64.b64encode(raw).decode("ascii")
    wrapper = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1 1">'
        f'<image width="1" height="1" preserveAspectRatio="none" href="data:image/png;base64,{encoded}"/>'
        "</svg>"
    )
    return "data:image/svg+xml," + quote(wrapper, safe="")


def _cell(root: Element, cell_id: str, value: str, style: str, *, vertex: bool = True, **attrs: str) -> Element:
    cell = SubElement(root, "mxCell", {"id": cell_id, "value": value, "style": style, "parent": "1", **attrs})
    if vertex:
        cell.set("vertex", "1")
    return cell


def _geometry(cell: Element, x: float, y: float, width: float, height: float, *, relative: bool = False) -> None:
    SubElement(cell, "mxGeometry", {"x": str(x), "y": str(y), "width": str(width), "height": str(height), "as": "geometry", **({"relative": "1"} if relative else {})})


def rect(root: Element, cell_id: str, x: float, y: float, width: float, height: float, fill: str, stroke: str, *, rounded: bool = True, value: str = "", font_size: int = 14, bold: bool = False, align: str = "center", valign: str = "middle", stroke_width: int = 1) -> str:
    style = f"rounded={'1' if rounded else '0'};whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};strokeWidth={stroke_width};fontColor={DARK};fontSize={font_size};align={align};verticalAlign={valign};"
    if bold:
        style += "fontStyle=1;"
    cell = _cell(root, cell_id, value if "<" in value else escape(value, quote=True), style)
    _geometry(cell, x, y, width, height)
    return cell_id


def text(root: Element, cell_id: str, x: float, y: float, width: float, height: float, value: str, *, color: str = DARK, size: int = 14, bold: bool = False, align: str = "center", valign: str = "middle", font_style: int = 0) -> str:
    style = f"text;html=1;strokeColor=none;fillColor=none;fontColor={color};fontSize={size};align={align};verticalAlign={valign};whiteSpace=wrap;"
    if bold:
        style += "fontStyle=1;"
    if font_style:
        style += f"fontStyle={font_style};"
    cell = _cell(root, cell_id, value, style)
    _geometry(cell, x, y, width, height)
    return cell_id


def image(root: Element, cell_id: str, x: float, y: float, width: float, height: float, uri: str) -> str:
    style = f"shape=image;image={uri};imageAspect=0;aspect=fixed;strokeColor=none;fillColor=none;"
    cell = _cell(root, cell_id, "", style)
    _geometry(cell, x, y, width, height)
    return cell_id


def edge(root: Element, cell_id: str, source: str, target: str, *, color: str = DARK, width: int = 2, dashed: bool = False, end: str = "classic") -> str:
    dash = "dashed=1;dashPattern=4 4;" if dashed else ""
    style = f"edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;strokeColor={color};strokeWidth={width};endArrow={end};endFill=1;{dash}"
    cell = _cell(root, cell_id, "", style, vertex=False, edge="1", source=source, target=target)
    SubElement(cell, "mxGeometry", {"relative": "1", "as": "geometry"})
    return cell_id


def dot(root: Element, cell_id: str, x: float, y: float, size: float, color: str = BLUE) -> str:
    style = f"ellipse;html=1;fillColor={color};strokeColor={color};"
    cell = _cell(root, cell_id, "", style)
    _geometry(cell, x, y, size, size)
    return cell_id


def build(output: Path, asset_dir: Path, chat: Path, chip: Path) -> None:
    root = Element("root")
    SubElement(root, "mxCell", {"id": "0"})
    SubElement(root, "mxCell", {"id": "1", "parent": "0"})
    n = 2

    def new() -> str:
        nonlocal n
        value = str(n)
        n += 1
        return value

    # Relative image paths keep the Draw.io file portable within the paper figure directory.
    chat_asset = asset_dir / "chat_orange.svg"
    chip_asset = asset_dir / "chip_purple.svg"
    chat_asset.write_text(chat.read_text(encoding="utf-8").replace("#2c2c2c", "#ff7b38"), encoding="utf-8")
    chip_asset.write_text(chip.read_text(encoding="utf-8").replace("#13227a", "#6d46b8"), encoding="utf-8")
    uri = {
        "chat": _data_uri(chat, "image/svg+xml"),
        "chip": _data_uri(chip, "image/svg+xml"),
        "hull": _data_uri(asset_dir / "kcs_wireframe.png", "image/png"),
        "engine": _data_uri(asset_dir / "kcs_engine.png", "image/png"),
        "edited": _data_uri(asset_dir / "kcs_edited_hull.png", "image/png"),
        "region": _data_uri(asset_dir / "kcs_region.png", "image/png"),
        "operation": _data_uri(asset_dir / "kcs_operation.png", "image/png"),
        "parameters": _data_uri(asset_dir / "kcs_parameters.png", "image/png"),
    }

    # Figure canvas and (a) title strip.
    rect(root, new(), 3, 3, 1666, 336, "#ffffff", "#334155", rounded=False, stroke_width=1)
    rect(root, new(), 3, 3, 1666, 50, "#f3f7fb", "#334155", rounded=False, value="(a) NL2Hull Framework", font_size=25, bold=True, align="left")

    natural = rect(root, new(), 20, 75, 340, 184, "#fff3e7", "#f4c08b", value="", rounded=True)
    text(root, new(), 32, 84, 316, 30, "Natural-language request", size=18, bold=True)
    image(root, new(), 42, 126, 68, 68, uri["chat"])
    rect(root, new(), 116, 124, 226, 76, "#ffffff", "#f4a261", rounded=True, value="<i>Make the bow fuller<br/>and increase the length<br/>of the bulb.</i>", font_size=14, align="left")

    chip_box = rect(root, new(), 398, 75, 220, 184, "#f0eaff", "#cbb9ef", value="", rounded=True)
    text(root, new(), 414, 84, 188, 34, "Chip", color=PURPLE, size=22, bold=True)
    rect(root, new(), 447, 138, 116, 70, "#f8f3ff", PURPLE, rounded=True)
    image(root, new(), 466, 147, 78, 52, uri["chip"])
    text(root, new(), 565, 147, 35, 40, "❄", color="#1c9ce8", size=23)

    actions = rect(root, new(), 658, 75, 290, 184, "#f0f3ff", "#d9def0", value="", rounded=True)
    text(root, new(), 670, 84, 266, 30, "Ordered action projection", size=18, bold=True)
    for row, (label, fill, stroke, color) in enumerate([
        ("A1:  bow + expand", "#fff4e8", "#f6ad55", "#d5651b"),
        ("A2:  bulb + lengthen", "#eaf5ff", "#6aa7ed", "#2371bd"),
        ("A3:  deck + raise", "#eaf8ed", "#7abe8a", "#258447"),
    ]):
        rect(root, new(), 682, 114 + row * 36, 242, 31, fill, stroke, rounded=True, value=label, font_size=14, align="left")
    # Correct the vertical positions of the action rows with explicit overlays.
    # The small labels below are kept separate so they remain editable in Draw.io.
    text(root, new(), 680, 114, 242, 31, "<b>A1:</b> bow + expand", color="#713c13", size=14, align="left")
    text(root, new(), 680, 150, 242, 31, "<b>A2:</b> bulb + lengthen", color="#215e99", size=14, align="left")
    text(root, new(), 680, 186, 242, 31, "<b>A3:</b> deck + raise", color="#24753c", size=14, align="left")
    text(root, new(), 690, 220, 220, 28, "⋮", size=22)

    engine = rect(root, new(), 980, 75, 300, 184, "#e4f2ff", "#9ac7f1", value="", rounded=True)
    text(root, new(), 992, 84, 276, 30, "CFFD Engine", size=20, bold=True)
    image(root, new(), 996, 118, 268, 125, uri["engine"])

    edited = rect(root, new(), 1320, 75, 334, 184, "#eaf8ec", "#a7d8ae", value="", rounded=True)
    text(root, new(), 1334, 84, 306, 30, "Edited hull", size=20, bold=True)
    image(root, new(), 1332, 118, 310, 126, uri["edited"])

    edge(root, new(), natural, chip_box, width=2)
    edge(root, new(), chip_box, actions, width=2)
    edge(root, new(), actions, engine, width=2)
    edge(root, new(), engine, edited, width=2)

    # Bottom panel backgrounds and headers.
    b_panel = rect(root, new(), 3, 341, 555, 597, "#fffaf4", "#334155", rounded=False)
    c_panel = rect(root, new(), 561, 341, 548, 597, "#fbf9ff", "#334155", rounded=False)
    d_panel = rect(root, new(), 1113, 341, 555, 597, "#f8fcff", "#334155", rounded=False)
    rect(root, new(), 3, 341, 555, 47, "#fff0df", "#334155", rounded=False, value="(b) SDD Dataset & SDDBench", font_size=21, bold=True, align="left")
    rect(root, new(), 561, 341, 548, 47, "#f0eafe", "#334155", rounded=False, value="(c) Chip", font_size=21, bold=True, align="left")
    rect(root, new(), 1113, 341, 555, 47, "#e4f2ff", "#334155", rounded=False, value="(d) CFFD Engine", font_size=21, bold=True, align="left")

    # (b) dataset panel.
    text(root, new(), 20, 398, 310, 28, "1.  Ship-form collection", color="#9f3024", size=17, bold=True, align="left")
    text(root, new(), 342, 398, 192, 28, "12 source hulls", size=13, align="right")
    rect(root, new(), 20, 431, 520, 86, "#ffffff", "#f1e1d2", rounded=True)
    for i in range(10):
        image(root, new(), 32 + (i % 5) * 96, 441 + (i // 5) * 35, 82, 29, uri["hull"])
    text(root, new(), 497, 463, 30, 32, "⋯", size=22)

    text(root, new(), 20, 528, 310, 28, "2.  Structured FFD actions", color="#9f3024", size=17, bold=True, align="left")
    text(root, new(), 342, 528, 192, 28, "30k FFD records", size=13, align="right")
    for x, label, key in [(20, "Region", "region"), (182, "Operation", "operation"), (344, "Parameters", "parameters")]:
        rect(root, new(), x, 561, 150, 119, "#ffffff", "#f1e1d2", rounded=True)
        image(root, new(), x + 9, 570, 132, 76, uri[key])
        text(root, new(), x + 8, 646, 134, 25, label, size=13, bold=True)
    rect(root, new(), 506, 561, 34, 119, "#ffffff", "#f1e1d2", rounded=True)
    text(root, new(), 506, 601, 34, 30, "⋯", size=20)

    text(root, new(), 20, 692, 350, 28, "3.  Natural-language requests", color="#9f3024", size=17, bold=True, align="left")
    text(root, new(), 342, 692, 192, 28, "English natural-language requests", size=12, align="right")
    rect(root, new(), 20, 725, 520, 135, "#ffffff", "#f1e1d2", rounded=True)
    image(root, new(), 34, 754, 45, 45, uri["chat"])
    for x, phrase in [(88, "Make the bulb fuller."), (208, "Lengthen the bulb."), (328, "Raise the deck.")]:
        rect(root, new(), x, 748, 108, 61, "#fffdfb", "#f4b77e", rounded=True, value=phrase, font_size=11)
    text(root, new(), 466, 763, 45, 30, "⋯", size=20)

    # (c) Chip panel.
    text(root, new(), 580, 398, 190, 25, "Natural-language request", color=PURPLE, size=14, bold=True, align="left")
    rect(root, new(), 580, 431, 206, 92, "#ffffff", "#d9c9f5", rounded=True)
    image(root, new(), 590, 455, 44, 44, uri["chat"])
    text(root, new(), 641, 446, 132, 67, "<i>Make the bow fuller<br/>and increase the length<br/>of the bulb.</i>", size=11, align="left")
    text(root, new(), 580, 535, 210, 25, "Typed decision questions", color=PURPLE, size=14, bold=True, align="left")
    questions = [("Count", "How many actions?", "#2d8eea"), ("Region", "Which part to edit?", "#2d8eea"), ("Operation", "Expand / Lengthen / ...", "#2d8eea"), ("Level", "Local / Global", "#2d8eea"), ("Constraints", "Symmetry / Volume / ...", "#2d8eea")]
    for i, (heading, detail, color) in enumerate(questions):
        yy = 566 + i * 52
        rect(root, new(), 580, yy, 210, 45, "#ffffff", "#e7def8", rounded=True)
        rect(root, new(), 588, yy + 9, 24, 24, color, color, rounded=False)
        text(root, new(), 620, yy + 3, 160, 18, heading, size=11, bold=True, align="left")
        text(root, new(), 620, yy + 20, 160, 18, detail, size=9, align="left")

    rect(root, new(), 814, 488, 78, 246, "#f8f2ff", PURPLE, rounded=True)
    image(root, new(), 828, 505, 50, 50, uri["chip"])
    text(root, new(), 820, 574, 66, 40, "Chip", color=PURPLE, size=16, bold=True)
    text(root, new(), 820, 630, 66, 28, "❄", color="#1c9ce8", size=20)
    text(root, new(), 806, 450, 94, 27, "", size=12)
    text(root, new(), 908, 398, 182, 25, "Candidate probabilities", color=PURPLE, size=14, bold=True)
    rect(root, new(), 908, 431, 182, 185, "#ffffff", "#d9c9f5", rounded=True)
    probabilities = [("bow", "0.72", 0.72, ORANGE), ("midship", "0.18", 0.18, "#9ba4b2"), ("stern", "0.05", 0.05, "#b7c0cd"), ("deck", "0.05", 0.05, "#b7c0cd")]
    for i, (label, value, amount, color) in enumerate(probabilities):
        yy = 451 + i * 30
        text(root, new(), 918, yy, 62, 20, label, size=10, align="left")
        rect(root, new(), 978, yy + 3, 72, 15, "#edf1f7", "#edf1f7", rounded=False)
        rect(root, new(), 978, yy + 3, max(4, 72 * amount), 15, color, color, rounded=False)
        text(root, new(), 1051, yy, 31, 20, value, color=color, size=10, align="right")
    text(root, new(), 918, 570, 130, 22, "⋮", size=18, align="left")
    text(root, new(), 918, 625, 172, 25, "Ordered action sequence", color=PURPLE, size=14, bold=True, align="left")
    for i, (label, fill, stroke, color) in enumerate([
        ("A1: bow + expand", "#fff4e8", "#f6ad55", "#713c13"),
        ("A2: bulb + lengthen", "#eaf5ff", "#6aa7ed", "#215e99"),
        ("A3: deck + raise", "#eaf8ed", "#7abe8a", "#24753c"),
    ]):
        yy = 659 + i * 38
        rect(root, new(), 908, yy, 182, 30, fill, stroke, rounded=True)
        text(root, new(), 916, yy + 4, 168, 22, label, color=color, size=10, align="left")
    text(root, new(), 918, 774, 160, 22, "⋮", size=18, align="left")

    # (d) CFFD Engine, four process rows.
    rows = [(398, "1", "NURBS waterlines + profiles"), (549, "2", "Control-point deformation"), (700, "3", "Hull reconstruction"), (851, "4", "Checks")]
    for yy, number, label in rows:
        rect(root, new(), 1125, yy, 531, 138 if number != "4" else 75, "#ffffff", "#cce3f6", rounded=True)
        rect(root, new(), 1134, yy + 10, 38, 38, "#4d9bd8", "#4d9bd8", rounded=True, value=number, font_size=18, bold=True)
        text(root, new(), 1182, yy + 10, 330, 28, label, size=14, bold=True, align="left")

    image(root, new(), 1190, 438, 190, 78, uri["hull"])
    text(root, new(), 1385, 427, 80, 20, "Bow profile", size=9)
    text(root, new(), 1480, 427, 95, 20, "Midship profile", size=9)
    text(root, new(), 1583, 427, 70, 20, "Stern profile", size=9)
    for base_x, shape in [(1403, "bow"), (1503, "mid"), (1602, "stern")]:
        # Editable profile schematics: polygonal curves and control points.
        pts = [(base_x, 475), (base_x + 20, 460), (base_x + 20, 492), (base_x, 475)] if shape == "bow" else [(base_x, 460), (base_x + 26, 451), (base_x + 26, 499), (base_x, 492), (base_x, 460)]
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            text(
                root,
                new(),
                min(x0, x1),
                min(y0, y1),
                abs(x1 - x0) + 4,
                abs(y1 - y0) + 4,
                "╱" if y1 != y0 else "─",
                color=BLUE,
                size=12,
            )
        for px, py in pts[:-1]:
            dot(root, new(), px - 3, py - 3, 6, BLUE)

    image(root, new(), 1190, 590, 430, 90, uri["operation"])
    image(root, new(), 1190, 741, 430, 90, uri["hull"])
    for x, label in [(1150, "Symmetry"), (1270, "Deck line"), (1390, "Volume / Draught"), (1530, "Monotonicity")]:
        rect(root, new(), x, 890, 112 if label != "Volume / Draught" else 132, 28, "#e9f8ec", "#b5dfbd", rounded=True)
        text(root, new(), x + 5, 894, (102 if label != "Volume / Draught" else 122), 20, f"✓  {label}", color="#318849", size=9, align="left")

    # A small legend note makes the real model provenance explicit without changing the visual layout.
    text(root, new(), 1200, 824, 400, 16, "KCS benchmark hull; highlighted box denotes the bulb region", color="#53718b", size=8, align="center")

    graph = Element("mxGraphModel", {"dx": "1672", "dy": "941", "grid": "1", "gridSize": "10", "page": "0", "pageWidth": "1672", "pageHeight": "941", "background": "#ffffff"})
    graph.append(root)
    diagram = Element("diagram", {"name": "NL2Hull Framework"})
    diagram.append(graph)
    mxfile = Element("mxfile", {"host": "drawio", "version": "26.0.0"})
    mxfile.append(diagram)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(b"<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n" + tostring(mxfile, encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--chat", type=Path, default=Path("/Users/huowenhua/Downloads/聊天.svg"))
    parser.add_argument("--chip", type=Path, default=Path("/Users/huowenhua/Downloads/芯片.svg"))
    args = parser.parse_args()
    build(args.output, args.assets, args.chat, args.chip)


if __name__ == "__main__":
    main()
