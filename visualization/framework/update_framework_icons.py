"""Style supplied SVGs and replace only framework icon cells in-place.

This updater preserves manual Draw.io layout edits and all source SVG files.
Filled question symbols follow the reference figure. The engine SVG already
encodes a thin contour as a filled compound path; adding an outline would create
unwanted double lines, so its native thickness is retained.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from urllib.parse import quote
from xml.etree import ElementTree as ET

SVG_NAMESPACE = "http://www.w3.org/2000/svg"
QUESTION_BLUE = "#2d8eea"
ENGINE_BLUE = "#4d9bd8"
# Optical viewports equalize the visible symbol height in the existing 24px slots.
# The shapes and their holes are retained; no source path is redrawn.
QUESTION_ICONS = (
    ("Count", "count.svg", "176 176 672 672"),
    ("Region", "region.svg", "8 -22 944 944"),
    ("Operation", "operation.svg", "-36 -66 1110 1110"),
    ("Level", "volume-level.svg", "-50 -50 1124 1124"),
    ("Constraints", "constraints.svg", "-16 -10 1056 1056"),
)
ENGINE_CELL = "c-0gONerv2akHK6UkcaI-157"


def style_svg(source: Path, color: str, viewbox: str | None = None) -> str:
    original = ET.parse(source).getroot()
    ET.register_namespace("", SVG_NAMESPACE)
    root = ET.Element(f"{{{SVG_NAMESPACE}}}svg", {
        "viewBox": viewbox or original.get("viewBox", "0 0 1024 1024"),
        "width": "200", "height": "200", "preserveAspectRatio": "xMidYMid meet",
    })
    for child in original:
        shape = deepcopy(child)
        for element in shape.iter():
            if element.tag.rsplit("}", 1)[-1] not in {"path", "rect", "circle", "ellipse", "polygon", "polyline", "line"}:
                continue
            element.attrib.pop("p-id", None)
            element.attrib.pop("style", None)
            element.set("fill", color)
            # Native filled contours have their line thickness baked into their
            # geometry; no extra stroke means clean single contours at small sizes.
            element.set("stroke", "none")
            element.set("stroke-width", "0")
            element.set("stroke-linecap", "round")
            element.set("stroke-linejoin", "round")
        root.append(shape)
    return ET.tostring(root, encoding="unicode")


def _image_style(svg: str) -> str:
    uri = "data:image/svg+xml," + quote(svg, safe="")
    return f"shape=image;image={uri};imageAspect=1;aspect=fixed;strokeColor=none;fillColor=none;"


def _find_question_icon(cells: list[ET.Element], heading: str) -> ET.Element:
    labels = []
    for cell in cells:
        geometry = cell.find("mxGeometry")
        if cell.get("value") != heading or geometry is None:
            continue
        # The (c) panel is the only target; (b) also contains a Region label.
        if 560.0 <= float(geometry.get("x", "0")) <= 800.0 and 500.0 <= float(geometry.get("y", "0")) <= 750.0:
            labels.append(cell)
    if len(labels) != 1:
        raise ValueError(f"Expected one question heading {heading!r}, found {len(labels)}")
    geometry = labels[0].find("mxGeometry")
    x = float(geometry.get("x", "0"))
    y = float(geometry.get("y", "0"))
    matches = []
    for cell in cells:
        g = cell.find("mxGeometry")
        if g is None or cell.get("parent") != labels[0].get("parent") or cell.get("value", ""):
            continue
        if (float(g.get("width", "0")) == 24 and float(g.get("height", "0")) == 24
                and abs(float(g.get("x", "0")) - (x - 32)) < 0.01
                and abs(float(g.get("y", "0")) - (y + 6)) < 0.01):
            matches.append(cell)
    if len(matches) != 1:
        raise ValueError(f"Expected one icon slot for {heading}, found {len(matches)}")
    return matches[0]


def update(diagram: Path, assets: Path) -> list[str]:
    # Resolve every input before changing any diagram or creating styled assets.
    required = [assets / filename for _, filename, _ in QUESTION_ICONS] + [assets / "engine.svg"]
    for source in required:
        if not source.is_file():
            raise FileNotFoundError(source)
    tree = ET.parse(diagram)
    cells = list(tree.getroot().iter("mxCell"))
    replacements = []
    for heading, filename, viewport in QUESTION_ICONS:
        cell = _find_question_icon(cells, heading)
        replacements.append((cell, filename, style_svg(assets / filename, QUESTION_BLUE, viewport)))
    engines = [c for c in cells if c.get("id") == ENGINE_CELL]
    if len(engines) != 1:
        raise ValueError("Cannot uniquely find the existing engine icon")
    replacements.append((engines[0], "engine.svg", style_svg(assets / "engine.svg", ENGINE_BLUE)))
    styled_dir = assets / "styled_icons"
    styled_dir.mkdir(exist_ok=True)
    for cell, filename, svg in replacements:
        (styled_dir / filename).write_text(svg + "\n", encoding="utf-8")
        cell.set("style", _image_style(svg))
    diagram.write_bytes(
        b'<?xml version="1.0" encoding="UTF-8"?>\n'
        + ET.tostring(tree.getroot(), encoding="utf-8")
    )
    return [cell.get("id") for cell, _, _ in replacements]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagram", type=Path, default=Path("docs/paper/pics/framework/nl2hull_framework.drawio"))
    parser.add_argument("--assets", type=Path, default=Path("docs/paper/pics/framework/assets"))
    args = parser.parse_args()
    print("Updated icon cells: " + ", ".join(update(args.diagram, args.assets)))


if __name__ == "__main__":
    main()
