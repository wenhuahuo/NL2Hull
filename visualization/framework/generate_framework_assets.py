"""Render bright, smooth, aspect-preserving assets for the NL2Hull framework.

KCS assets retain the actual waterline loft and FFD result. Smooth shading is a
rendering operation only: it never smooths or displaces the mesh vertices.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
import vtk
from PIL import Image
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy

from nurbs_ship_reconstruction.core.geometry import (
    evaluate_waterline_3d, extract_waterline, load_hull, skin_waterlines,
)
from nurbs_ship_reconstruction.core.nurbs import evaluate
from nurbs_ship_reconstruction.core.profile import fit_profile
from nurbs_ship_reconstruction.deformation.ffd import FFDAction, apply_ffd_actions

plt.rcParams.update({"font.family": "Times New Roman"})
BLUE = "#2b80d7"
ORANGE = "#f28c28"
GREEN = "#55a868"
# This is the existing elev=18, azim=-116 view with the bow facing left/front.
ELEVATION = np.deg2rad(18.0)
AZIMUTH = np.deg2rad(-116.0)
CAMERA = np.array([-np.cos(ELEVATION) * np.cos(AZIMUTH),
                   -np.cos(ELEVATION) * np.sin(AZIMUTH), np.sin(ELEVATION)])
BULB_BOX = np.array([[0.87, -0.028, -0.002], [1.012, 0.028, 0.047]])
# Arrows originate at the nose, keel and two flanks of the actual bulb.
FFD_ARROWS = (
    ((0.998, 0.0, 0.022), (1.078, 0.0, 0.022)),
    ((0.970, 0.0, 0.004), (0.970, 0.0, -0.047)),
    ((0.969, -0.019, 0.026), (0.969, -0.082, 0.026)),
    ((0.963, 0.019, 0.026), (0.963, 0.082, 0.026)),
)


def _basis(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    direction = direction / np.linalg.norm(direction)
    right = np.cross([0.0, 0.0, 1.0], direction)
    right /= np.linalg.norm(right)
    up = np.cross(direction, right)
    return right, up


def _project(points: np.ndarray, direction: np.ndarray) -> np.ndarray:
    right, up = _basis(direction)
    return np.asarray(points) @ np.column_stack((right, up))


def _polydata(mesh: trimesh.Trimesh) -> vtk.vtkPolyData:
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(np.asarray(mesh.vertices, dtype=float), deep=True))
    faces = np.asarray(mesh.faces, dtype=np.int64)
    cells = vtk.vtkCellArray()
    cells.SetData(numpy_to_vtkIdTypeArray(np.arange(0, 3 * (len(faces) + 1), 3, dtype=np.int64), deep=True),
                  numpy_to_vtkIdTypeArray(faces.ravel(), deep=True))
    poly = vtk.vtkPolyData()
    poly.SetPoints(points)
    poly.SetPolys(cells)
    return poly


def _surface_image(mesh: trimesh.Trimesh, direction: np.ndarray,
                   extra: np.ndarray | None = None, width: int = 1200
                   ) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Render opaque, smoothly shaded blue without dropping any triangles.

    The overview's bright opaque surface/no-edge convention is retained; VTK
    interpolates point normals and depth-tests the dense loft for the oblique view.
    Returned extent and projection are also used for all overlay annotations.
    """
    direction = direction / np.linalg.norm(direction)
    screen = _project(mesh.vertices, direction)
    if extra is not None:
        screen = np.vstack((screen, _project(extra, direction)))
    lower, upper = screen.min(axis=0), screen.max(axis=0)
    span = upper - lower
    pad = max(span[0] * 0.015, 0.003)
    lower -= pad
    upper += pad
    extent = (lower[0], upper[0], lower[1], upper[1])
    height = max(80, int(round(width * (upper[1] - lower[1]) / (upper[0] - lower[0]))))
    # Match viewport ratio precisely, eliminating subpixel overlay drift.
    mid_y = 0.5 * (lower[1] + upper[1])
    half_height = (upper[0] - lower[0]) * height / width / 2.0
    extent = (lower[0], upper[0], mid_y - half_height, mid_y + half_height)

    clean = vtk.vtkCleanPolyData()
    clean.SetInputData(_polydata(mesh))
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(clean.GetOutputPort())
    normals.ComputePointNormalsOn()
    normals.ComputeCellNormalsOff()
    normals.SetFeatureAngle(55.0)
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOn()
    normals.SplittingOn()
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(normals.GetOutputPort())
    mapper.ScalarVisibilityOff()
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    prop = actor.GetProperty()
    prop.SetColor(0.23, 0.57, 0.90)
    prop.SetInterpolationToPhong()
    prop.SetAmbient(0.60)
    prop.SetDiffuse(0.40)
    prop.SetSpecular(0.14)
    prop.SetSpecularPower(28.0)
    prop.EdgeVisibilityOff()

    renderer = vtk.vtkRenderer()
    renderer.SetBackground(1, 1, 1)
    renderer.SetBackgroundAlpha(0)
    renderer.AddActor(actor)

    right, up = _basis(direction)
    center = right * np.mean(extent[:2]) + up * np.mean(extent[2:])
    # Keep the focal plane near the mesh to fit depth clipping.
    center += direction * np.mean(np.asarray(mesh.vertices) @ direction)
    camera = renderer.GetActiveCamera()
    camera.ParallelProjectionOn()
    camera.SetFocalPoint(*center)
    camera.SetPosition(*(center + direction * 3.0))
    camera.SetViewUp(*up)
    camera.SetParallelScale(half_height)
    renderer.ResetCameraClippingRange()
    window = vtk.vtkRenderWindow()
    window.SetOffScreenRendering(1)
    window.SetAlphaBitPlanes(1)
    window.SetMultiSamples(8)
    window.SetSize(width, height)
    window.AddRenderer(renderer)
    window.Render()
    capture = vtk.vtkWindowToImageFilter()
    capture.SetInput(window)
    capture.SetInputBufferTypeToRGBA()
    capture.ReadFrontBufferOff()
    capture.Update()
    raw = vtk_to_numpy(capture.GetOutput().GetPointData().GetScalars())
    image = np.flipud(raw.reshape(height, width, 4)).copy()
    window.Finalize()
    return image, extent


def _corners(box: np.ndarray) -> np.ndarray:
    lo, hi = box
    return np.array([[lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]],
                     [hi[0], hi[1], lo[2]], [lo[0], hi[1], lo[2]],
                     [lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]],
                     [hi[0], hi[1], hi[2]], [lo[0], hi[1], hi[2]]])


def _draw_cuboid(ax, direction: np.ndarray, color: str) -> None:
    corners = _project(_corners(BULB_BOX), direction)
    edges = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
             (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))
    for start, end in edges:
        ax.plot(*corners[[start, end]].T, color=color, linewidth=1.5,
                alpha=0.90, zorder=10, solid_capstyle="round")


def _front_mesh(mesh: trimesh.Trimesh, x_min: float = 0.79) -> trimesh.Trimesh:
    """Clip exactly to a forward plane; retain all surface triangles in the crop."""
    plane = vtk.vtkPlane()
    plane.SetOrigin(x_min, 0.0, 0.0)
    plane.SetNormal(1.0, 0.0, 0.0)
    clip = vtk.vtkClipPolyData()
    clip.SetInputData(_polydata(mesh))
    clip.SetClipFunction(plane)
    triangulate = vtk.vtkTriangleFilter()
    triangulate.SetInputConnection(clip.GetOutputPort())
    triangulate.Update()
    poly = triangulate.GetOutput()
    vertices = vtk_to_numpy(poly.GetPoints().GetData()).copy()
    faces = vtk_to_numpy(poly.GetPolys().GetConnectivityArray()).reshape(-1, 3)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def _control_points(waterlines) -> np.ndarray:
    controls = []
    for waterline in waterlines:
        for curve in (waterline.after, waterline.fore):
            points = np.asarray(curve["control_points"], dtype=float)
            if points.shape[1] == 2:
                points = np.column_stack((points, np.full(len(points), waterline.z)))
            controls.append(points)
    # Use the same normalized world coordinates as the mesh, without a second
    # normalization that would shift the annotation away from the loft.
    return np.vstack(controls)


def _render(mesh: trimesh.Trimesh, output: Path, *, waterlines=None,
            highlight: bool = False, operation: bool = False,
            parameters: bool = False, front_only: bool = False,
            side: bool = False, waterline_overlay: bool = False) -> None:
    direction = np.array([0.0, 1.0, 0.0]) if side else CAMERA
    if front_only:
        # Crop in geometry space, not by stretching a complete hull into a tile.
        mesh = _front_mesh(mesh)
    marked = highlight or operation or parameters
    extra = _corners(BULB_BOX) if marked else None
    if operation:
        extra = np.vstack((extra, np.asarray(FFD_ARROWS).reshape(-1, 3)))
    points = None
    if waterlines is not None and marked:
        controls = _control_points(waterlines)
        points = controls[(controls[:, 0] >= BULB_BOX[0, 0])
                          & (controls[:, 2] <= BULB_BOX[1, 2])
                          & (controls[:, 1] <= 0.0)]
        points = points[::max(1, len(points) // 24)]
    image, extent = _surface_image(mesh, direction, extra)
    height, width = image.shape[:2]
    if not marked and not waterline_overlay:
        Image.fromarray(image).save(output)
        _trim(output)
        return
    fig = plt.figure(figsize=(width / 180.0, height / 180.0), dpi=180)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(image, extent=extent, origin="upper", zorder=1)
    ax.set_xlim(extent[:2])
    ax.set_ylim(extent[2:])
    ax.set_aspect("equal")
    ax.axis("off")
    if waterline_overlay and waterlines is not None:
        # Sparse visible-side curves retain the meaning of 'NURBS waterlines'
        # without recreating the dense wireframe that hid the surface curvature.
        for waterline in waterlines[::max(1, len(waterlines) // 6)]:
            path = evaluate_waterline_3d(waterline, samples=160)
            path[:, 1] *= -1.0
            line = _project(path, direction)
            ax.plot(*line.T, color="#2875b9", alpha=0.38, linewidth=0.5, zorder=3)
    if marked:
        _draw_cuboid(ax, direction, GREEN if parameters else ORANGE)
        if points is not None and len(points):
            projected = _project(points, direction)
            ax.scatter(*projected.T, s=6, color=GREEN if parameters else ORANGE,
                       edgecolors="white", linewidths=0.25, zorder=11)
    if operation:
        for start, end in FFD_ARROWS:
            p, q = _project(np.array([start, end]), direction)
            ax.annotate("", xy=q, xytext=p,
                        arrowprops={"arrowstyle": "-|>", "color": ORANGE,
                                    "linewidth": 2.0, "mutation_scale": 15},
                        annotation_clip=False, zorder=12)
    fig.savefig(output, transparent=True, pad_inches=0)
    plt.close(fig)
    _trim(output)


def _curve(nurbs: dict) -> np.ndarray:
    return evaluate(np.linspace(0.0, 1.0, 300),
                    np.asarray(nurbs["control_points"]),
                    np.asarray(nurbs["weights"]), knots=np.asarray(nurbs["knots"]))


def _section(mesh: trimesh.Trimesh, x: float) -> np.ndarray:
    path = mesh.section(plane_normal=[1.0, 0.0, 0.0], plane_origin=[x, 0.0, 0.0])
    if path is None or len(path.entities) == 0:
        raise ValueError(f"Hull has no transverse section at x={x}")
    entity = max(path.entities, key=lambda item: len(item.points))
    return np.asarray(path.vertices[entity.points])[:, [1, 2]]


def _profile_panel(points: np.ndarray, controls: np.ndarray, output: Path,
                   *, longitudinal: bool) -> None:
    fig, ax = plt.subplots(figsize=(2.2, 1.25), dpi=240)
    ax.plot(*points.T, color=BLUE, linewidth=3.0)
    ax.scatter(*controls.T, s=16, color=BLUE, zorder=3)
    lo, hi = points.min(axis=0), points.max(axis=0)
    padding = np.maximum((hi - lo) * 0.06, 0.002)
    ax.set_xlim((hi[0] + padding[0], lo[0] - padding[0]) if longitudinal
                else (lo[0] - padding[0], hi[0] + padding[0]))
    ax.set_ylim(lo[1] - padding[1], hi[1] + padding[1])
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(output, transparent=True, pad_inches=0)
    plt.close(fig)
    _trim(output)


def _fit_canvas(path: Path, aspect: float, width: int = 1200) -> None:
    """Pad a transparent asset to a target ratio without stretching its content."""
    with Image.open(path) as source:
        image = source.convert("RGBA")
        height = max(1, round(width / aspect))
        scale = min(width / image.width, height / image.height)
        resized = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
        canvas = Image.new("RGBA", (width, height))
        canvas.paste(resized, ((width - resized.width) // 2, (height - resized.height) // 2))
        canvas.save(path)


def _trim(path: Path, padding: int = 5) -> None:
    with Image.open(path) as source:
        image = source.convert("RGBA")
        box = image.getchannel("A").getbbox()
        if box is None:
            raise ValueError(f"Empty rendered asset: {path}")
        crop = image.crop(box)
        out = Image.new("RGBA", (crop.width + 2 * padding, crop.height + 2 * padding))
        out.paste(crop, (padding, padding))
        out.save(path)


def _render_d1_panels(hull, reconstructed, profile, waterlines, output: Path) -> None:
    _render(reconstructed, output / "kcs_d1_hull.png", waterlines=waterlines,
            side=True)
    _fit_canvas(output / "kcs_d1_hull.png", aspect=190 / 62)
    for name, curve in (("bow", profile.stem_nurbs), ("stern", profile.stern_nurbs)):
        _profile_panel(_curve(curve), np.asarray(curve["control_points"]),
                       output / f"kcs_d1_{name}_profile.png", longitudinal=True)
        _fit_canvas(output / f"kcs_d1_{name}_profile.png", aspect=78 / 62)
    section = _section(hull.mesh, 0.5)
    _profile_panel(section, section[::max(1, len(section) // 8)],
                   output / "kcs_d1_midship_profile.png", longitudinal=False)
    _fit_canvas(output / "kcs_d1_midship_profile.png", aspect=78 / 62)


def _overview_reader():
    path = Path(__file__).resolve().parents[1] / "classic_hull_overview" / "plot_classic_hull_overview.py"
    spec = importlib.util.spec_from_file_location("classic_hull_overview", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    overview = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(overview)
    return overview


def _render_collection(dataset_dir: Path, output: Path) -> list[dict]:
    """Reuse the reviewed overview's STL reader (including S-175 reference)."""
    overview = _overview_reader()
    tile_w, tile_h = 360, 100
    sheet = Image.new("RGBA", (6 * tile_w, 2 * tile_h))
    sources = []
    for index, spec in enumerate(overview.HULLS):
        source = dataset_dir / spec["source"]
        raw, faces = overview.read_decimated_stl(source, target_faces=15000)
        vertices = overview.normalize_vertices(raw)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        rendered, _ = _surface_image(mesh, CAMERA, width=800)
        tile = Image.fromarray(rendered)
        # Remove per-model camera margins before fitting; every hull occupies
        # almost the entire cell width, at the unchanged oblique camera angle.
        bbox = tile.getchannel("A").getbbox()
        if bbox is None:
            raise ValueError(f"Empty collection model {spec['id']}")
        tile = tile.crop(bbox)
        scale = min((tile_w - 10) / tile.width, (tile_h - 6) / tile.height)
        tile = tile.resize((round(tile.width * scale), round(tile.height * scale)), Image.Resampling.LANCZOS)
        x = (index % 6) * tile_w + (tile_w - tile.width) // 2
        y = (index // 6) * tile_h + (tile_h - tile.height) // 2
        sheet.paste(tile, (x, y))
        sources.append({"id": spec["id"], "source": str(source), "faces": len(faces),
                        "display_reference": "generation_reference.vtp" if spec["id"] == "s_175_u_water" else "source STL"})
    sheet.save(output)
    return sources


def _make_meshes(dataset_dir: Path):
    hull = load_hull(dataset_dir / "KCS")
    profile = fit_profile(hull)
    # Denser sampling uses the existing production loft; no geometry smoothing
    # or stretched dimensions are used to obtain the polished appearance.
    waterlines = [extract_waterline(hull, float(z), profile=profile)
                  for z in np.linspace(0.0, hull.depth, 65)]
    reconstructed = skin_waterlines(waterlines, samples=160)
    edited_waterlines = apply_ffd_actions(waterlines, [FFDAction("bulb", "outward", magnitude_level=5)])
    edited = skin_waterlines(edited_waterlines, samples=160)
    return hull, profile, waterlines, reconstructed, edited


def generate(dataset_dir: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    hull, profile, waterlines, reconstructed, edited = _make_meshes(dataset_dir)
    _render(reconstructed, output / "kcs_engine.png", waterlines=waterlines, highlight=True)
    _render(reconstructed, output / "kcs_region.png", waterlines=waterlines, highlight=True, front_only=True)
    _render(reconstructed, output / "kcs_operation.png", waterlines=waterlines,
            operation=True, front_only=True)
    _render(reconstructed, output / "kcs_parameters.png", waterlines=waterlines,
            parameters=True, front_only=True)
    _render(edited, output / "kcs_edited_hull.png", side=True)
    _render(reconstructed, output / "kcs_wireframe.png")
    _render_d1_panels(hull, reconstructed, profile, waterlines, output)
    sources = _render_collection(dataset_dir, output / "ship_collection.png")
    (output / "framework_asset_manifest.json").write_text(json.dumps({
        "kcs_source": str(hull.stl_path), "waterline_count": len(waterlines),
        "samples_per_half_waterline": 160, "reconstructed_faces": len(reconstructed.faces),
        "ffd": {"region": "bulb", "operation": "outward", "magnitude_level": 5},
        "display": {"aspect_preserved": True, "vertex_smoothing": False,
                    "smooth_normals": True, "camera_direction": CAMERA.tolist(),
                    "bulb_box": BULB_BOX.tolist(), "operation_arrows": FFD_ARROWS,
                    "operation_arrows_role": "illustrative alternatives; not all applied to edited hull"},
        "collection": sources,
    }, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("datasets/classic_hulls"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generate(args.dataset, args.output)


if __name__ == "__main__":
    main()
