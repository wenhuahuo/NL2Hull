# NURBS ship reconstruction

This project reproduces the vertical parameterization idea in
`docs/related_works/j.issn.1673-3185.2017.05.004-2026-09-18_06-16-25.md`:
source STL waterlines are reduced to low-dimensional degree-3 NURBS curves,
then skinned through the design-draft levels.

## Run

```bash
python -m pip install -e .
python -m nurbs_ship_reconstruction.cli
```

The benchmark uses every hull listed in `datasets/classic_hulls/manifest.yaml`.
Each experiment uses a unique revision name of the form
`<revision>_classic_hulls_nurbs_skinning`. Results live under `outputs/` and
are ignored by Git. The runner never clears or overwrites an existing run.

Current full run: `outputs/v006_classic_hulls_nurbs_skinning/`.
Earlier runs are kept as evidence:

- `v001_classic_hulls_nurbs_skinning`: failed on DTMB5415 baseline waterline sampling
- `v002_classic_hulls_nurbs_skinning`: failed on DTMB5415 bulb-truncated waterlines
- `v003_classic_hulls_nurbs_skinning`: first completed reconstruction, free flat-section y
- `v004_classic_hulls_nurbs_skinning`: locked a fake parallel midbody on every waterline
- `v005_classic_hulls_nurbs_skinning`: extracted `Lpf` but still forced y = beam

For a short smoke run:

```bash
python -m nurbs_ship_reconstruction.cli --hull wigley_hull --levels 12 \
  --output outputs/v006_smoke_wigley_hull
```

## Method and scope

1. Each original STL is read directly and normalized by its own bounding-box
   length; no canonical STL or STEP geometry is used as input.
2. Horizontal STL intersections are reduced to the maximum half-breadth at
   longitudinal bins. For every waterline, area and centroid are integrated
   from that extracted breadth curve.
3. Each half-body is fitted with an 8-control-point clamped cubic NURBS curve.
   Controls are ordered from the outside end toward midships. `P4=P5=P6` are
   coincident, and `y3 = y4 = y7` is the midship half-breadth, matching the
   paper's straight-section constraint. Free variables are
   `[x1, y1, x2, y2, x3, y3, x_flat, y_flat, ω1, ω2, ω3]`. Forcing `y = Bwf` on
   the straight section, as in the paper's design model, inflates reverse-fit
   error on hulls without a true parallel midbody; those ablations are `v004`
   and `v005`. The paper's evolutionary search
   on area/centroid only is replaced by a bounded least-squares fit to the
   extracted waterline, because those three scalar targets underdetermine the
   NURBS variables. Area and centroid are still computed as extracted control
   parameters and reported as reconstruction errors.
4. The fitted waterlines are mirrored about the extracted centerline and
   skinned from baseline to the dataset's reviewed draft fraction. The output
   is therefore an immersed hull surface, not a reconstruction of the deck
   above design draft. Stem/stern contour NURBS models from the paper are not
   fitted independently; their effect is captured only through the extracted
   waterline endpoints.

Each hull result contains:

- `source_parameters.json`: extracted waterline targets and NURBS control data;
- `reconstructed_immersed_hull.stl`: reconstructed skin;
- `metrics.json`: bidirectional sampled vertex distances and waterline errors;
- `comparison.png`: plan/profile/waterline/error visualization.

The run root contains `run_manifest.json`, `benchmark_summary.csv/json`, and
`benchmark_summary.png`. The manifest records input, output, status, job ID
(`null` for a local run), and failure reason for every case.
