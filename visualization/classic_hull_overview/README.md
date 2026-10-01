# Figure 1: classic hull overview

`plot_classic_hull_overview.py` generates the twelve-hull overview used for
`Fig.~\ref{fig:source-hulls}`. It reads the original STL meshes from
`datasets/classic_hulls`, normalizes each hull to unit length while preserving
its aspect ratio, and renders side and front orthographic projections.

The displayed names follow the local manifest and external terminology review:

- DTC — Duisburg Test Case
- DTMB 5415 — preliminary surface-combatant benchmark hull
- KCS — KRISO Container Ship
- KVLCC2 — KRISO Very Large Crude Carrier 2
- Container ship hull — descriptive label for a generic GrabCAD asset
- Frigate hull — descriptive label; the local `firgate` directory is a source-name spelling error
- NPL Round Bilge 4a — 4a variant from the NPL round-bilge series
- NPL Round Bilge — companion source geometry
- S-175 container ship — ITTC benchmark hull
- Series 60 hull — Series 60 methodical-series hull
- Wigley hull — Wigley mathematical benchmark hull
- Workboat hull — descriptive label for a generic GrabCAD asset

The NPL labels deliberately distinguish the retained 4a variant from the
full-scale source geometry. The dataset manifest records that the 4a variant has
geometrically different breadth/depth ratios; the two entries should not be
presented as a simple model-scale/full-scale pair.

Naming checks used public references for DTC, KCS, KVLCC2, DTMB 5415, and S-175:

- DTC: [Duisburg Test Case](https://www.ittc.info/media/11874/75-02-06-06.pdf)
- KCS: [KRISO standard ships](https://www.kriso.re.kr/menu.es?mid=a20206000000)
- KVLCC2: [KRISO VLCC / KVLCC2](https://www.nmri.go.jp/study/research_organization/fluid_performance/cfd/cfdws05/Detail/KVLCC/tanker.html)
- DTMB 5415: [NMRI benchmark description](https://www.nmri.go.jp/study/research_organization/fluid_performance/cfd/cfdws05/Detail/5415/combatant.html)
- S-175: [ITTC S-175 benchmark description](https://www.ittc.info/media/9711/75-02-07-025.pdf)
- NPL 4a: [NPL round-bilge series reference](https://eprints.soton.ac.uk/46442/1/071.pdf)

The three GrabCAD-derived generic assets do not have a verifiable standard
proper name in the available source records, so descriptive labels are used.

## Render

```bash
python visualization/classic_hull_overview/plot_classic_hull_overview.py
```

The script exports SVG, PDF, PNG, and TIFF files in this directory. VTK is used
for STL reading and mesh decimation, while matplotlib performs all figure
rendering and export. The S-175 source STL is a multi-shell model; aggressive
triangle decimation collapsed its longitudinal extent. For the overview tile,
the script therefore uses the reviewed `generation_reference.vtp` sections to
build a compact surface, preserving the recorded unit-length geometry and
avoiding the failed decimation result.
