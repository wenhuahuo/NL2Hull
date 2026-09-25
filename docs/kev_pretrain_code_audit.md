# Kev pretraining readiness audit

Revision: `v041_kev4b_ship_cleaned`

## Input policy

The full language snapshot is not used unchanged. The pretraining input excludes the
8,076 template rows whose rollback wording does not match the structured target:

- `undo_previous_step`
- `restore_original_state`
- `change_back_previous_plan`
- `multi_action_combination_rollback`

The 2,019 `continuous_multi_step` rows remain available as sequential-edit examples,
but they are not evidence for a true undo/restore capability. Exact text duplicates are
deduplicated with test taking priority over validation and validation taking priority
over train. The resulting snapshot has 134,558 records:

| split | records | questions |
|---|---:|---:|
| train | 88,604 | 784,004 |
| validation | 22,758 | 198,558 |
| test | 23,196 | 201,774 |

## Fixed issues

- Kev preparation now validates sample/source hull IDs, action indices, source turn
  alignment, duplicate structured IDs, and the 1--3 action limit.
- Kev preparation supports explicit exclusion of invalid rollback rows and deterministic
  text deduplication; both decisions are recorded in `manifest.json`.
- Benchmark field denominators count missing predicted actions as incorrect rather than
  silently omitting them.
- Explicit magnitude values are validated per action, including mixed-action inputs.
- FFD displacement preservation now measures the current loft volume, including vertical
  deformations, rather than integrating areas at stale waterline elevations.
- Geometry metrics are named `vertex_nn_*` to avoid claiming surface Chamfer/Hausdorff
  distances for sampled-vertex nearest-neighbor measurements.
- Kev report comparison exposes directional improvement only for metrics with a clear
  monotonic direction; calibration quantities without one are left unspecified.
- The active profile implementation is documented as the current 32-control model,
  rather than the paper's 14/19-control model.
- The misspelled top-level `visulization/` directory was renamed to `visualization/`.

## Known limitations retained explicitly

- Natural-language generation validates format and numeric preservation, not full
  semantic equivalence of region, operation, constraints, and qualitative level.
- The hull split is hull-ID based; related-hull/family similarity is not automatically
  audited, so generalization claims must be limited to the declared hull split.
- Pi and Jev baselines do not expose identical numeric prediction capabilities and
  should not be compared with one undifferentiated headline metric.
- Longitudinal-order repair in FFD is an implementation constraint; future work should
  report or reject materially repaired actions before making geometric-effect claims.

## Run/provenance policy

The old `v039_kev4b_ship` run and its 43,376-record snapshot are retained as historical
evidence. The new Slurm entry point is:

```text
scripts/slurm/v041-kev4b-ship-cleaned.sbatch
```

It refuses to overwrite an existing run directory, verifies the frozen input hash,
records code/script/data hashes, materializes a fresh Kev data directory, and runs
baseline validation/test benchmarks before training.

Smoke outputs, smoke scripts, and smoke logs were removed only after confirming that no
jobs were running or queued. The formal reconstruction smoke outputs under earlier
`v001`--`v019d` revisions were retained because they are historical geometry evidence,
not part of the language/ Kev smoke workflow.
