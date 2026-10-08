# Project Agent Conventions

## Scope

This file records storage and naming conventions for the NL2Hull project. Follow it for new development and deployment artifacts.

## File storage

- Production Python packages belong under `src/`, separated by function:
  - `src/dataset/` for dataset preparation;
  - `src/end2end/` for end-to-end execution;
  - `src/nurbs_ship_reconstruction/` for geometry and reconstruction;
  - `src/demo/` for reusable Hugging Face Space demo code.
- Executable project scripts belong under `scripts/`, separated by function. Deployment helpers belong under `scripts/deploy/`.
- Tests belong under `tests/` and mirror the relevant source package where practical.
- Logs belong under `logs/`; generated experiment or validation results belong under `outputs/`.
- Hugging Face Space source files are maintained under `src/demo/` and uploaded to the Space root with their required Space filenames. Space-only small demo assets belong under `src/demo/assets/`; model weights and large datasets must remain remote and must not be committed.
- Paper and submission artifacts remain under `docs/` and are not part of the software deployment.

## Naming

- Use descriptive, content-based names; do not use numeric experiment prefixes for new work.
- Use lowercase `snake_case` for Python files, directories, and scripts.
- Use `app.py`, `requirements.txt`, and `README.md` only where Hugging Face Spaces requires those exact root filenames.
- Use `nl2hull_demo` for the local demo component and `nl2hull_space` for deployment-related identifiers when a prefix is needed.
- Logs and result directories must describe the operation, for example `logs/apps/nl2hull_space/` and `outputs/nl2hull_space_smoke/`.

## Change management

- Make the smallest necessary change and keep commits focused by content type.
- Do not commit model weights, datasets, virtual environments, caches, or generated build artifacts.
- Before deleting, overwriting, or renaming artifacts, check for running jobs and conflicting targets.
- Do not push Git history unless the user explicitly requests it. Deploying a requested Hugging Face Space may push only the Space files to that Space.
