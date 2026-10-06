# Artifact build and verification scripts

Run commands from the repository root using the Python 3.11 project environment. These scripts prepare artifacts; they never push GitHub commits, record a video, or generate a proposal.

## Demo and software verification

```powershell
.venv/Scripts/python.exe scripts/restore_demo.py
.venv/Scripts/python.exe scripts/check_quality.py
.venv/Scripts/python.exe scripts/build_report.py
```

`restore_demo.py` verifies the bundled archive checksum, rejects unsafe members, checks all existing output files before any copy, and preserves existing matching files. `check_quality.py` runs all tests and records actual counts in `reports/software_quality.json`. `build_report.py` checks the frozen run, refreshes English chart labels, and creates the six-chapter PDF, editable report text, summary, and presentation data.

Fill `configs/submission.json` with real member full names and NIM before the final report build. Empty identities are explicitly disclosed rather than invented.

## Editable presentation authoring

The deck source is `build_slides.mjs`. It uses the Codex bundled **@oai/artifact-tool** JavaScript runtime and presentation validators; this is an artifact-authoring dependency, not an app runtime dependency. The committed PPTX can be edited normally in PowerPoint even when the authoring runtime is unavailable.

Use the runtime paths returned by Codex `load_workspace_dependencies`. Do not install substitute presentation libraries. Link `.build/node_modules` to the bundled Node packages, copy the builder into `.build/`, and configure:

```powershell
$env:PRESENTATIONS_SKILL_DIR = '<installed presentations skill directory>'
$env:RUNTIME_PYTHON = '<bundled Python executable>'
$env:RUNTIME_NODE_MODULES = '<bundled Node packages directory>'
$env:PROJECT_ROOT = (Get-Location).Path
Copy-Item scripts/build_slides.mjs .build/slides-builder.mjs
& '<bundled Node executable>' .build/slides-builder.mjs
.venv/Scripts/python.exe scripts/build_slides_pdf.py
```

The builder creates 16 editable slides, validates layout/package/font/native-table/native-chart requirements, and writes full-size previews under ignored `.build/slides/`. `build_slides_pdf.py` creates a PDF viewing copy from those previews. Human visual review is still required; package checks do not guarantee correct appearance in PowerPoint or another application.

## Frozen run archive

```powershell
.venv/Scripts/python.exe scripts/build_demo_bundle.py
```

Run only when intentionally refreshing the distribution archive from the unchanged verified run. This compresses all five candidate models and their evidence without changing their hashes, includes the active pointer and English charts, omits run logs, and updates `demo/manifest.json`. Original historical metadata may retain its original language because it is immutable evidence.

## Public links and four-file package

```powershell
.venv/Scripts/python.exe scripts/check_public_links.py --references
.venv/Scripts/python.exe scripts/package_submission.py
```

Public checks are anonymous and write observed HTTP statuses to `reports/publication_checks.json`; unpushed slide/data targets will fail. A failed DOI/reference HEAD request does not imply that the cited publication is false; inspect it in a browser. The required slide/dataset/source links must ultimately work without login.

Commit required source and output files before packaging. The source ZIP uses Git's tracked file list, pins file checksums and the current commit, and excludes ignored references/cache/uploads. Packaging refuses dirty tracked changes unless `--allow-dirty` is intentionally supplied, verifies raw-data checksum and software test status, and produces only the four requested files in `submission/`.
