# Submission checklist

## Prepared scope

The supplied course guideline requires the complete KDD process, at least five model families, justified evaluation metrics, a six-chapter PDF report, member full names and NIM, and no-login public links to slides, dataset, and source code. The project report covers Chapters 1-6 and the five completed models using the original frozen experiment.

This package intentionally contains only **presentation slides, dataset, source code, and the PDF report**. The team manages the proposal, demo video, and presentation activities separately. No names, student IDs, instructor approval, deadlines, or video URLs are invented.

## Actions still required from the team

1. Add each member's real full name and NIM to `configs/submission.json`, for example `{"full_name":"Real Full Name","nim":"Real Student ID"}`. Add class if known. Do not submit the report while its identity notice remains.
2. Regenerate the report and presentation after identity updates. Use `python scripts/build_report.py`, then the slide authoring command documented in `scripts/README.md`. Complete the publication checks and commits below before refreshing the four submission copies.
3. Review the report, slide order, model caveats, and group attribution. Confirm any instructor-specific template, schedule, or group rules that were not supplied. The proposal rubric mentions NLP; this does not turn the tabular project into an NLP experiment. Resolve any proposal-specific requirement with the instructor rather than inventing an NLP method.
4. Push the prepared local commits to the existing public GitHub repository using `git push origin main`. No force push or history rewrite is needed. Commit refreshed identity/report/slides before pushing. This preparation does not automatically publish GitHub changes.
5. Check all three report appendix links from a signed-out/incognito browser. Slides must download, the dataset must download, and code must be visible without login. Run `python scripts/check_public_links.py`, then `python scripts/build_report.py` to update the publication evidence and remove the pending-publication notice only when all required links pass. Commit the refreshed report and evidence, and push again. Anonymous HTTP checks are not a substitute for a human download check.
6. Prepare the video demo, conduct the presentation, and manage the proposal as requested. Rehearse on the actual presentation computer using the included frozen demo; do not retrain during the demo.
7. From a clean committed checkout, run `python scripts/package_submission.py` to refresh the four final files. Upload `submission/DepoRank_Report.pdf` and any required files to the course portal. Confirm deadline, successful upload, and instructor access.

## Four final files

| File | What to submit |
| --- | --- |
| `submission/DepoRank_Presentation.pptx` | Editable stakeholder-oriented slides |
| `submission/DepoRank_Dataset.zip` | Unchanged dataset, official dictionary, provenance, and license attribution |
| `submission/DepoRank_Source_Code.zip` | Runnable source snapshot, tests, docs, dataset, and trusted compressed demo |
| `submission/DepoRank_Report.pdf` | Six-chapter academic report with cover identities and appendix links |

The source ZIP is a download fallback, not a replacement for the public GitHub repository. It excludes `.git`, virtual environments, caches, uploaded data, raw unpacked model directories, private reference documents, and redundant nested submission archives.

## Public link targets

- Slides: https://github.com/DlegendZ/DepoRank/raw/refs/heads/main/presentation/slides.pptx
- Dataset: https://github.com/DlegendZ/DepoRank/raw/refs/heads/main/data/raw/bank-additional-full.csv
- Source: https://github.com/DlegendZ/DepoRank
- Official dataset fallback: https://archive.ics.uci.edu/static/public/222/bank%2Bmarketing.zip

The public repository was confirmed without authentication on October 6, 2026. Slide and project-dataset URLs remain publication targets until the new commits are pushed. The PDF states this limitation; do not confuse local availability with public availability.

## Evidence preserved

Original model parameters, raw checksums, split boundaries, predictions, winner, threshold, and selection lock remain unchanged. Cosmetic translation is limited to app/CLI text, supporting documents, and chart labels. Historical experiment metadata is retained byte for byte when it is hash-locked. English descriptions accompany these original records.
