# Reviewer

Browser-based reviewer for three PathLearn output types:

1. `human_readable` (image + sentence)
2. `machine_friendly` (grounded image/text records)
3. `evaluation` (QA benchmark items)

Human/machine items display:
- `RAW` transcript text
- `REFINED` text (if generated)
- prior-context window
- eye-attention contour image (if generated in final outputs)

## Build Manifest

```bash
python3 reviewers/build_review_manifest.py
```

Default output:
- `reviewers/data/review_manifest.json`

## Launch Reviewer

```bash
python3 reviewers/launch_reviewer.py
```

Then open:
- `http://127.0.0.1:8765/reviewers/web/index.html`

Direct cohort links:
- `http://127.0.0.1:8765/reviewers/web/index.html?type=human_readable`
- `http://127.0.0.1:8765/reviewers/web/index.html?type=machine_friendly`
- `http://127.0.0.1:8765/reviewers/web/index.html?type=evaluation`

## Review Workflow

- Filter by type (`human_readable`, `machine_friendly`, `evaluation`)
- For each item, click:
  - `Yes` (valid)
  - `No` (invalid)
- Add optional reviewer note
- Export decisions as JSON (`review_decisions.json`)

Decisions are also cached in browser localStorage.
