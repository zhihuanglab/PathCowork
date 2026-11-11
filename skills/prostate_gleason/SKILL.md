# Prostate needle biopsy — Gleason score

Use this decision workflow with an existing WSI agent. [Installation and Claude integration](../../install_instruct_claude.md) explains model setup and how to map the helper calls in `skill.py` to your agent. `config.yaml` contains public-grounder prompts and sampling parameters.

**Input** a needle-biopsy slide. **Output** a Gleason score (primary + secondary), the regions examined, and the per-patch calls behind it.

## Steps
1. **Cores** — segment the tissue cores on the thumbnail.
2. **Where to look (low power)** — the grounder sees each slide as one low-power field and outlines up to 6 regions whose architecture deserves a closer look. It proposes; it does not grade. An outline that lands mostly off the tissue is clipped back onto it.
3. **Patch budget** — 320 µm patches: 80 % inside the proposed regions (by area), 20 % in the surrounding tissue as a contrast. The budget is the effort setting (default 30).
4. **What it is (high power)** — each patch is graded benign / atypical / pattern 3 / 4 / 5 by a general multimodal model.
5. **Score** — primary = the most frequent malignant pattern, secondary = the next most frequent (the primary again if there is only one).

## Limitations
- Primary/secondary come from patch counts, not from measured tumour area.
- The patch grader is a general-purpose multimodal model reached through an OpenAI-compatible endpoint; the released grounder is used only for step 2.
