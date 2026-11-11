# Breast — Nottingham histologic grade

Use this decision workflow with an existing WSI agent. [Installation and Claude integration](../../install_instruct_claude.md) explains model setup and how to map the helper calls in `skill.py` to your agent. `config.yaml` contains public-grounder prompts and sampling parameters.

**Input** a slide of invasive breast carcinoma. **Output** the three component scores (tubule formation, nuclear pleomorphism, mitotic count), their sum and grade 1–3, with the tumour outlines and every field that was read.

## Steps
1. **Tumour** — outline the invasive carcinoma, one polygon per focus, per tissue piece. The grade is read across all foci.
2. **Tubule formation (low power)** — 1.4 mm fields spread over the whole tumour, because it is a proportion of the whole tumour. One multi-image judgement gives the score; per-field reads are the fallback.
3. **Mitotic count** — detector over every focus; the score is taken from the densest 2 mm².
4. **Nuclear pleomorphism (high power)** — 240 µm fields ranked by distance to that mitotic hotspot and kept ≥ 600 µm apart. A field with no carcinoma abstains.
5. **Grade** — sum of the three 1–3 scores: 3–5 grade 1, 6–7 grade 2, 8–9 grade 3. A component that could not be read is named and the grade is withheld — never guessed.

## Limitations
- Tubule formation is under-read when the sampled fields miss the tubular part of the tumour, which pushes the grade up.
- Nuclear grade 1 is rarely called.
- Mitotic cut-offs depend on field area and should be set for the cohort.
- Tubule and nuclear reads use a general multimodal model (OpenAI-compatible endpoint); the released grounder does step 1.

The mitosis stage uses a separate detector supplied by the host agent. If it is unavailable, report that component as unavailable; do not substitute zero.
