# Skin — melanocytic lesion

Use this decision workflow with an existing WSI agent. [Installation and Claude integration](../../install_instruct_claude.md) explains model setup and how to map the helper calls in `skill.py` to your agent. `config.yaml` contains public-grounder prompts and sampling parameters.

**Input** a skin biopsy slide. **Output** junctional/dermal foci labelled by concept, each with the model's own outline, and a count of candidate mitotic figures. It lists findings; it does not call melanoma versus naevus.

## Steps
1. **Pieces** — segment the tissue pieces; drop debris.
2. **Junction** — trace the dermo-epidermal junction of each piece in one ~6 mm field.
3. **Read along the junction** — 1.5 mm fields placed along it (up to 8 per piece). Each field is asked six concepts, one request per concept: confluent proliferation, junctional nests, pagetoid spread, lentiginous proliferation, dermal nests, melanophages. Overlapping hits merge into one focus carrying every matching concept. The public grounder returns shapes without calibrated confidence scores.
4. **Mitoses** — a detector run at 0.46 µm/px over each piece and its dermis, merged by centre.

## Limitations
- The public grounder does not provide calibrated confidence scores; retain the raw response and distinguish empty findings from failed requests.
- The mitosis pass is capped per call, so the count is a sampled estimate ("candidate" figures).

The mitosis stage uses a separate detector supplied by the host agent. If it is unavailable, report that component as unavailable; do not substitute zero.
