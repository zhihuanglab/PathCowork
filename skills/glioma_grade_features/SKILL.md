# Glioma — WHO grade-defining features

Use this decision workflow with an existing WSI agent. [Installation and Claude integration](../../install_instruct_claude.md) explains model setup and how to map the helper calls in `skill.py` to your agent. `config.yaml` contains public-grounder prompts and sampling parameters.

**Input** a glioma resection slide. **Output** which of four features were located — necrosis, infiltrating margin, microvascular proliferation, pseudopalisading — each with its outline, and a grade-IV call (microvascular proliferation, or necrosis with pseudopalisading).

## Steps
1. **Tissue** — segment; drop debris.
2. **Overview read (~8 µm/px)** — necrosis zones and the infiltrating margin on the whole section.
3. **High-power fields (2.3 mm at ~3 µm/px)** — placed on the necrosis rims; the model localizes microvascular proliferation and pseudopalisading. These two are smaller than one visual token on the overview, which is why they are read here.
4. **Checks (effort above "fast")** — a pseudopalisade must sit on necrosis confirmed in its own field; a focus on a fold or coverslip-bubble rim is voided; an overview with no necrosis earns a systematic high-power survey.

## Effort
`fast` raw reads · `high` + the checks, survey only when the overview finds no necrosis · `extra` survey always · `max` + an independent second opinion per focus (needs a second model endpoint).

## Limitations
- Higher effort adds fields and cross-checks. Performance depends on the chosen models and cohort. The public-grounder integration does not reproduce the original organ-specific model deployments.
