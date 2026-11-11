# Lymph node — metastasis screening

Use this decision workflow with an existing WSI agent. [Installation and Claude integration](../../install_instruct_claude.md) explains model setup and how to map the helper calls in `skill.py` to your agent. `config.yaml` contains public-grounder prompts and sampling parameters.

**Input** a slide of lymph nodes (colorectal regional nodes using `variants.colon`, breast sentinel/axillary nodes using `variants.breast` in `config.yaml`). **Output** the node count and, per node, whether a metastasis is suspected, with one focus outline. These are candidates for review, not a diagnosis.

## Steps
1. **Find the nodes** — the grounder proposes lymph nodes on the overview; each proposal is matched to a dense-tissue blob whose contour becomes the node, and unmatched blobs are kept. Pale solid masses with no lymphoid rim are added as a second candidate class, since a node replaced by tumour no longer looks lymphoid.
2. **Screen each node** — one read of the whole node, then a grid of smaller tiles (~2.5 mm for colon, ~1.2 mm for breast) that can flip a negative node by vote; a small metastasis is only a few pixels at whole-node scale.
3. **Positive nodes** — one outline of the most convincing focus, and a check for tumour necrosis.
4. **Count** — a rim-less nodule counts as a node only if carcinoma or necrosis is found in it.

## Limitations
- Deposits smaller than the tile scale (isolated tumour cells, small micrometastases) are missed.
- Breast sentinel-node slides often carry no resolution tag — provide the measured micrometres-per-pixel value to the agent.
