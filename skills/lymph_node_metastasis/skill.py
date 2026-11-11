"""Lymph node — screen every node on the slide for metastatic carcinoma.

One skill for regional nodes of a colorectal resection (variants.colon) and for
breast sentinel/axillary nodes (variants.breast in config.yaml): the prompts are shared,
only the organ wording differs. The output is a list of candidates for review.
"""
from pathcowork.analysis import colon_skills as ln
from pathcowork.analysis import common_skills as common
from pathcowork.analysis import skin_skills            # drop_debris is organ-agnostic
from pathcowork.headless import evidence

PROJECT = "colon_ln"       # host agent selects breast_ln for sentinel nodes


async def run(ctx) -> dict:
    # 1. Tissue on the thumbnail.
    if not skin_skills.drop_debris(await common.segment_tissue()):
        return {"finding": ("LN screening", "No usable tissue on this slide."), "evidence": []}

    # 2. Find every lymph node section on the overview. The model proposes nodes;
    #    each proposal is matched to a dense-tissue blob whose contour becomes the
    #    node, and pale rim-less masses (a node replaced by tumour does not look
    #    lymphoid) are added as a second candidate class.
    nodes = await ln.find_nodes()
    if not nodes:
        return {"finding": ln.finding([], []), "nodes": [], "evidence": []}

    # 3. Screen each node: one read of the whole node, then — because a small
    #    metastasis is a few pixels at that scale — a grid of ~2.5 mm tiles that
    #    can flip a negative node by vote. Positive nodes get one focus outline
    #    and a check for tumour necrosis.
    screens = await ln.screen_nodes(nodes)

    by_id = {s["node_id"]: s for s in screens}
    ev = []
    for n in nodes:
        s = by_id.get(n["node_id"], {})
        if s.get("focus"):
            ev.append(evidence(f"suspected metastasis · node {n['node_id'] + 1}", s["focus"]))
        if s.get("necrosis"):
            ev.append(evidence(f"tumour necrosis · node {n['node_id'] + 1}", s["necrosis"]))
    return {
        "finding": ln.finding(nodes, screens),
        "nodes": [{"node": n["node_id"] + 1, "bbox": n["bbox"], "no_lymphoid_rim": bool(n.get("pale")),
                   "metastasis": by_id.get(n["node_id"], {}).get("present"),
                   "judged_non_nodal": bool(by_id.get(n["node_id"], {}).get("not_a_node")),
                   "description": by_id.get(n["node_id"], {}).get("description")}
                  for n in nodes],
        "evidence": ev,
    }
