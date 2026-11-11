"""Skin — melanocytic lesion: read the dermo-epidermal junction, then count mitoses.

A dermatopathologist reads a melanocytic lesion from the junction outward, so the
skill first traces the junction on every tissue piece, then asks six junctional
and dermal concepts along it, and finally counts mitotic figures.
"""
from pathcowork.analysis import common_skills as common
from pathcowork.analysis import skin_skills as skin
from pathcowork.headless import evidence

PROJECT = "dermpath"


async def run(ctx) -> dict:
    # 1. Tissue pieces on the thumbnail; debris dropped.
    tissues = skin.drop_debris(await common.segment_tissue())
    if not tissues:
        return {"finding": ("Impression", "No usable tissue on this slide."), "evidence": []}

    # 2. Trace the dermo-epidermal junction of each piece (one ~6 mm field per piece).
    dej = await skin.dermoepidermal_junctions(tissues)

    # 3. Walk each junction in 1.5 mm fields. Every field is asked each concept as
    #    its own request (single-concept grounding is what the model was trained
    #    on); hits under the confidence floor are dropped and overlapping hits are
    #    merged into one focus that carries every matching concept.
    hits = []
    for tissue in tissues:
        fields = skin.sample_along_dej(tissue, dej.get(tissue["tissue_id"]))
        if fields:
            hits += [(tissue, focus) for focus in await skin.read_junction(fields)]

    # 4. Mitotic figures, over each piece and over its dermis (the larger side the
    #    junction splits the piece into); the two passes overlap, so merge by centre.
    dermis = [skin.dermis_side(t, dej.get(t["tissue_id"])) for t in tissues]
    mitoses = common.merge_mitoses(
        await common.detect_mitoses(tissues) + await common.detect_mitoses(dermis))

    concepts = sorted({l for _, f in hits for l in f.get("labels", [])})
    return {
        "finding": skin.finding(hits, mitoses),
        "junctions_traced": sum(1 for t in tissues if dej.get(t["tissue_id"])),
        "junctional_foci": len(hits),
        "concepts": concepts,
        "mitotic_figures": len(mitoses),
        "evidence": ([evidence(" · ".join(f.get("labels", [])), f["polygon"],
                               confidence=f.get("confidence")) for _, f in hits]
                     + [{"label": "mitotic figure (candidate)", "center": [int(m["center"][0]), int(m["center"][1])],
                         "score": round(float(m.get("score", 0)), 3)} for m in mitoses]),
    }
