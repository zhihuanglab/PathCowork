"""Prostate needle biopsy — Gleason score.

Low power decides where to look; high power decides what it is. The overview
model only proposes regions worth examining — it never grades. Grading is done
patch by patch at high power, inside those regions and, as a contrast, in the
surrounding tissue.
"""
from collections import Counter

from pathcowork.analysis import common_skills as common
from pathcowork.analysis import prostate_skills as prostate
from pathcowork.headless import evidence
from pathcowork.tools.classifier import argmax_class

PROJECT = "prostate"


async def run(ctx) -> dict:
    # 1. The cores on the thumbnail.
    cores = await common.segment_tissue()
    if not cores:
        return {"finding": ("Gleason", "No tissue detected on this slide."), "evidence": []}

    # 2. Low power: regions whose architecture deserves a closer look
    #    (clipped back onto tissue when the model overshoots a core).
    regions = await prostate.low_power_roi_proposal()

    # 3. The patch budget: 320 µm patches, 80 % inside the proposed regions (by
    #    area) and 20 % in the surrounding tissue. If the grader calls cancer
    #    everywhere, including the contrast patches, something is wrong.
    patches = prostate.plan_patch_budget(regions, cores)

    # 4. High power: grade each patch (benign / atypical / pattern 3 / 4 / 5).
    results = await prostate.classify_patches(patches)

    # 5. Aggregate: primary = the most frequent malignant pattern, secondary = the next.
    grade = prostate.compute_gleason(results)
    primary, secondary, worst = grade

    calls = Counter()
    per_region = {r["roi_id"]: Counter() for r in regions}
    for r in results:
        cls = argmax_class(r["preds"])[0]
        calls[cls] += 1
        if r["patch"].get("source") == "roi":
            per_region.setdefault(r["patch"].get("parent_roi_id"), Counter())[cls] += 1
    return {
        "finding": prostate.finding(grade),
        "gleason": {"primary": primary, "secondary": secondary, "score": primary + secondary}
                   if primary else None,
        "patches_graded": len(results),
        "patch_calls": dict(calls),
        "evidence": [evidence("; ".join(r.get("labels") or ["region of interest"]), r["polygon"],
                              patch_calls=dict(per_region.get(r["roi_id"], {})))
                     for r in regions],
    }
