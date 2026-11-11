"""Glioma — locate the WHO grade-defining features.

Two magnifications, because the features live at two scales: necrosis and the
infiltrating margin are regional and visible on the whole-section overview;
microvascular proliferation and pseudopalisading are smaller than one visual
token at that scale and only become readable in ~2.3 mm fields at ~3 µm/px.
"""
from pathcowork.analysis import common_skills as common
from pathcowork.analysis import gbm_skills as gbm
from pathcowork.analysis import skin_skills            # drop_debris is organ-agnostic
from pathcowork.headless import evidence

PROJECT = "gbm"


async def run(ctx) -> dict:
    # 1. Tissue on the thumbnail; fragments under a tenth of the largest piece are debris.
    tissues = skin_skills.drop_debris(await common.segment_tissue())
    if not tissues:
        return {"finding": ("GBM features", "No usable tissue on this slide."), "evidence": []}

    # 2. Overview read (~8 µm/px): necrosis zones and the infiltrating margin.
    zones = await gbm.read_overview_zones(tissues)

    # 3. High-power fields (2.3 mm at ~3 µm/px) placed on the necrosis rims —
    #    where microvascular proliferation and pseudopalisading occur.
    fields = gbm.plan_tiles(zones, tissues)
    foci = await gbm.read_tiles(fields)

    # 4. Script protocol, by effort tier. Above "fast": a pseudopalisade must sit on
    #    necrosis confirmed in its own field; a focus on a fold or bubble rim is
    #    voided; an overview with no necrosis earns a systematic high-power survey.
    tier = gbm.tier_from_sample_max(ctx.sample_max)
    notes = []
    if tier != "fast":
        zones, foci, notes = await gbm.apply_script_protocol(zones, fields, foci, tier)

    # 5. The finding, and the grade-IV rule used in the slide-level evaluation.
    def has(sub):
        return any(sub in l.lower() for f in foci for l in f["labels"])

    present = {
        "necrosis": any(z["concept"] == "necrosis" for z in zones),
        "infiltrating_margin": any(z["concept"] == "infiltrating" for z in zones),
        "microvascular_proliferation": has("microvascular"),
        "pseudopalisading": has("palisad"),
    }
    grade_iv = present["microvascular_proliferation"] or (
        present["necrosis"] and present["pseudopalisading"])
    return {
        "finding": gbm.finding(zones, foci),
        "features": present,
        "grade_call": "grade IV features present" if grade_iv else "no grade IV features",
        "effort": tier,
        "fields_examined": len(fields),
        "notes": notes,
        "evidence": ([evidence(z["concept"], z["polygon"], scale="overview") for z in zones]
                     + [evidence(" · ".join(f["labels"]), f["polygon"], scale="2.3 mm field") for f in foci]),
    }
