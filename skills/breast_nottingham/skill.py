"""Breast — Nottingham histologic grade of an invasive carcinoma.

Three components, each read at the magnification a pathologist reads it at:
tubule formation is a proportion of the whole tumour (low power, spread wide);
mitoses are counted in the most active 2 mm²; nuclear pleomorphism is judged at
high power in fields ranked by that mitotic hotspot.
"""
from pathcowork.analysis import breast_skills as breast
from pathcowork.analysis import common_skills as common
from pathcowork.analysis import skin_skills            # drop_debris is organ-agnostic
from pathcowork.headless import evidence

PROJECT = "breast"


async def _collect(stream) -> list:
    return [r async for r in stream]


async def run(ctx) -> dict:
    sampling = ctx.protocol.get("sampling", {})

    # 1. Tissue, then the invasive tumour: one outline per focus, per tissue piece.
    tissues = skin_skills.drop_debris(await common.segment_tissue())
    if not tissues:
        return {"finding": ("Nottingham grade", "No usable tissue on this slide."), "evidence": []}
    regions = await breast.propose_tumor_regions(tissues)
    if not regions:
        return {"finding": ("Nottingham grade", "No invasive tumour localized — nothing to grade."),
                "evidence": []}
    _, tumor = breast.combine_tumor(regions)        # the grade is read across ALL foci
    tubule_n, pleo_n = breast.field_budget()

    # 2. Tubule formation — 1.4 mm fields spread over the whole tumour. The score is
    #    one judgement over all the fields at once (it is a whole-tumour proportion);
    #    the per-field reads are the fallback if that call fails.
    tub_fields = breast.plan_fields(regions, tissues, float(sampling.get("tubule_field_um", 1400)),
                                    tubule_n, "tubule")
    await breast.prepare_fields(tub_fields)
    tub_multi = await breast.read_tubules_multi(tub_fields)
    tub_reads = await _collect(breast.read_tubules_stream(tub_fields))
    tubule = breast.score_tubules(tub_multi, tub_reads)

    # 3. Mitotic count — detector over every tumour focus; score the densest 2 mm².
    mitoses = await common.detect_mitoses(regions)
    mito = breast.mitotic_score(mitoses)

    # 4. Nuclear pleomorphism — 240 µm fields ranked by distance to the mitotic
    #    hotspot and kept apart from each other; fields with no carcinoma abstain.
    hotspot = breast.mitosis_hotspot(mitoses) or tumor.get("hotspot")
    pleo_fields = breast.plan_fields(regions, tissues, float(sampling.get("pleo_field_um", 240)),
                                     pleo_n, "pleo", around=hotspot)
    await breast.prepare_fields(pleo_fields)
    pleo_multi = await breast.read_pleomorphism_multi(pleo_fields)
    pleo_reads = await _collect(breast.read_pleomorphism_stream(pleo_fields))
    pleo = breast.score_pleomorphism(pleo_multi, pleo_reads)

    # 5. Sum the three 1–3 scores. A component that could not be read is named,
    #    never guessed — the grade is then reported as unavailable.
    result = breast.nottingham_grade(tubule, pleo, mito)

    def box(p):
        (cx, cy), h = p["center"], p["size_px"] // 2
        return [[cx - h, cy - h], [cx + h, cy - h], [cx + h, cy + h], [cx - h, cy + h]]

    return {
        "finding": breast.finding(result),
        "grade": result.get("grade"),
        "components": {k: result.get(k) for k in ("tubule", "pleomorphism", "mitosis")},
        "tubule_percent": result.get("tubule_percent"),
        "mitoses_in_hotspot": result.get("mitosis_hotspot"),
        "mitoses_total": result.get("mitosis_total"),
        "evidence": ([evidence("invasive carcinoma", r["polygon"]) for r in regions]
                     + [evidence("tubule field", box(p)) for p in tub_fields]
                     + [evidence("nuclear field", box(p)) for p in pleo_fields]),
    }
