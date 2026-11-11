# Install the grounder and connect the five skills

Give this document to Claude in the checkout of your WSI agent or
[TissueLab](https://github.com/zhihuanglab/TissueLab). The five folders in
[`skills/`](skills/) provide the decision workflows, explanations and parameters.

`skill.py` preserves the ordered analysis steps. Its `pathcowork.*` calls refer
to operations supplied by the original application; that application library
is not distributed here. Claude should map these operations to your agent's
slide reader, field planner, model clients and report writer, or implement the
missing adapters in your application. The public configs use the released
grounder and are inputs to that integration. They are not configs for importing
the original application unchanged.

## 1. Download and serve PathCowork-Grounder

For local use, download both files from
[PathCowork-Grounder-27B](https://huggingface.co/zhihuanglab/pathcowork-grounder-27B-GGUF/tree/main):

- `pathcowork-grounder-27B-Q4_K_M.gguf`
- `mmproj-pathcowork-grounder-27B-f16.gguf`

Load the model and its matching vision projector in LM Studio, using its
llama.cpp runtime. Start the local server in LM Studio's Developer tab. The
reference setup is macOS, LM Studio 0.4.21+2 and llama.cpp Metal runtime 2.62.1
on an Apple M5 Pro Mac with 64 GB unified memory. See the main
[system requirements](README.md#system-requirements-and-tested-environments).
The two Q4 files occupy approximately 17.74 GB together; inference uses
additional memory. Backend installation is estimated at 10 minutes with
prerequisites installed, excluding model downloads and agent integration.

Use the server address displayed by LM Studio. For its usual local address:

```bash
export PATHCOWORK_GROUNDER_BASE_URL=http://127.0.0.1:1234/v1
# Use your server token if authentication is enabled; otherwise leave empty.
export PATHCOWORK_GROUNDER_API_KEY=""
curl "$PATHCOWORK_GROUNDER_BASE_URL/models" \
  -H "Authorization: Bearer $PATHCOWORK_GROUNDER_API_KEY"
```

Set `PATHCOWORK_GROUNDER_MODEL` to the identifier of the loaded grounder from
that response. A repository name or filename is not necessarily the server's
model identifier.

[PathCowork-Grounder-Pro-1T](https://huggingface.co/zhihuanglab/pathcowork-grounder-pro-1T)
is an alternative for server deployments. It is a LoRA adapter for
`moonshotai/Kimi-K2.6`; serve the base model with that adapter enabled, then
configure the endpoint and model identifier accordingly. The
[upstream deployment guide](https://huggingface.co/moonshotai/Kimi-K2.6/blob/main/docs/deploy_guidance.md)
provides an eight-H200 reference configuration. A base-model-only endpoint does
not serve the PathCowork adapter.

## 2. Check one image request

The following example uses only Python's standard library. Save it as
`check_grounder.py` in your application. Supply a JPEG H&E crop and its actual
resolution in micrometres per pixel **after resizing**. The model accepts one
image and a scale-aware localization prompt and returns labelled shapes.

```python
import base64
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

image_path, organ, um_per_px, field_mm, concept = sys.argv[1:]
prompt = (
    f"H&E-stained {organ} section, {float(um_per_px):.3g} um/px, "
    f"field {float(field_mm):.2f} mm wide. "
    f"Find every one of the {concept} in this image and outline each."
)
image = base64.b64encode(Path(image_path).read_bytes()).decode()
payload = {
    "model": os.environ["PATHCOWORK_GROUNDER_MODEL"],
    "temperature": 0,
    "max_tokens": 2048,
    "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + image}},
        {"type": "text", "text": prompt},
    ]}],
}
headers = {"Content-Type": "application/json"}
key = os.environ.get("PATHCOWORK_GROUNDER_API_KEY")
if key:
    headers["Authorization"] = "Bearer " + key
base = os.environ["PATHCOWORK_GROUNDER_BASE_URL"].rstrip("/")
request = urllib.request.Request(
    base + "/chat/completions", data=json.dumps(payload).encode(), headers=headers,
)
with urllib.request.urlopen(request, timeout=300) as response:
    completion = json.load(response)
if completion["choices"][0].get("finish_reason") == "length":
    raise RuntimeError("Grounder response was truncated")
raw = completion["choices"][0]["message"]["content"].strip()
if raw.lower().rstrip(".") == "there are none":
    shapes = []
else:
    block = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", raw, flags=re.S | re.I)
    shapes = json.loads(block.group(1) if block else raw)
    if not isinstance(shapes, list):
        raise ValueError("Expected a JSON list of shapes")
    for shape in shapes:
        points = shape.get("polygon_2d") or shape.get("polyline_2d")
        minimum = 3 if "polygon_2d" in shape else 2
        if not points or len(points) < minimum:
            raise ValueError("Missing or incomplete shape")
        if any(len(p) != 2 or any(not isinstance(v, (int, float)) or not 0 <= v <= 1000
                                 for v in p) for p in points):
            raise ValueError("Invalid shape coordinates")
print(json.dumps({"raw_response": raw, "shapes": shapes}, indent=2))
```

For a **1000-pixel-wide crop at 1 micrometre/pixel**, for example:

```bash
python check_grounder.py skin_crop.jpg skin 1 1.0 \
  "junctional melanocytic nests" > grounder_result.json
```

Expected output is a JSON object containing the raw model response and a list
of `polygon_2d` or `polyline_2d` shapes with labels. A valid absent finding gives
an empty list. Connection errors, malformed responses and missing cache entries
must remain errors or unavailable results, rather than becoming negative findings.
Examples of prompts and responses are in the
[model card](https://huggingface.co/zhihuanglab/pathcowork-grounder-27B-GGUF).
The API uses LM Studio's
[Chat Completions](https://lmstudio.ai/docs/developer/openai-compat/chat-completions)
and [List Models](https://lmstudio.ai/docs/developer/openai-compat/models) endpoints.

## 3. Connect the skills to the agent

Paste the following instruction into Claude in your application checkout:

> Read this document, the selected `skills/<name>/SKILL.md`, its `config.yaml`,
> and `skill.py`. Implement the decision sequence in my existing WSI agent.
> Reuse the application's slide reader, tissue mask, crop extraction, model
> clients and report writer. Map the named `pathcowork.*` helper calls to these
> facilities; implement missing helpers locally and update the imports.
> Use the public grounder through the endpoint above. Load prompts and sampling
> parameters from the selected config. Keep the order of analysis and the field
> sizes in `skill.py`; use actual slide resolution when converting micrometres
> to pixels. Return structured JSON with findings and evidence in level-0 slide
> coordinates. Keep request failures and unavailable model components explicit.
> First verify the one-image grounder request, then run one skill on my slide
> and inspect its evidence overlays and report.

The host adapter supplies a run context containing `slide`, `slide_id`, `mpp`,
`protocol` (the loaded YAML) and `sample_max` (the effort budget). The helper
names in the Python files define the requested operations:

| Skill | Operations to connect |
|---|---|
| Prostate | Segment cores; ground candidate regions; allocate 320 um patches inside and outside them; classify patches; aggregate malignant patterns |
| Skin | Segment pieces; trace the junction; sample 1.5 mm fields along it; ground each requested concept; merge overlapping findings; run the host mitosis detector |
| Breast | Ground invasive carcinoma; sample tubule and nuclear fields; obtain component reads; find the mitotic hotspot; combine the available component scores |
| Glioma | Ground overview features; plan 2.328 mm fields; ground high-power features; apply the selected checks and, for `max`, the host's second-opinion model |
| Lymph node | Find nodes and tissue contours; screen whole nodes and zoom fields; apply the selected colon/breast vote parameters; retain metastatic-focus outlines |

Grounder localization supplies regions. Prostate patch grading and breast
tubule/nuclear scoring use a separate general multimodal model supplied by the
host agent. Configure its model ID and credentials using the `classifier`
environment-variable names in the YAML. Skin and breast mitosis counting needs
the host's mitosis detector; the grounder weights do not include that detector.
If a required component is missing, return an incomplete result with a null
score. Do not infer a zero mitotic count or a complete grade from a failed call.

For lymph nodes, select `variants.colon` or `variants.breast` from its single
config. For glioma `max`, require a working second-opinion model or explicitly
record that the requested tier could not be completed. The extra concept
prompts in these recipes can go beyond the model card's listed training targets;
validate their behaviour on your intended data.

### Coordinates and output

Grounder coordinates span **0-1000** across the image sent to the model. For a
level-0 crop with origin `(x0, y0)`, width `w0` and height `h0`:

```python
x_level0 = x0 + x_model / 1000.0 * w0
y_level0 = y0 + y_model / 1000.0 * h0
```

Include white padding in that transform if the crop is padded before inference.
Keep junctions as open polylines. Do not close them into polygons. Use the crop's
physical width divided by its rendered pixel width for the prompt's resolution.

Write one JSON report per slide with `skill`, `slide`, `status`, `finding`,
skill-specific results and `evidence`. Region evidence has a label, level-0
polygon or polyline, and a bounding box; mitosis evidence may be a centre point
with a score. Include model identifiers, parameters, runtime and any missing
components. The workflow's `finding` tuple becomes an object with `label` and
`body`. An `evidence(label, polygon, **extra)` adapter constructs these records.

The recorded skin-agent demonstration takes approximately 2-3 minutes after
setup. This is an illustration of the application workflow; a new integration's
runtime depends on its models, hardware and field count. These public-grounder
recipes describe an integration path, rather than an exact replay of the
paper's organ-specific model deployments and quantitative results.

Research use is governed by the repository [license](LICENSE).
