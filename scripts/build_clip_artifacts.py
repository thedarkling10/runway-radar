from runway_radar.attributes.clip_extractor import load_model, build_label_embeddings, compute_image_embeddings, compute_label_baselines
from runway_radar.acquisition.register import read_metadata

import json
from runway_radar.api.db import REPO_ROOT

model, tokenizer, preprocess = load_model()
label_embeddings = build_label_embeddings(model, tokenizer)

rows = read_metadata()
image_embeddings = compute_image_embeddings(rows, model, preprocess)

baselines = compute_label_baselines(image_embeddings=image_embeddings, label_embeddings=label_embeddings)
logit_scale = model.logit_scale.exp().item()

axes = {}
for axis, labels in label_embeddings.items():
    axes[axis] = {}
    for label, tensor in labels.items():
        axes[axis][label] = {
            "embedding": tensor.tolist(),
            "baseline": baselines[axis][label],
        }

artifact = {"logit_scale": logit_scale, "axes":axes}

out_path = REPO_ROOT / "src" / "runway_radar" / "api" / "artifacts" / "clip_labels.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(artifact, f)
print(f"saved {sum(len(v) for v in axes.values())} labels to {out_path}")