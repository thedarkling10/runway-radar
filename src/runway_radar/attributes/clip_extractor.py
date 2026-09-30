"""CLIP zero-shot attribute extraction for Gucci runway looks.

Uses prompt-ensembled zero-shot classification (see vocab.py) rather than
fine-tuning: with ~200 images and no labels, fine-tuning would consume the
entire dataset just to train, leaving nothing held out to validate against.

Preprocessing pads each image to a square before resizing, instead of
CLIP's default resize-then-center-crop. Runway photos here are tall and
narrow (roughly 2:3 to 1:3); a center crop would cut off heads and/or
feet, destroying exactly the neckline/hemline/footwear signal these
attributes depend on.
"""

from __future__ import annotations

import csv
from pathlib import Path

import open_clip
import pillow_avif  # noqa: F401 -- registers the AVIF codec with Pillow
import torch
from PIL import Image, ImageOps
from torchvision import transforms

from runway_radar.acquisition.register import RAW_DIR, read_metadata, season_slug
from runway_radar.attributes.image_utils import crop_letterbox
from runway_radar.attributes.vocab import ATTRIBUTE_AXES, build_prompts, iter_labels

# "-quickgelu" matches the activation function the original OpenAI weights
# were trained with; the plain "ViT-B-32" config defaults to standard GELU
# and open_clip will warn about a mismatch (and predictions can be worse).
MODEL_NAME = "ViT-B-32-quickgelu"
PRETRAINED = "openai"

OUTPUT_CSV = Path(__file__).resolve().parents[3] / "data" / "processed" / "attributes.csv"
PAD_FILL = (128, 128, 128)  # neutral gray letterbox


def pad_to_square(image: Image.Image) -> Image.Image:
    size = max(image.size)
    return ImageOps.pad(image, (size, size), color=PAD_FILL, centering=(0.5, 0.5))


def build_preprocess(image_size: int, mean, std):
    return transforms.Compose(
        [
            transforms.Lambda(pad_to_square),
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

def load_model(device: str = "cpu"):
    model, _, _ = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
    model.eval().to(device)
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)

    image_size = model.visual.image_size
    if isinstance(image_size, (tuple, list)):
        image_size = image_size[0]
    preprocess = build_preprocess(image_size, open_clip.OPENAI_DATASET_MEAN, open_clip.OPENAI_DATASET_STD)
    return model, tokenizer, preprocess

# text encoder
@torch.no_grad()
def build_label_embeddings(model, tokenizer, device: str = "cpu") -> dict[str, dict[str, torch.Tensor]]:
    """For each attribute axis, embed every label as the L2-normalized
    average of its ensembled prompt embeddings (prompt ensembling)."""
    label_embeddings: dict[str, dict[str, torch.Tensor]] = {}
    for axis in ATTRIBUTE_AXES:
        label_embeddings[axis] = {}
        for label, phrasings in iter_labels(axis):
            prompts = [p for phrasing in phrasings for p in build_prompts(phrasing)]
            tokens = tokenizer(prompts).to(device)
            embeds = model.encode_text(tokens)
            embeds = embeds / embeds.norm(dim=-1, keepdim=True)
            mean_embed = embeds.mean(dim=0)
            label_embeddings[axis][label] = mean_embed / mean_embed.norm()
    return label_embeddings


@torch.no_grad()
def compute_image_embeddings(rows, model, preprocess, device: str = "cpu") -> dict[str, torch.Tensor]:
    """Embed every look once, up front. Calibration needs every image's
    embedding available before any label can be picked, so this replaces
    the old one-image-at-a-time encoding inside classify_image."""
    image_embeddings: dict[str, torch.Tensor] = {}
    for row in rows:
        path = RAW_DIR / season_slug(row["season"], row["year"]) / row["line"] / row["image_filename"]
        image = crop_letterbox(Image.open(path).convert("RGB"))
        tensor = preprocess(image).unsqueeze(0).to(device)
        embed = model.encode_image(tensor)
        embed = embed / embed.norm(dim=-1, keepdim=True)
        image_embeddings[row["look_id"]] = embed.squeeze(0)
    return image_embeddings


def compute_label_baselines(
    image_embeddings: dict[str, torch.Tensor], label_embeddings: dict[str, dict[str, torch.Tensor]]
) -> dict[str, dict[str, float]]:
    """For each axis, each label's baseline is its average raw similarity
    across the WHOLE dataset -- how close it sits to a typical image,
    regardless of content. Subtracting this later is what corrects the
    hubness problem (see docs/problems_and_fixes.md): some labels sit
    generically close to almost every image in a stylistically narrow
    dataset like ours, and would otherwise win by default."""
    image_matrix = torch.stack(list(image_embeddings.values()))  # (N, 512)

    label_baselines: dict[str, dict[str, float]] = {}
    for axis, labels in label_embeddings.items():
        label_names = list(labels.keys())
        label_matrix = torch.stack([labels[name] for name in label_names])  # (K, 512)
        sims = image_matrix @ label_matrix.T  # (N, K) -- every image x every label
        baseline_per_label = sims.mean(dim=0)  # (K,) -- averaged down each column
        label_baselines[axis] = {
            name: baseline_per_label[i].item() for i, name in enumerate(label_names)
        }
    return label_baselines


# image encoder using ViT design (vision transformers)
def classify_image(
    image_embed: torch.Tensor,
    model,
    label_embeddings: dict[str, dict[str, torch.Tensor]],
    label_baselines: dict[str, dict[str, float]],
) -> dict[str, tuple[str, float]]:
    logit_scale = model.logit_scale.exp()

    results = {}
    for axis, labels in label_embeddings.items():
        label_names = list(labels.keys())
        label_matrix = torch.stack([labels[name] for name in label_names])
        sims = image_embed @ label_matrix.T
        baseline = torch.tensor([label_baselines[axis][name] for name in label_names])
        adjusted = sims - baseline
        probs = (adjusted * logit_scale).softmax(dim=0)
        best_idx = probs.argmax().item()
        results[axis] = (label_names[best_idx], round(probs[best_idx].item(), 4))
    return results


def run_extraction():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {MODEL_NAME} ({PRETRAINED}) on {device}...")
    model, tokenizer, preprocess = load_model(device)

    print("Building label embeddings...")
    label_embeddings = build_label_embeddings(model, tokenizer, device)

    rows = read_metadata()
    print(f"Embedding {len(rows)} looks...")
    image_embeddings = compute_image_embeddings(rows, model, preprocess, device)

    print("Computing per-label calibration baselines...")
    label_baselines = compute_label_baselines(image_embeddings, label_embeddings)

    print(f"Classifying {len(rows)} looks...")
    output_rows = []
    for i, row in enumerate(rows, start=1):
        image_embed = image_embeddings[row["look_id"]]
        predictions = classify_image(image_embed, model, label_embeddings, label_baselines)
        for axis, (label, confidence) in predictions.items():
            output_rows.append(
                {
                    "look_id": row["look_id"],
                    "axis": axis,
                    "predicted_label": label,
                    "confidence": confidence,
                }
            )
        if i % 25 == 0 or i == len(rows):
            print(f"  {i}/{len(rows)}")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["look_id", "axis", "predicted_label", "confidence"])
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Wrote {len(output_rows)} predictions to {OUTPUT_CSV}")


if __name__ == "__main__":
    run_extraction()
