"""Attribute vocabulary for CLIP zero-shot classification of Gucci runway
looks, Alessandro Michele era (2015-2022).

Each axis is a dict of {label: [prompt phrasings]}. Multiple phrasings per
label are averaged at inference time (prompt ensembling) to reduce
sensitivity to exact wording -- a standard CLIP zero-shot robustness trick.

Labels were chosen to reflect the actual visual language of the Michele
era specifically (maximalism, 70s references, monogram prints, layering),
not a generic fashion vocabulary -- a generic vocabulary would blur exactly
the trends this project is trying to detect.

Color is intentionally NOT modeled here: CLIP is weak at precise color
naming, so dominant color palettes are extracted separately via k-means
pixel clustering (see color_palette.py).
"""

from __future__ import annotations

# Reusable prompt templates; each is filled with a label description.
PROMPT_TEMPLATES = [
    "a photo of a runway look with {}",
    "a fashion model wearing {}",
    "a high-fashion garment featuring {}",
    "a runway outfit that has {}",
]

SILHOUETTE = {
    "oversized_boxy": [
        "an oversized, boxy silhouette",
        "loose, voluminous tailoring",
    ],
    "romantic_bohemian": [
        "a flowing, romantic bohemian silhouette",
        "soft draping and loose flowing fabric",
    ],
    "seventies_flared": [
        "a 1970s-inspired flared silhouette",
        "flared trousers or a flared skirt",
    ],
    "structured_tailored": [
        "a structured, sharply tailored silhouette",
        "a fitted blazer or tailored suit",
    ],
    "layered_eclectic": [
        "an eclectic, heavily layered outfit",
        "mismatched layered clothing pieces worn together",
    ],
    "mini_length": [
        "a mini-length hemline",
        "a short skirt or short dress",
    ],
    "maxi_length": [
        "a maxi-length, floor-sweeping hemline",
        "a long flowing gown or long coat",
    ],
}

NECKLINE_COLLAR = {
    "pussy_bow": [
        "a pussy-bow blouse collar",
        "a blouse with a bow tied at the neck",
    ],
    "ruffled_high_collar": [
        "a ruffled high collar",
        "a high victorian-style collar",
    ],
    "peter_pan_collar": [
        "a Peter Pan collar",
        "a small rounded schoolgirl collar",
    ],
    "plunging_neckline": [
        "a deep plunging neckline",
        "a low-cut revealing neckline",
    ],
    "turtleneck": [
        "a turtleneck",
        "a high close-fitting neck covering the throat",
    ],
}

FABRIC_TEXTURE = {
    "velvet": [
        "velvet fabric",
        "a soft plush velvet texture",
    ],
    "sequined_metallic": [
        "sequined or metallic fabric",
        "a sparkling sequined garment",
    ],
    "faux_fur_trim": [
        "faux fur trim or a faux fur coat",
        "fluffy fur-like material",
    ],
    "brocade_jacquard": [
        "brocade or jacquard fabric",
        "a richly woven textured fabric",
    ],
    "lace": [
        "lace fabric",
        "a delicate lace overlay",
    ],
    "leather": [
        "leather material",
        "a leather jacket or leather trousers",
    ],
}

PATTERN = {
    "logo_monogram": [
        "a logo or monogram print",
        "a repeating brand logo pattern",
    ],
    "floral_print": [
        "a floral print",
        "a pattern of flowers",
    ],
    "animal_print": [
        "an animal print",
        "a leopard, snake, or tiger print pattern",
    ],
    "striped": [
        "a striped pattern",
        "horizontal or vertical stripes",
    ],
    "polka_dot": [
        "a polka dot pattern",
        "a pattern of round dots",
    ],
    "psychedelic_geometric": [
        "a psychedelic or geometric print",
        "a bold abstract geometric pattern",
    ],
}

EMBELLISHMENT = {
    "crystal_rhinestone": [
        "crystal or rhinestone embellishment",
        "sparkling jeweled decoration",
    ],
    "applique_patches": [
        "appliqué patches",
        "decorative fabric patches sewn onto the garment",
    ],
    "embroidery": [
        "embroidery",
        "intricate embroidered detailing",
    ],
    "pearls": [
        "pearl embellishment",
        "decorative pearls sewn onto the garment",
    ],
    "statement_bow": [
        "an oversized statement bow",
        "a large decorative bow",
    ],
    "layered_jewelry": [
        "layered statement jewelry",
        "multiple necklaces or brooches worn together",
    ],
}

ATTRIBUTE_AXES = {
    "silhouette": SILHOUETTE,
    "neckline_collar": NECKLINE_COLLAR,
    "fabric_texture": FABRIC_TEXTURE,
    "pattern": PATTERN,
    "embellishment": EMBELLISHMENT,
}


def iter_labels(axis: str):
    """Yield (label, [phrasings]) pairs for a given attribute axis."""
    yield from ATTRIBUTE_AXES[axis].items()


def build_prompts(phrasing: str) -> list[str]:
    """Expand a single label phrasing into all templated prompt variants."""
    return [template.format(phrasing) for template in PROMPT_TEMPLATES]
