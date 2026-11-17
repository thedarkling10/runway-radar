import os
import psutil

def rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1024**2


print(f"at start (before importing anything): {rss_mb():.0f} MB")

import torch 
print(f"after torch import: {rss_mb():.0f} MB")

import open_clip
print(f"after clip import: {rss_mb():.0f} MB")

from runway_radar.attributes.clip_extractor import load_model
model, tokenizer, preprocess = load_model()
print(f"after load_model(): {rss_mb():.0f} MB")

print([name for name, _ in model.named_children()])
for name, child in model.named_children():
    n = sum(p.numel() for p in child.parameters())
    print(f"{name}: {n/1e6:.1f}M params = {n*4/1024**2:.0f} MB")

from runway_radar.attributes.clip_extractor import build_label_embeddings
build_label_embeddings(model, tokenizer)
print(f"after build_label_embeddings(): {rss_mb():.0f} MB")

# after encoding one image 

from PIL import Image
from runway_radar.attributes.image_utils import crop_letterbox

image = crop_letterbox(Image.open("data/raw/fall_2020/womens/look_003.avif").convert("RGB"))
tensor = preprocess(image).unsqueeze(0)
with torch.no_grad():
    embed = model.encode_image(tensor)
print(f"after encoding one image: {rss_mb():.0f} MB")

# results:
# at start (before importing anything): 17 MB
# after torch import: 192 MB
# after clip import: 335 MB
# after load_model(): 944 MB
# ['visual', 'transformer', 'token_embedding', 'ln_final']
# visual: 87.8M params = 335 MB
# transformer: 37.8M params = 144 MB
# token_embedding: 25.3M params = 96 MB
# ln_final: 0.0M params = 0 MB
# after build_label_embeddings(): 970 MB
# after encoding one image: 980 MB

# we see quite a lot of memory used, therefore loading the model at each request is not a quite a good idea
# even loading it once at startup won't fit if using a 512MB host
# i'll try to shrink the footprint in order to fit a smaller host