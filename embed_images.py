import pandas as pd
import torch
from tqdm import tqdm
import clip
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import os
from concurrent.futures import ThreadPoolExecutor

# ----------------------------
# Config
# ----------------------------
BATCH_SIZE = 128          # increase if memory allows
CACHE_DIR = "cached_images"
INTERIM_SAVE = 10000      # save embeddings every 10k images

# ----------------------------
# Prepare environment
# ----------------------------
os.makedirs(CACHE_DIR, exist_ok=True)

# Load preprocessed data
df = pd.read_parquet("preprocessed.parquet")
image_links = df['image_link'].tolist()

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)

# Load CLIP model
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

# ----------------------------
# Helper function to load & preprocess image
# ----------------------------
def load_preprocess(url_idx):
    url, idx = url_idx
    cache_path = f"{CACHE_DIR}/{idx}.jpg"
    if os.path.exists(cache_path):
        img = Image.open(cache_path).convert("RGB")
    else:
        try:
            response = requests.get(url, timeout=10)
            img = Image.open(BytesIO(response.content)).convert("RGB")
            img.save(cache_path)
        except:
            img = Image.new("RGB", (224, 224), color=(0,0,0))
    return preprocess(img)

# ----------------------------
# Generate embeddings
# ----------------------------
all_embeddings = []

for i in tqdm(range(0, len(image_links), BATCH_SIZE)):
    batch_links = image_links[i:i+BATCH_SIZE]

    # Threaded loading + preprocessing
    with ThreadPoolExecutor(max_workers=16) as executor:
        images = list(executor.map(load_preprocess, [(url, i+idx) for idx, url in enumerate(batch_links)]))

    images = torch.stack(images).to(device)

    # Encode with mixed precision
    with torch.no_grad():
        with torch.autocast(device_type='mps', dtype=torch.float16):
            image_emb = model.encode_image(images)
            image_emb /= image_emb.norm(dim=-1, keepdim=True)  # normalize
            all_embeddings.append(image_emb.cpu().numpy())

    # Interim save
    if (i + BATCH_SIZE) % INTERIM_SAVE == 0:
        np.save(f"image_embeddings_partial_{i}.npy", np.vstack(all_embeddings))
        print(f"Saved interim embeddings at index {i}")

# Final save
image_embeddings = np.vstack(all_embeddings)
np.save("image_embeddings.npy", image_embeddings)
print("Saved final image embeddings:", image_embeddings.shape)
