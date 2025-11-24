import pandas as pd
import torch
from tqdm import tqdm
import clip
import numpy as np

# Load preprocessed CSV/Parquet
df = pd.read_parquet("preprocessed.parquet")

# Only take training + test combined
texts = df['content_clean'].tolist()

# Device (MPS for Mac Apple GPU, fallback to CPU)
device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)

# Load CLIP model
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

# Batch size for text embeddings
BATCH_SIZE = 256
all_embeddings = []

with torch.no_grad():
    for i in tqdm(range(0, len(texts), BATCH_SIZE)):
        batch_texts = texts[i:i+BATCH_SIZE]
        tokens = clip.tokenize(batch_texts, truncate=True).to(device)
        text_emb = model.encode_text(tokens)
        text_emb /= text_emb.norm(dim=-1, keepdim=True)  # normalize
        all_embeddings.append(text_emb.cpu().numpy())

# Concatenate all embeddings and save
text_embeddings = np.vstack(all_embeddings)
np.save("text_embeddings.npy", text_embeddings)
print("Saved text embeddings:", text_embeddings.shape)
