#!/usr/bin/env python3
"""
train_full.py
Full multimodal stacking pipeline (Fusion MLP + LightGBM/XGBoost + Ridge meta-learner)
Handles missing features, GPU/MPS acceleration, and outputs SMAPE.
"""

import os, gc, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import lightgbm as lgb
import xgboost as xgb

# ----------------------------
# CONFIG
# ----------------------------
DATA_PARQUET = "preprocessed.parquet"
TEXT_EMB_FILE = "text_embeddings.npy"
IMG_EMB_FILE  = "image_embeddings.npy"

N_FOLDS = 5
RANDOM_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 128
MLP_EPOCHS = 12
MLP_LR = 1e-3
N_PCA_TEXT = 384   # Sentence-BERT embeddings
N_PCA_IMG  = 1280  # EfficientNet-B0 embeddings
TARGET_COL = "price"

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
print("DEVICE:", DEVICE)

# ----------------------------
# HELPERS
# ----------------------------
def safe_smape(y_true, y_pred, eps=1e-6):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom = np.where(denom < eps, eps, denom)
    return 100.0 * np.mean(np.abs(y_pred - y_true) / denom)

def free_mem():
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

# ----------------------------
# 1) Load data & embeddings
# ----------------------------
print("Loading data...")
df = pd.read_parquet(DATA_PARQUET)
text_emb = np.load(TEXT_EMB_FILE).astype(np.float32)
img_emb  = np.load(IMG_EMB_FILE).astype(np.float32)

train_mask = df['is_test']==0
test_mask  = df['is_test']==1
sample_ids_test = df.loc[test_mask, 'sample_id'].values
y_raw = df.loc[train_mask, TARGET_COL].values.astype(np.float32)

# ----------------------------
# 2) Text numeric features
# ----------------------------
text_numeric_cols = [
    'content_length', 'word_count', 'avg_word_length',
    'max_quantity', 'quantity_count', 'price_keyword_count', 'brand_count'
]

# Fill missing columns with zeros
for c in text_numeric_cols:
    if c not in df.columns:
        print(f"Column '{c}' not found, filling with zeros.")
        df[c] = 0.0

text_numeric_feats = np.column_stack([df[c].fillna(0) for c in text_numeric_cols]).astype(np.float32)

# ----------------------------
# 3) Normalize embeddings + PCA
# ----------------------------
def row_norm(a): return a / (np.linalg.norm(a, axis=1, keepdims=True)+1e-8)
text_emb = row_norm(text_emb)
img_emb  = row_norm(img_emb)

pca_text = PCA(n_components=min(N_PCA_TEXT,text_emb.shape[1]), random_state=RANDOM_SEED)
pca_img  = PCA(n_components=min(N_PCA_IMG,img_emb.shape[1]), random_state=RANDOM_SEED)
pca_text.fit(text_emb[train_mask.values])
pca_img.fit(img_emb[train_mask.values])
text_pca = pca_text.transform(text_emb)
img_pca  = pca_img.transform(img_emb)

joblib.dump(pca_text,"pca_text.pkl")
joblib.dump(pca_img,"pca_img.pkl")

# ----------------------------
# 4) Combine features
# ----------------------------
X_all = np.hstack([text_pca, img_pca, text_numeric_feats])
print("Combined feature shape:", X_all.shape)

X_train = X_all[train_mask.values]
X_test  = X_all[test_mask.values]
y_train = np.log1p(y_raw)

# impute + scale
imp = SimpleImputer(strategy="median")
scaler = StandardScaler()
X_train = scaler.fit_transform(imp.fit_transform(X_train))
X_test  = scaler.transform(imp.transform(X_test))
joblib.dump(imp,"imputer.pkl")
joblib.dump(scaler,"scaler.pkl")

# ----------------------------
# 5) Fusion MLP
# ----------------------------
class Subnet(nn.Module):
    def __init__(self, in_dim, hidden=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden//2),
            nn.ReLU()
        )
    def forward(self,x): return self.net(x)

class FusionMLP(nn.Module):
    def __init__(self,text_dim,img_dim,num_dim,hidden=128):
        super().__init__()
        self.text_net = Subnet(text_dim, hidden=hidden)
        self.img_net  = Subnet(img_dim, hidden=hidden)
        self.num_net  = Subnet(num_dim, hidden=hidden//2)
        fused_dim = (hidden//2) + (hidden//2) + (hidden//4)
        self.head = nn.Sequential(
            nn.Linear(fused_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden,1)
        )
    def forward(self,t,im,n):
        t_out = self.text_net(t)
        im_out = self.img_net(im)
        n_out = self.num_net(n)
        fused = torch.cat([t_out,im_out,n_out],dim=1)
        return self.head(fused).squeeze(-1)

t_dim, i_dim, n_dim = text_pca.shape[1], img_pca.shape[1], text_numeric_feats.shape[1]

X_text_all = text_pca.astype(np.float32)
X_img_all  = img_pca.astype(np.float32)
X_num_all  = text_numeric_feats.astype(np.float32)

oof_mlp_log = np.zeros(len(X_train),dtype=np.float32)
test_preds_mlp_log_folds = []

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

for fold,(tr_idx,val_idx) in enumerate(kf.split(X_train)):
    print(f"MLP Fold {fold+1}/{N_FOLDS}")
    
    t_tr = torch.from_numpy(X_text_all[tr_idx]).float().to(DEVICE)
    im_tr = torch.from_numpy(X_img_all[tr_idx]).float().to(DEVICE)
    n_tr = torch.from_numpy(X_num_all[tr_idx]).float().to(DEVICE)
    y_tr = torch.from_numpy(y_train[tr_idx]).float().to(DEVICE)

    t_val = torch.from_numpy(X_text_all[val_idx]).float().to(DEVICE)
    im_val = torch.from_numpy(X_img_all[val_idx]).float().to(DEVICE)
    n_val = torch.from_numpy(X_num_all[val_idx]).float().to(DEVICE)
    y_val = torch.from_numpy(y_train[val_idx]).float().to(DEVICE)

    train_ds = TensorDataset(t_tr,im_tr,n_tr,y_tr)
    val_ds   = TensorDataset(t_val,im_val,n_val,y_val)
    train_loader = DataLoader(train_ds,batch_size=BATCH_SIZE,shuffle=True)
    val_loader   = DataLoader(val_ds,batch_size=BATCH_SIZE,shuffle=False)

    model = FusionMLP(t_dim,i_dim,n_dim).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(),lr=MLP_LR,weight_decay=1e-5)
    criterion = nn.L1Loss()

    best_val_loss = 1e9
    best_state = None
    for ep in range(MLP_EPOCHS):
        model.train()
        running,count = 0.0,0
        for xb_t,xb_im,xb_n,yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb_t,xb_im,xb_n), yb)
            loss.backward()
            optimizer.step()
            running += loss.item()*xb_t.size(0)
            count += xb_t.size(0)
        avg_train = running/max(1,count)

        model.eval()
        v_running,v_count=0.0,0
        with torch.no_grad():
            for xb_t,xb_im,xb_n,yb in val_loader:
                v_running += criterion(model(xb_t,xb_im,xb_n), yb).item()*xb_t.size(0)
                v_count += xb_t.size(0)
        avg_val = v_running/max(1,v_count)
        print(f" Epoch {ep+1}/{MLP_EPOCHS} train_loss={avg_train:.4f} val_loss={avg_val:.4f}")
        if avg_val<best_val_loss:
            best_val_loss=avg_val
            best_state=model.state_dict()
    model.load_state_dict(best_state)
    model.eval()

    # OOF
    with torch.no_grad():
        oof_mlp_log[val_idx] = model(t_val,im_val,n_val).cpu().numpy()

    # test fold
    with torch.no_grad():
        t_test = torch.from_numpy(X_text_all[test_mask.values]).float().to(DEVICE)
        im_test = torch.from_numpy(X_img_all[test_mask.values]).float().to(DEVICE)
        n_test = torch.from_numpy(X_num_all[test_mask.values]).float().to(DEVICE)
        test_preds_mlp_log_folds.append(model(t_test,im_test,n_test).cpu().numpy())
    free_mem()

test_preds_mlp_log = np.mean(test_preds_mlp_log_folds,axis=0)
print("OOF MLP SMAPE:", safe_smape(np.expm1(y_train), np.expm1(oof_mlp_log)))
joblib.dump(oof_mlp_log,"oof_mlp_log.pkl")
joblib.dump(test_preds_mlp_log,"test_preds_mlp_log.pkl")

# ----------------------------
# 6) LightGBM
# ----------------------------
dtrain_full = lgb.Dataset(X_train,label=y_train)
lgb_model = lgb.LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=128,
    random_state=RANDOM_SEED
)
lgb_model.fit(X_train, y_train)
lgb_test_preds = lgb_model.predict(X_test)

# ----------------------------
# 7) XGBoost
# ----------------------------
xgb_model = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    tree_method="gpu_hist" if torch.cuda.is_available() else "hist",
    random_state=RANDOM_SEED
)
xgb_model.fit(X_train, y_train)
xgb_test_preds = xgb_model.predict(X_test)

# ----------------------------
# 8) Stacking / Meta-learner
# ----------------------------
stack_train = np.column_stack([oof_mlp_log, lgb_model.predict(X_train), xgb_model.predict(X_train)])
stack_test  = np.column_stack([test_preds_mlp_log, lgb_test_preds, xgb_test_preds])

meta_model = Ridge(alpha=1.0,random_state=RANDOM_SEED)
meta_model.fit(stack_train, y_train)
final_test_preds_log = meta_model.predict(stack_test)
final_test_preds = np.expm1(final_test_preds_log)

# ----------------------------
# 9) Save submission
# ----------------------------
submission = pd.DataFrame({"sample_id": sample_ids_test, "price": final_test_preds})
submission.to_csv("test_out.csv", index=False)
print("Saved predictions -> test_out.csv")

# Final OOF SMAPE
oof_smape_final = safe_smape(np.expm1(y_train), np.expm1(meta_model.predict(stack_train)))
print("Final OOF SMAPE:", oof_smape_final)
