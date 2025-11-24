#!/usr/bin/env python3
"""
train_full_fixed_callbacks.py
Optimized multimodal stacking pipeline:
 - Fusion MLP (PyTorch; MPS/CUDA if available)
 - LightGBM + XGBoost (CPU)
 - Stacking meta-learner (Ridge)
"""

import os, gc, warnings, joblib
warnings.filterwarnings("ignore")
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

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
BATCH_SIZE = 256
MLP_EPOCHS = 12
MLP_LR = 1e-3
PCA_TEXT_DIM = 128
PCA_IMG_DIM  = 128
TARGET_COL = "price"

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
print("Using device:", DEVICE)

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
print("Loading data & embeddings...")
df = pd.read_parquet(DATA_PARQUET)
text_emb_full = np.load(TEXT_EMB_FILE).astype(np.float32)
img_emb_full = np.load(IMG_EMB_FILE).astype(np.float32)

train_mask = df['is_test'] == 0
test_mask  = df['is_test'] == 1
sample_ids_test = df.loc[test_mask, 'sample_id'].values

y_raw = df.loc[train_mask, TARGET_COL].values.astype(np.float32)

numeric_cols = ['value_norm','ipq','content_len','word_count']
for c in numeric_cols:
    if c not in df.columns:
        df[c] = 0.0

def safe_col(name, default=0.0):
    return df[name].fillna(default).values.reshape(-1,1) if name in df.columns else np.full((len(df),1), default)

content_len = safe_col('content_len')
word_count = safe_col('word_count')
value_norm = safe_col('value_norm')
ipq = safe_col('ipq')

# ----------------------------
# 2) Normalize embeddings + PCA
# ----------------------------
def row_norm(a):
    n = np.linalg.norm(a, axis=1, keepdims=True) + 1e-8
    return a / n

text_emb_full = row_norm(text_emb_full)
img_emb_full = row_norm(img_emb_full)

pca_text = PCA(n_components=min(PCA_TEXT_DIM, text_emb_full.shape[1]), random_state=RANDOM_SEED)
pca_img  = PCA(n_components=min(PCA_IMG_DIM, img_emb_full.shape[1]), random_state=RANDOM_SEED)
pca_text.fit(text_emb_full[train_mask.values])
pca_img.fit(img_emb_full[train_mask.values])
text_pca = pca_text.transform(text_emb_full)
img_pca  = pca_img.transform(img_emb_full)
joblib.dump(pca_text, "pca_text.pkl")
joblib.dump(pca_img, "pca_img.pkl")

extra_feats = np.hstack([
    ipq.astype(np.float32),
    value_norm.astype(np.float32),
    content_len.astype(np.float32),
    word_count.astype(np.float32),
    (value_norm / (ipq + 1e-6)).astype(np.float32),
    (content_len / (word_count + 1e-6)).astype(np.float32)
]).astype(np.float32)

X_all = np.hstack([text_pca, img_pca, extra_feats]).astype(np.float32)
X_train = X_all[train_mask.values]
X_test  = X_all[test_mask.values]
y_train = np.log1p(y_raw)

# Tree scaling
imp = SimpleImputer(strategy="median")
scaler = StandardScaler()
X_train_tree = scaler.fit_transform(imp.fit_transform(X_train))
X_test_tree  = scaler.transform(imp.transform(X_test))
joblib.dump(imp, "imputer.pkl")
joblib.dump(scaler, "scaler.pkl")

# MLP scaling
sc_t, sc_i, sc_n = StandardScaler(), StandardScaler(), StandardScaler()
text_block = sc_t.fit_transform(text_pca[train_mask.values])
img_block  = sc_i.fit_transform(img_pca[train_mask.values])
num_block  = sc_n.fit_transform(extra_feats[train_mask.values])

text_block_full = np.zeros_like(text_pca, dtype=np.float32)
img_block_full  = np.zeros_like(img_pca, dtype=np.float32)
num_block_full  = np.zeros_like(extra_feats, dtype=np.float32)
text_block_full[train_mask.values] = text_block
text_block_full[test_mask.values] = sc_t.transform(text_pca[test_mask.values])
img_block_full[train_mask.values] = img_block
img_block_full[test_mask.values] = sc_i.transform(img_pca[test_mask.values])
num_block_full[train_mask.values] = num_block
num_block_full[test_mask.values] = sc_n.transform(extra_feats[test_mask.values])

joblib.dump(sc_t, "scaler_text.pkl")
joblib.dump(sc_i, "scaler_img.pkl")
joblib.dump(sc_n, "scaler_num.pkl")

X_text_all = text_block_full
X_img_all = img_block_full
X_num_all = num_block_full

# ----------------------------
# 3) Fusion MLP
# ----------------------------
class Subnet(nn.Module):
    def __init__(self, in_dim, hidden=128, out_dim=64, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
            nn.ReLU()
        )
    def forward(self, x):
        return self.net(x)

class FusionMLP(nn.Module):
    def __init__(self, text_dim, img_dim, num_dim,
                 t_out=64, i_out=64, n_out=32, hidden=128):
        super().__init__()
        self.text_net = Subnet(text_dim, hidden, t_out, 0.15)
        self.img_net  = Subnet(img_dim, hidden, i_out, 0.15)
        self.num_net  = Subnet(num_dim, max(64, hidden//2), n_out, 0.1)
        fused_dim = t_out + i_out + n_out
        self.head = nn.Sequential(
            nn.Linear(fused_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, 1)
        )
    def forward(self, t, im, n):
        return self.head(torch.cat([self.text_net(t), self.img_net(im), self.num_net(n)], dim=1)).squeeze(-1)

oof_mlp_log = np.zeros(len(X_train), dtype=np.float32)
test_preds_mlp_log_folds = []
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    print(f"MLP Fold {fold+1}/{N_FOLDS}")
    
    t_train = torch.from_numpy(X_text_all[train_mask.values][train_idx]).float().to(DEVICE)
    im_train = torch.from_numpy(X_img_all[train_mask.values][train_idx]).float().to(DEVICE)
    n_train = torch.from_numpy(X_num_all[train_mask.values][train_idx]).float().to(DEVICE)
    y_train_fold = torch.from_numpy(y_train[train_idx]).float().to(DEVICE)
    
    t_val = torch.from_numpy(X_text_all[train_mask.values][val_idx]).float().to(DEVICE)
    im_val = torch.from_numpy(X_img_all[train_mask.values][val_idx]).float().to(DEVICE)
    n_val = torch.from_numpy(X_num_all[train_mask.values][val_idx]).float().to(DEVICE)
    y_val_fold = torch.from_numpy(y_train[val_idx]).float().to(DEVICE)
    
    train_ds = TensorDataset(t_train, im_train, n_train, y_train_fold)
    val_ds = TensorDataset(t_val, im_val, n_val, y_val_fold)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    model = FusionMLP(t_train.shape[1], im_train.shape[1], n_train.shape[1]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=MLP_LR, weight_decay=1e-5)
    criterion = nn.L1Loss()
    
    best_val_loss = 1e9
    best_state = None
    no_improve = 0
    patience = 3
    
    for epoch in range(MLP_EPOCHS):
        model.train()
        train_loss = 0.0
        count = 0
        for xb_t, xb_im, xb_n, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb_t, xb_im, xb_n)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb_t.size(0)
            count += xb_t.size(0)
        train_loss /= max(1, count)
        
        model.eval()
        val_loss_total = 0.0
        val_preds_fold = []
        with torch.no_grad():
            for xb_t, xb_im, xb_n, yb in val_loader:
                preds = model(xb_t, xb_im, xb_n)
                val_preds_fold.append(preds.cpu().numpy())
                val_loss_total += criterion(preds, yb).item() * xb_t.size(0)
        val_loss = val_loss_total / max(1, len(val_idx))
        print(f"  Epoch {epoch+1} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print("  Early stopping MLP")
                break
    
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    
    oof_mlp_log[val_idx] = np.concatenate(val_preds_fold)
    
    with torch.no_grad():
        t_test = torch.from_numpy(X_text_all[test_mask.values]).float().to(DEVICE)
        im_test = torch.from_numpy(X_img_all[test_mask.values]).float().to(DEVICE)
        n_test = torch.from_numpy(X_num_all[test_mask.values]).float().to(DEVICE)
        test_preds_mlp_log_folds.append(model(t_test, im_test, n_test).cpu().numpy())
    
    free_mem()

test_preds_mlp_log = np.mean(np.vstack(test_preds_mlp_log_folds), axis=0)
oof_mlp_price = np.expm1(oof_mlp_log)
test_mlp_price = np.expm1(test_preds_mlp_log)
print("MLP OOF SMAPE:", safe_smape(y_raw, oof_mlp_price))

# ----------------------------
# 4) LightGBM + XGBoost
# ----------------------------
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
oof_lgb_log = np.zeros(len(X_train), dtype=np.float32)
oof_xgb_log = np.zeros(len(X_train), dtype=np.float32)
test_preds_lgb, test_preds_xgb = [], []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_tree)):
    print(f"Tree Fold {fold+1}/{N_FOLDS}")
    X_tr, X_val = X_train_tree[train_idx], X_train_tree[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    # LightGBM with callbacks
    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval = lgb.Dataset(X_val, label=y_val)
    lgb_params = {
        "objective": "regression", "metric": "l1", "verbosity": -1,
        "learning_rate": 0.03, "num_leaves":128, "feature_fraction":0.9,
        "bagging_fraction":0.8, "bagging_freq":2, "seed": RANDOM_SEED
    }
    booster_lgb = lgb.train(
        lgb_params, dtrain, num_boost_round=2000, valid_sets=[dval],
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=500)]
    )
    oof_lgb_log[val_idx] = booster_lgb.predict(X_val, num_iteration=booster_lgb.best_iteration)
    test_preds_lgb.append(booster_lgb.predict(X_test_tree, num_iteration=booster_lgb.best_iteration))
    
    # XGBoost
    dtrain_x = xgb.DMatrix(X_tr, label=y_tr)
    dval_x = xgb.DMatrix(X_val, label=y_val)
    booster_x = xgb.train({
        "objective":"reg:squarederror","eval_metric":"mae","learning_rate":0.03,"max_depth":8,"seed":RANDOM_SEED
    }, dtrain_x, num_boost_round=2000, evals=[(dval_x,"val")], early_stopping_rounds=50, verbose_eval=500)
    oof_xgb_log[val_idx] = booster_x.predict(dval_x, iteration_range=(0, booster_x.best_iteration))
    test_preds_xgb.append(booster_x.predict(xgb.DMatrix(X_test_tree), iteration_range=(0, booster_x.best_iteration)))

oof_lgb_price = np.expm1(oof_lgb_log)
oof_xgb_price = np.expm1(oof_xgb_log)
test_lgb_price = np.expm1(np.mean(np.vstack(test_preds_lgb), axis=0))
test_xgb_price = np.expm1(np.mean(np.vstack(test_preds_xgb), axis=0))
print("LightGBM OOF SMAPE:", safe_smape(y_raw, oof_lgb_price))
print("XGBoost OOF SMAPE:", safe_smape(y_raw, oof_xgb_price))

# ----------------------------
# 5) Stacking Meta-Learner
# ----------------------------
stack_X = np.vstack([oof_mlp_price, oof_lgb_price, oof_xgb_price]).T
stack_test = np.vstack([test_mlp_price, test_lgb_price, test_xgb_price]).T
meta = Ridge(alpha=1.0)
meta.fit(stack_X, y_raw)
final_preds = meta.predict(stack_test)
print("Stacked predictions ready.")

np.save("stacked_test_preds.npy", final_preds)
joblib.dump(meta, "meta_model_ridge.pkl")

# ----------------------------
# 6) Save final predictions
# ----------------------------
submission = pd.DataFrame({
    "sample_id": sample_ids_test,
    "predicted_price": final_preds
})
submission.to_csv("test_final.csv", index=False)
print("Saved final predictions to test_final.csv")
