#!/usr/bin/env python3
"""
train.py
Multimodal stacking pipeline (Fusion MLP + LightGBM/XGBoost + meta-learner)
with Optuna hyperparameter tuning. Compatible with macOS M1/M2/M3 (MPS).
"""

import os, gc, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import argparse

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA, TruncatedSVD

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import lightgbm as lgb
import xgboost as xgb
import optuna
from sklearn.feature_extraction.text import TfidfVectorizer

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
N_PCA_TEXT = 128
N_PCA_IMG  = 128
OPTUNA_TRIALS = 20
META_MODEL = "ridge"
TARGET_COL = "price"

# Optional CLI overrides for faster runs
parser = argparse.ArgumentParser()
parser.add_argument("--trials", type=int, default=OPTUNA_TRIALS)
parser.add_argument("--mlp_epochs", type=int, default=MLP_EPOCHS)
parser.add_argument("--pca_text", type=int, default=N_PCA_TEXT)
parser.add_argument("--pca_img", type=int, default=N_PCA_IMG)
args, _ = parser.parse_known_args()
OPTUNA_TRIALS = args.trials
MLP_EPOCHS = args.mlp_epochs
N_PCA_TEXT = args.pca_text
N_PCA_IMG = args.pca_img

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
y_raw = df.loc[train_mask, 'price'].values.astype(np.float32)

# numeric features
numeric_cols = ['value_norm','ipq','content_len','word_count']
for c in numeric_cols:
    if c not in df.columns:
        df[c] = 0.0
X_num_all = df[numeric_cols].fillna(0.0).values.astype(np.float32)

# ----------------------------
# 2) Normalize embeddings + PCA
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

# engineered numeric features
ipq = df['ipq'].fillna(0).values.reshape(-1,1).astype(np.float32)
valnorm = df['value_norm'].fillna(0).values.reshape(-1,1).astype(np.float32)
content_len = df['content_len'].fillna(0).values.reshape(-1,1).astype(np.float32)
word_count = df['word_count'].fillna(0).values.reshape(-1,1).astype(np.float32)
ratio_val_per_item = (valnorm/(ipq+1e-6)).astype(np.float32)
ratio_len_per_word = (content_len/(word_count+1e-6)).astype(np.float32)
log_ipq = np.log1p(ipq)
log_valnorm = np.log1p(valnorm)
log_content_len = np.log1p(content_len)
log_word_count = np.log1p(word_count)
log_ratio_val_per_item = np.log1p(ratio_val_per_item)

# Target encoding (OOF) for brand and unit_norm to reduce leakage
brands = df['brand'].fillna('unknown').astype(str).values
units  = df['unit_norm'].fillna('none').astype(str).values

brand_te = np.zeros(len(df), dtype=np.float32)
unit_te  = np.zeros(len(df), dtype=np.float32)

brand_counts = pd.Series(brands[train_mask.values]).value_counts()
unit_counts  = pd.Series(units[train_mask.values]).value_counts()
brand_count_all = pd.Series(brands).map(brand_counts).fillna(0).to_numpy(dtype=np.float32).reshape(-1,1)
unit_count_all  = pd.Series(units).map(unit_counts).fillna(0).to_numpy(dtype=np.float32).reshape(-1,1)

kf_te = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
y_log_all_train = np.log1p(y_raw)
global_mean = float(np.mean(y_log_all_train))
for tr_idx, val_idx in kf_te.split(np.where(train_mask.values)[0]):
    tr_rows = np.where(train_mask.values)[0][tr_idx]
    val_rows = np.where(train_mask.values)[0][val_idx]
    # Brand TE
    br_mean = pd.Series(y_log_all_train[tr_idx], index=pd.Series(brands[tr_rows]).values).groupby(level=0).mean()
    brand_te[val_rows] = pd.Series(brands[val_rows]).map(br_mean).fillna(global_mean).to_numpy(dtype=np.float32)
    # Unit TE
    un_mean = pd.Series(y_log_all_train[tr_idx], index=pd.Series(units[tr_rows]).values).groupby(level=0).mean()
    unit_te[val_rows] = pd.Series(units[val_rows]).map(un_mean).fillna(global_mean).to_numpy(dtype=np.float32)
# For test rows, use full-train means
br_full = pd.Series(y_log_all_train, index=pd.Series(brands[train_mask.values]).values).groupby(level=0).mean()
un_full = pd.Series(y_log_all_train, index=pd.Series(units[train_mask.values]).values).groupby(level=0).mean()
brand_te[test_mask.values] = pd.Series(brands[test_mask.values]).map(br_full).fillna(global_mean).to_numpy(dtype=np.float32)
unit_te[test_mask.values]  = pd.Series(units[test_mask.values]).map(un_full).fillna(global_mean).to_numpy(dtype=np.float32)

brand_te = brand_te.reshape(-1,1)
unit_te  = unit_te.reshape(-1,1)

extra_feats = np.concatenate([
    ipq, valnorm, content_len, word_count,
    ratio_val_per_item, ratio_len_per_word,
    log_ipq, log_valnorm, log_content_len, log_word_count, log_ratio_val_per_item,
    brand_count_all, unit_count_all, brand_te, unit_te
], axis=1).astype(np.float32)

# scale MLP inputs per block using train split statistics
scaler_text = StandardScaler()
scaler_img  = StandardScaler()
scaler_num  = StandardScaler()
text_pca_scaled = scaler_text.fit_transform(text_pca[train_mask.values])
text_pca_scaled_full = np.zeros_like(text_pca, dtype=np.float32)
text_pca_scaled_full[train_mask.values] = text_pca_scaled
text_pca_scaled_full[test_mask.values]  = scaler_text.transform(text_pca[test_mask.values])

img_pca_scaled = scaler_img.fit_transform(img_pca[train_mask.values])
img_pca_scaled_full = np.zeros_like(img_pca, dtype=np.float32)
img_pca_scaled_full[train_mask.values] = img_pca_scaled
img_pca_scaled_full[test_mask.values]  = scaler_img.transform(img_pca[test_mask.values])

extra_feats_scaled = scaler_num.fit_transform(extra_feats[train_mask.values])
extra_feats_scaled_full = np.zeros_like(extra_feats, dtype=np.float32)
extra_feats_scaled_full[train_mask.values] = extra_feats_scaled
extra_feats_scaled_full[test_mask.values]  = scaler_num.transform(extra_feats[test_mask.values])

joblib.dump(scaler_text, "scaler_text.pkl")
joblib.dump(scaler_img,  "scaler_img.pkl")
joblib.dump(scaler_num,  "scaler_num.pkl")

# combine all base dense features
X_all_dense = np.hstack([text_pca,img_pca,extra_feats])

# Add TF-IDF SVD features (trees only) for stronger text signal
print("Fitting TF-IDF -> SVD features for trees...")
text_series = df['content_clean'].fillna('').astype(str)
vectorizer = TfidfVectorizer(min_df=5, max_features=50000, ngram_range=(1,2))
vectorizer.fit(text_series[train_mask.values])
tfidf_all = vectorizer.transform(text_series)
svd = TruncatedSVD(n_components=128, random_state=RANDOM_SEED)
svd.fit(tfidf_all[train_mask.values])
tfidf_svd_all = svd.transform(tfidf_all).astype(np.float32)
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
joblib.dump(svd, "tfidf_svd.pkl")

X_all = np.hstack([X_all_dense, tfidf_svd_all])
print("Combined feature shape:", X_all.shape)

# split train/test (for tree models)
X_train = X_all[train_mask.values]
X_test  = X_all[test_mask.values]
y_train_raw = y_raw
y_train = np.log1p(y_train_raw)

# impute + scale
imp = SimpleImputer(strategy="median")
scaler = StandardScaler()
X_train = scaler.fit_transform(imp.fit_transform(X_train))
X_test  = scaler.transform(imp.transform(X_test))
joblib.dump(imp,"imputer.pkl")
joblib.dump(scaler,"scaler.pkl")

# ----------------------------
# 3) Fusion MLP
# ----------------------------
print("Training Fusion MLP OOF predictions...")

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
        fused_dim = (hidden//2) + (hidden//2) + (hidden//4)  # FIXED
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

t_dim = text_pca.shape[1]
i_dim = img_pca.shape[1]
n_dim = extra_feats.shape[1]

# Use scaled blocks for MLP stability
X_text_all = text_pca_scaled_full.astype(np.float32)
X_img_all  = img_pca_scaled_full.astype(np.float32)
X_num_all  = extra_feats_scaled_full.astype(np.float32)

oof_mlp_log = np.zeros(len(X_train),dtype=np.float32)
test_preds_mlp_log_folds = []
mlp_fold_smape = []

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

for fold,(tr_idx,val_idx) in enumerate(kf.split(X_train)):
    print(f"MLP Fold {fold+1}/{N_FOLDS}")
    
    t_tr = torch.from_numpy(X_text_all[tr_idx]).float()
    im_tr = torch.from_numpy(X_img_all[tr_idx]).float()
    n_tr = torch.from_numpy(X_num_all[tr_idx]).float()
    y_tr = torch.from_numpy(y_train[tr_idx]).float()

    t_val = torch.from_numpy(X_text_all[val_idx]).float()
    im_val = torch.from_numpy(X_img_all[val_idx]).float()
    n_val = torch.from_numpy(X_num_all[val_idx]).float()
    y_val = torch.from_numpy(y_train[val_idx]).float()

    train_ds = TensorDataset(t_tr,im_tr,n_tr,y_tr)
    val_ds   = TensorDataset(t_val,im_val,n_val,y_val)
    train_loader = DataLoader(train_ds,batch_size=BATCH_SIZE,shuffle=True,num_workers=0)
    val_loader   = DataLoader(val_ds,batch_size=BATCH_SIZE,shuffle=False,num_workers=0)

    model = FusionMLP(t_dim,i_dim,n_dim,hidden=128).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(),lr=MLP_LR,weight_decay=1e-5)
    criterion = nn.L1Loss()

    best_val_loss = 1e9
    best_state = None
    patience, no_improve = 3, 0
    for ep in range(MLP_EPOCHS):
        model.train()
        running,count = 0.0,0
        for xb_t,xb_im,xb_n,yb in train_loader:
            xb_t,xb_im,xb_n,yb = xb_t.to(DEVICE),xb_im.to(DEVICE),xb_n.to(DEVICE),yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb_t,xb_im,xb_n), yb)
            loss.backward()
            optimizer.step()
            running += loss.item()*xb_t.size(0)
            count += xb_t.size(0)
        avg_train = running/max(1,count)

        # validate
        model.eval()
        v_running,v_count=0.0,0
        with torch.no_grad():
            for xb_t,xb_im,xb_n,yb in val_loader:
                xb_t,xb_im,xb_n,yb = xb_t.to(DEVICE),xb_im.to(DEVICE),xb_n.to(DEVICE),yb.to(DEVICE)
                v_running += criterion(model(xb_t,xb_im,xb_n), yb).item()*xb_t.size(0)
                v_count += xb_t.size(0)
        avg_val = v_running/max(1,v_count)
        print(f" Epoch {ep+1}/{MLP_EPOCHS} train_loss={avg_train:.4f} val_loss={avg_val:.4f}")
        if avg_val<best_val_loss:
            best_val_loss=avg_val
            best_state=model.state_dict()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve>=patience:
                print(" Early stopping triggered for this fold")
                break

    model.load_state_dict(best_state)
    model.eval()

    # store OOF
    with torch.no_grad():
        val_preds_log = model(t_val.to(DEVICE),im_val.to(DEVICE),n_val.to(DEVICE)).cpu().numpy()
        oof_mlp_log[val_idx] = val_preds_log
    # fold SMAPE in original space
    fold_smape = safe_smape(np.expm1(y_val.cpu().numpy()), np.expm1(val_preds_log))
    mlp_fold_smape.append(fold_smape)
    print(f" MLP Fold {fold+1} SMAPE: {fold_smape:.4f}")

    # test predictions fold
    with torch.no_grad():
        t_test = torch.from_numpy(X_text_all[test_mask.values]).float().to(DEVICE)
        im_test = torch.from_numpy(X_img_all[test_mask.values]).float().to(DEVICE)
        n_test = torch.from_numpy(X_num_all[test_mask.values]).float().to(DEVICE)
        test_preds_mlp_log_folds.append(model(t_test,im_test,n_test).cpu().numpy())
    free_mem()

test_preds_mlp_log = np.mean(test_preds_mlp_log_folds,axis=0)
oof_smape = safe_smape(np.expm1(y_train), np.expm1(oof_mlp_log))
print(f"OOF MLP SMAPE: {oof_smape:.4f}")
print("MLP per-fold SMAPE:", [f"{s:.4f}" for s in mlp_fold_smape])

joblib.dump(oof_mlp_log,"oof_mlp_log.pkl")
joblib.dump(test_preds_mlp_log,"test_preds_mlp_log.pkl")

# ----------------------------
# LIGHTGBM / XGBOOST / STACKING
# ----------------------------
# LightGBM
def objective_lgb(trial):
    param = {
        "objective": "regression",
        "metric": "l1",
        "verbosity": -1,
        "learning_rate": trial.suggest_float("lr",1e-3,0.1,log=True),
        "num_leaves": trial.suggest_int("num_leaves",31,512),
        "feature_fraction": trial.suggest_float("feature_fraction",0.5,1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction",0.5,1.0),
        "bagging_freq": trial.suggest_int("bagging_freq",1,10),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 200),
        "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 1.0),
        "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 2.0),
        "seed": RANDOM_SEED
    }
    val_scores=[]
    for tr_idx,val_idx in kf.split(X_train):
        X_tr,X_val = X_train[tr_idx],X_train[val_idx]
        y_tr,y_val = y_train[tr_idx],y_train[val_idx]
        dtrain = lgb.Dataset(X_tr,label=y_tr)
        dval = lgb.Dataset(X_val,label=y_val)
        booster = lgb.train(param,dtrain,num_boost_round=600,
                            valid_sets=[dval],
                            callbacks=[lgb.early_stopping(50)])
        preds = booster.predict(X_val)
        val_scores.append(safe_smape(np.expm1(y_val), np.expm1(preds)))
    return np.mean(val_scores)

print("Training LightGBM with Optuna...")
study_lgb = optuna.create_study(direction="minimize",
                                  sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
                                  pruner=optuna.pruners.MedianPruner(n_warmup_steps=2))
study_lgb.optimize(objective_lgb,n_trials=OPTUNA_TRIALS, show_progress_bar=False)
best_lgb_params = study_lgb.best_trial.params
print("Best LightGBM params:",best_lgb_params)

# OOF LightGBM to avoid leakage in meta-learner
oof_lgb_log = np.zeros(len(X_train), dtype=np.float32)
lgb_test_preds_log_folds = []
lgb_fold_smape = []
for fold,(tr_idx,val_idx) in enumerate(kf.split(X_train)):
    X_tr,X_val = X_train[tr_idx],X_train[val_idx]
    y_tr,y_val = y_train[tr_idx],y_train[val_idx]
    dtrain = lgb.Dataset(X_tr,label=y_tr)
    dval = lgb.Dataset(X_val,label=y_val)
    booster = lgb.train({**best_lgb_params,"objective":"regression","metric":"l1","verbosity":-1},
                        dtrain,num_boost_round=2000,valid_sets=[dval],callbacks=[lgb.early_stopping(100)])
    val_preds = booster.predict(X_val)
    oof_lgb_log[val_idx] = val_preds
    lgb_test_preds_log_folds.append(booster.predict(X_test))
    fold_smape = safe_smape(np.expm1(y_val), np.expm1(val_preds))
    lgb_fold_smape.append(fold_smape)
    print(f" LGB Fold {fold+1} SMAPE: {fold_smape:.4f}")

lgb_test_preds_log = np.mean(lgb_test_preds_log_folds, axis=0)
print("OOF LGB SMAPE:", f"{safe_smape(np.expm1(y_train), np.expm1(oof_lgb_log)):.4f}")
print("LGB per-fold SMAPE:", [f"{s:.4f}" for s in lgb_fold_smape])
joblib.dump(oof_lgb_log, "oof_lgb_log.pkl")

# XGBoost
def objective_xgb(trial):
    param = {
        "tree_method":"gpu_hist" if torch.cuda.is_available() else "hist",
        "eval_metric":"mae",
        "learning_rate":trial.suggest_float("lr",1e-3,0.1,log=True),
        "max_depth":trial.suggest_int("max_depth",3,12),
        "subsample":trial.suggest_float("subsample",0.5,1.0),
        "colsample_bytree":trial.suggest_float("colsample_bytree",0.5,1.0),
        "n_estimators":1000,
        "random_state":RANDOM_SEED
    }
    val_scores=[]
    for tr_idx,val_idx in kf.split(X_train):
        X_tr,X_val = X_train[tr_idx],X_train[val_idx]
        y_tr,y_val = y_train[tr_idx],y_train[val_idx]
        model = xgb.XGBRegressor(**param)
        model.fit(X_tr,y_tr,eval_set=[(X_val,y_val)], verbose=False)
        preds = model.predict(X_val)
        val_scores.append(safe_smape(np.expm1(y_val), np.expm1(preds)))
    return np.mean(val_scores)

print("Training XGBoost with Optuna...")
study_xgb = optuna.create_study(direction="minimize",
                                  sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
                                  pruner=optuna.pruners.MedianPruner(n_warmup_steps=2))
study_xgb.optimize(objective_xgb,n_trials=OPTUNA_TRIALS, show_progress_bar=False)
best_xgb_params = study_xgb.best_trial.params
print("Best XGBoost params:",best_xgb_params)

# OOF XGBoost to avoid leakage in meta-learner
oof_xgb_log = np.zeros(len(X_train), dtype=np.float32)
xgb_test_preds_log_folds = []
xgb_fold_smape = []
for fold,(tr_idx,val_idx) in enumerate(kf.split(X_train)):
    X_tr,X_val = X_train[tr_idx],X_train[val_idx]
    y_tr,y_val = y_train[tr_idx],y_train[val_idx]
    model = xgb.XGBRegressor(**best_xgb_params, n_estimators=1200, random_state=RANDOM_SEED)
    model.fit(X_tr,y_tr,eval_set=[(X_val,y_val)], verbose=False)
    val_preds = model.predict(X_val)
    oof_xgb_log[val_idx] = val_preds
    xgb_test_preds_log_folds.append(model.predict(X_test))
    fold_smape = safe_smape(np.expm1(y_val), np.expm1(val_preds))
    xgb_fold_smape.append(fold_smape)
    print(f" XGB Fold {fold+1} SMAPE: {fold_smape:.4f}")

xgb_test_preds_log = np.mean(xgb_test_preds_log_folds, axis=0)
print("OOF XGB SMAPE:", f"{safe_smape(np.expm1(y_train), np.expm1(oof_xgb_log)):.4f}")
print("XGB per-fold SMAPE:", [f"{s:.4f}" for s in xgb_fold_smape])
joblib.dump(oof_xgb_log, "oof_xgb_log.pkl")

# ----------------------------
# Meta-learner
# ----------------------------
# Use consistent log-space features for meta-learner and avoid leakage
stack_train = np.column_stack([oof_mlp_log, oof_lgb_log, oof_xgb_log])
stack_test  = np.column_stack([test_preds_mlp_log, lgb_test_preds_log, xgb_test_preds_log])

if META_MODEL=="ridge":
    meta_model = Ridge(alpha=1.0,random_state=RANDOM_SEED)
else:
    meta_model = lgb.LGBMRegressor(n_estimators=500,random_state=RANDOM_SEED)

meta_model.fit(stack_train,y_train)
final_test_preds_log = meta_model.predict(stack_test)
final_test_preds = np.expm1(final_test_preds_log)
# Ensure positivity as per constraints
final_test_preds = np.clip(final_test_preds, 0.01, None).astype(float)

submission = pd.DataFrame({"sample_id":sample_ids_test,"price":final_test_preds})
submission.to_csv("test_out.csv",index=False)
print("Saved test predictions -> test_out.csv")
