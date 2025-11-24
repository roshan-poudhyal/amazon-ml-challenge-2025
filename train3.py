#!/usr/bin/env python3
"""
train_full_optuna.py
Optimized multimodal stacking pipeline with:
 - MLP (PyTorch) on MPS/CPU
 - LightGBM/XGBoost tuned with Optuna
 - Feature interactions
 - Stacking meta-learner (Ridge)
 - Final predictions saved to test_final.csv
"""

import os, gc, warnings, joblib
warnings.filterwarnings("ignore")
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
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
import optuna

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
MLP_EPOCHS = 25
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
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom = np.where(denom < eps, eps, denom)
    return 100.0 * np.mean(np.abs(y_pred - y_true) / denom)

def free_mem():
    gc.collect()
    if DEVICE=="cuda":
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

# Numeric features + interactions
numeric_cols = ['value_norm','ipq','content_len','word_count']
for c in numeric_cols:
    if c not in df.columns: df[c] = 0.0

def safe_col(name):
    return df[name].fillna(0.0).values.reshape(-1,1) if name in df.columns else np.zeros((len(df),1))

value_norm = safe_col('value_norm')
ipq = safe_col('ipq')
content_len = safe_col('content_len')
word_count = safe_col('word_count')
interaction_feats = np.hstack([
    value_norm*ipq,
    content_len/ (word_count+1e-6),
    value_norm/content_len
]).astype(np.float32)

extra_feats = np.hstack([value_norm, ipq, content_len, word_count, interaction_feats]).astype(np.float32)

# ----------------------------
# 2) Normalize embeddings + PCA
# ----------------------------
def row_norm(a):
    return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)

text_emb_full = row_norm(text_emb_full)
img_emb_full = row_norm(img_emb_full)

pca_text = PCA(n_components=min(PCA_TEXT_DIM,text_emb_full.shape[1]), random_state=RANDOM_SEED)
pca_img  = PCA(n_components=min(PCA_IMG_DIM,img_emb_full.shape[1]), random_state=RANDOM_SEED)
pca_text.fit(text_emb_full[train_mask.values])
pca_img.fit(img_emb_full[train_mask.values])
text_pca = pca_text.transform(text_emb_full)
img_pca  = pca_img.transform(img_emb_full)

joblib.dump(pca_text, "pca_text.pkl")
joblib.dump(pca_img, "pca_img.pkl")

X_all = np.hstack([text_pca, img_pca, extra_feats]).astype(np.float32)
X_train = X_all[train_mask.values]
X_test  = X_all[test_mask.values]
y_train = np.log1p(y_raw)

# Tree scaling
imp = SimpleImputer(strategy="median")
scaler = StandardScaler()
X_train_tree = scaler.fit_transform(imp.fit_transform(X_train))
X_test_tree  = scaler.transform(imp.transform(X_test))
joblib.dump(imp,"imputer.pkl")
joblib.dump(scaler,"scaler.pkl")

# MLP scaling blocks
sc_t, sc_i, sc_n = StandardScaler(), StandardScaler(), StandardScaler()
text_block = sc_t.fit_transform(text_pca[train_mask.values])
img_block  = sc_i.fit_transform(img_pca[train_mask.values])
num_block  = sc_n.fit_transform(extra_feats[train_mask.values])

text_block_full = np.zeros_like(text_pca,dtype=np.float32)
img_block_full  = np.zeros_like(img_pca,dtype=np.float32)
num_block_full  = np.zeros_like(extra_feats,dtype=np.float32)
text_block_full[train_mask.values] = text_block
text_block_full[test_mask.values] = sc_t.transform(text_pca[test_mask.values])
img_block_full[train_mask.values] = img_block
img_block_full[test_mask.values] = sc_i.transform(img_pca[test_mask.values])
num_block_full[train_mask.values] = num_block
num_block_full[test_mask.values] = sc_n.transform(extra_feats[test_mask.values])

joblib.dump(sc_t,"scaler_text.pkl")
joblib.dump(sc_i,"scaler_img.pkl")
joblib.dump(sc_n,"scaler_num.pkl")

X_text_all, X_img_all, X_num_all = text_block_full, img_block_full, num_block_full

# ----------------------------
# 3) Fusion MLP
# ----------------------------
class Subnet(nn.Module):
    def __init__(self, in_dim, hidden=256, out_dim=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, out_dim), nn.ReLU()
        )
    def forward(self, x): return self.net(x)

class FusionMLP(nn.Module):
    def __init__(self, text_dim, img_dim, num_dim):
        super().__init__()
        self.text_net = Subnet(text_dim)
        self.img_net  = Subnet(img_dim)
        self.num_net  = Subnet(num_dim)
        fused_dim = 128+128+128
        self.head = nn.Sequential(
            nn.Linear(fused_dim,256), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(256,1)
        )
    def forward(self, t, im, n):
        return self.head(torch.cat([self.text_net(t), self.img_net(im), self.num_net(n)],dim=1)).squeeze(-1)

oof_mlp_log = np.zeros(len(X_train),dtype=np.float32)
test_preds_mlp_log_folds = []

kf = KFold(n_splits=N_FOLDS,shuffle=True,random_state=RANDOM_SEED)
for fold,(train_idx,val_idx) in enumerate(kf.split(X_train)):
    print(f"MLP Fold {fold+1}/{N_FOLDS}")
    
    t_train = torch.from_numpy(X_text_all[train_mask.values][train_idx]).float().to(DEVICE)
    im_train = torch.from_numpy(X_img_all[train_mask.values][train_idx]).float().to(DEVICE)
    n_train = torch.from_numpy(X_num_all[train_mask.values][train_idx]).float().to(DEVICE)
    y_train_fold = torch.from_numpy(y_train[train_idx]).float().to(DEVICE)
    
    t_val = torch.from_numpy(X_text_all[train_mask.values][val_idx]).float().to(DEVICE)
    im_val = torch.from_numpy(X_img_all[train_mask.values][val_idx]).float().to(DEVICE)
    n_val = torch.from_numpy(X_num_all[train_mask.values][val_idx]).float().to(DEVICE)
    y_val_fold = torch.from_numpy(y_train[val_idx]).float().to(DEVICE)
    
    train_loader = DataLoader(TensorDataset(t_train,im_train,n_train,y_train_fold),
                              batch_size=BATCH_SIZE,shuffle=True)
    val_loader   = DataLoader(TensorDataset(t_val,im_val,n_val,y_val_fold),
                              batch_size=BATCH_SIZE,shuffle=False)
    
    model = FusionMLP(t_train.shape[1],im_train.shape[1],n_train.shape[1]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.7,patience=2)
    criterion = nn.L1Loss()
    
    best_val_loss=1e9
    best_state=None
    patience=5
    no_improve=0
    
    for epoch in range(MLP_EPOCHS):
        model.train()
        train_loss=0.0
        for xb_t,xb_im,xb_n,yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb_t,xb_im,xb_n), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*xb_t.size(0)
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss_total=0.0
        val_preds=[]
        with torch.no_grad():
            for xb_t,xb_im,xb_n,yb in val_loader:
                preds = model(xb_t,xb_im,xb_n)
                val_preds.append(preds.cpu().numpy())
                val_loss_total += criterion(preds,yb).item()*xb_t.size(0)
        val_loss = val_loss_total/len(val_loader.dataset)
        scheduler.step(val_loss)
        print(f"  Epoch {epoch+1}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
        
        if val_loss<best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            no_improve=0
        else:
            no_improve+=1
            if no_improve>=patience: break
    
    model.load_state_dict(best_state)
    model.eval()
    oof_mlp_log[val_idx] = np.concatenate(val_preds)
    
    with torch.no_grad():
        t_test = torch.from_numpy(X_text_all[test_mask.values]).float().to(DEVICE)
        im_test = torch.from_numpy(X_img_all[test_mask.values]).float().to(DEVICE)
        n_test = torch.from_numpy(X_num_all[test_mask.values]).float().to(DEVICE)
        test_preds_mlp_log_folds.append(model(t_test,im_test,n_test).cpu().numpy())
    free_mem()

test_preds_mlp_log = np.mean(np.vstack(test_preds_mlp_log_folds),axis=0)
oof_mlp_price = np.expm1(oof_mlp_log)
test_mlp_price = np.expm1(test_preds_mlp_log)
print("MLP OOF SMAPE:", safe_smape(y_raw,oof_mlp_price))

# ----------------------------
# 4) Optuna-tuned LGB/XGB
# ----------------------------
def tune_lgb(X, y):
    def objective(trial):
        params = {
            "objective":"regression", "metric":"l1","verbosity":-1,
            "num_leaves": trial.suggest_int("num_leaves",64,512),
            "learning_rate": trial.suggest_loguniform("lr",0.01,0.1),
            "feature_fraction": trial.suggest_uniform("feature_fraction",0.6,1.0),
            "bagging_fraction": trial.suggest_uniform("bagging_fraction",0.6,1.0),
            "bagging_freq": trial.suggest_int("bagging_freq",1,10),
            "min_child_samples": trial.suggest_int("min_child_samples",5,50),
            "seed": RANDOM_SEED
        }
        kf = KFold(n_splits=3,shuffle=True,random_state=RANDOM_SEED)
        smape_list=[]
        for tr_idx,val_idx in kf.split(X):
            dtrain = lgb.Dataset(X[tr_idx],label=y[tr_idx])
            dval = lgb.Dataset(X[val_idx],label=y[val_idx])
            booster = lgb.train(params,dtrain,valid_sets=[dval],verbose_eval=False,num_boost_round=500,early_stopping_rounds=50)
            pred = booster.predict(X[val_idx])
            smape_list.append(safe_smape(np.expm1(y[val_idx]),np.expm1(pred)))
        return np.mean(smape_list)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective,n_trials=25)
    print("Best LGB params:",study.best_params)
    return study.best_params

def tune_xgb(X,y):
    def objective(trial):
        params = {
            "objective":"reg:squarederror",
            "eval_metric":"mae",
            "learning_rate":trial.suggest_loguniform("lr",0.01,0.1),
            "max_depth":trial.suggest_int("max_depth",4,12),
            "subsample":trial.suggest_uniform("subsample",0.6,1.0),
            "colsample_bytree":trial.suggest_uniform("colsample_bytree",0.6,1.0),
            "min_child_weight":trial.suggest_int("min_child_weight",1,10),
            "seed":RANDOM_SEED
        }
        kf = KFold(n_splits=3,shuffle=True,random_state=RANDOM_SEED)
        smape_list=[]
        for tr_idx,val_idx in kf.split(X):
            dtrain = xgb.DMatrix(X[tr_idx],label=y[tr_idx])
            dval = xgb.DMatrix(X[val_idx],label=y[val_idx])
            booster = xgb.train(params,dtrain,num_boost_round=500,evals=[(dval,"val")],early_stopping_rounds=50,verbose_eval=False)
            pred = booster.predict(dval)
            smape_list.append(safe_smape(np.expm1(y[val_idx]),np.expm1(pred)))
        return np.mean(smape_list)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective,n_trials=25)
    print("Best XGB params:",study.best_params)
    return study.best_params

lgb_params_best = tune_lgb(X_train_tree,y_train)
xgb_params_best = tune_xgb(X_train_tree,y_train)

# ----------------------------
# 5) Train final LGB/XGB
# ----------------------------
kf = KFold(n_splits=N_FOLDS,shuffle=True,random_state=RANDOM_SEED)
oof_lgb_log = np.zeros(len(X_train),dtype=np.float32)
oof_xgb_log = np.zeros(len(X_train),dtype=np.float32)
test_preds_lgb,test_preds_xgb = [],[]

for fold,(train_idx,val_idx) in enumerate(kf.split(X_train_tree)):
    print(f"Tree Fold {fold+1}/{N_FOLDS}")
    X_tr,X_val = X_train_tree[train_idx],X_train_tree[val_idx]
    y_tr,y_val = y_train[train_idx],y_train[val_idx]
    
    # LGB
    dtrain = lgb.Dataset(X_tr,label=y_tr)
    dval   = lgb.Dataset(X_val,label=y_val)
    booster_lgb = lgb.train(lgb_params_best,dtrain,num_boost_round=2000,valid_sets=[dval],
                             verbose_eval=500,early_stopping_rounds=100)
    oof_lgb_log[val_idx] = booster_lgb.predict(X_val,num_iteration=booster_lgb.best_iteration)
    test_preds_lgb.append(booster_lgb.predict(X_test_tree,num_iteration=booster_lgb.best_iteration))
    
    # XGB
    dtrain_x = xgb.DMatrix(X_tr,label=y_tr)
    dval_x   = xgb.DMatrix(X_val,label=y_val)
    booster_x = xgb.train({**xgb_params_best,"objective":"reg:squarederror"},
                          dtrain_x,num_boost_round=2000,evals=[(dval_x,"val")],
                          early_stopping_rounds=100,verbose_eval=500)
    oof_xgb_log[val_idx] = booster_x.predict(dval_x,iteration_range=(0,booster_x.best_iteration))
    test_preds_xgb.append(booster_x.predict(xgb.DMatrix(X_test_tree),iteration_range=(0,booster_x.best_iteration)))

oof_lgb_price = np.expm1(oof_lgb_log)
oof_xgb_price = np.expm1(oof_xgb_log)
test_lgb_price = np.expm1(np.mean(np.vstack(test_preds_lgb),axis=0))
test_xgb_price = np.expm1(np.mean(np.vstack(test_preds_xgb),axis=0))
print("LightGBM OOF SMAPE:", safe_smape(y_raw,oof_lgb_price))
print("XGBoost OOF SMAPE:", safe_smape(y_raw,oof_xgb_price))

# ----------------------------
# 6) Stacking Meta-Learner
# ----------------------------
stack_X = np.vstack([oof_mlp_price,oof_lgb_price,oof_xgb_price]).T
stack_test = np.vstack([test_mlp_price,test_lgb_price,test_xgb_price]).T
meta = Ridge(alpha=1.0)
meta.fit(stack_X,y_raw)
final_preds = meta.predict(stack_test)
np.save("stacked_test_preds.npy",final_preds)

# Save to CSV
pd.DataFrame({"sample_id":sample_ids_test,"price":final_preds}).to_csv("test_final3.csv",index=False)
print("Saved final predictions to test_final.csv")
