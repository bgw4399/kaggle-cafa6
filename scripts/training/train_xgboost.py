import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve
import json
import joblib

# Config
FEATURE_FILE = "train_stacking_features.tsv"
MODEL_FILE = "xgboost_model.json"

def train_model():
    print("?? Training XGBoost Stacking Model...")
    
    # 1. Load Data
    print(f"   ?뱿 Loading Features from {FEATURE_FILE}...")
    try:
        df = pd.read_csv(FEATURE_FILE, sep="\t")
    except FileNotFoundError:
        print(f"??Error: {FEATURE_FILE} not found. Run generate_stacking_features.py first.")
        return
        
    print(f"   ?뱤 Data Shape: {df.shape}")
    print("   Features:", df.columns.tolist())
    
    # Features & Target
    # We use: pident, bitscore, log_evalue, term_freq
    features = ["pident", "bitscore", "log_evalue", "term_freq"]
    target = "label"
    
    X = df[features]
    y = df[target]
    
    # 2. Split (Train/Val)
    # Use Stratified Split
    print("   ?귨툘 Splitting Train/Val (80/20)...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 3. Train XGBoost
    print("   ?쭬 Training XGBoost...")
    # Calculate scale_pos_weight for imbalance
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    print(f"   ?뽳툘 Scale Pos Weight: {scale_pos_weight:.2f}")
    
    model = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        scale_pos_weight=scale_pos_weight,
        tree_method='hist', # Fast training
        early_stopping_rounds=50,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100
    )
    
    # 4. Evaluate
    print("   ?뵇 Evaluating...")
    y_pred_prob = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred_prob)
    
    # Find best F1 Threshold
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_pred_prob)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    
    print(f"   狩?Validation AUC: {auc:.4f}")
    print(f"   狩?Best F1: {best_f1:.4f} at Threshold: {best_thresh:.4f}")
    
    # 5. Save Model
    print(f"   ?뮶 Saving Model to {MODEL_FILE}...")
    model.save_model(MODEL_FILE)
    
    # Save Metadata (Threshold)
    metadata = {
        "best_threshold": float(best_thresh),
        "best_f1": float(best_f1),
        "auc": float(auc)
    }
    with open("./data/metadata/model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
        
    print("??Training Complete.")

if __name__ == "__main__":
    train_model()

