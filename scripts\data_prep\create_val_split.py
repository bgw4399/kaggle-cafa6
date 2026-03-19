import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

TRAIN_TERMS = "./data/raw/train/train_terms.tsv"
OUT_DIR = "./data/raw/train/Split"

def create_split():
    print("?? Creating Train/Val Split for CAFA6...")
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # 1. Load Data
    print("   ?뱿 Loading train_terms.tsv...")
    df = pd.read_csv(TRAIN_TERMS, sep='\t')
    
    # 2. Get Unique Proteins
    unique_pids = df['EntryID'].unique()
    print(f"   ?㎚ Total Proteins: {len(unique_pids):,}")
    
    # 3. Split IDs (80/20)
    train_ids, val_ids = train_test_split(unique_pids, test_size=0.2, random_state=42)
    print(f"   ?귨툘 Split: Train={len(train_ids):,}, Val={len(val_ids):,}")
    
    # 4. Filter DF
    # Use set for speed
    train_id_set = set(train_ids)
    val_id_set = set(val_ids)
    
    df_train = df[df['EntryID'].isin(train_id_set)]
    df_val = df[df['EntryID'].isin(val_id_set)]
    
    # 5. Save Splits
    path_train = os.path.join(OUT_DIR, "train_terms_split.tsv")
    path_val = os.path.join(OUT_DIR, "val_terms_split.tsv") # Use as GT for evaluation
    
    # Save Train (Standard Header)
    df_train.to_csv(path_train, sep='\t', index=False)
    
    # Save Val (For input to model AND for GT)
    df_val.to_csv(path_val, sep='\t', index=False)
    
    # Save ID lists separately for filtering embeddings
    np.save(os.path.join(OUT_DIR, "train_ids.npy"), train_ids)
    np.save(os.path.join(OUT_DIR, "val_ids.npy"), val_ids)
    
    print(f"?럦 Splits Saved in {OUT_DIR}")

if __name__ == "__main__":
    create_split()

