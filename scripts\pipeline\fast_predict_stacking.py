import pandas as pd
import numpy as np
import xgboost as xgb
import os
import sys

# Config
DIAMOND_OUT = "test_stacking_hits.tsv"
TRAIN_TERMS = "./data/raw/train/train_terms.tsv"
MODEL_FILE = "xgboost_model.json"
TERM_COUNTS_FILE = "./data/derived/term_counts.tsv"
OUTPUT_FILE = "./results/final_submission/submission_Stacking_XGB.tsv"

def clean_id_vectorized(series):
    # Vectorized ID cleaning
    # Extract second part if pipe exists, else keep original
    # Simple regex or string split
    return series.str.split('|').str[1].fillna(series)

def predict_fast():
    print("?? Starting FAST Stacking Prediction...")

    # 1. Load Resources
    print("   ?뱿 Loading Maps...")
    
    # Train Terms (ID -> Term)
    # We need a DataFrame for merging
    df_terms = pd.read_csv(TRAIN_TERMS, sep="\t", names=["id", "term", "aspect"])
    # Clean IDs in Train Terms if needed?
    # Usually Train Terms IDs are clean (e.g. P12345).
    # But check if we need to clean them?
    # Our clean_id function strips sp|...|...
    # Train Terms usually has just the ID.
    
    # Term Counts
    df_counts = pd.read_csv(TERM_COUNTS_FILE, sep="\t")
    # Map term -> freq
    df_freq = df_counts.set_index('term')['freq']
    
    # Load Model
    print(f"   ?쭬 Loading XGBoost Model {MODEL_FILE}...")
    model = xgb.Booster()
    model.load_model(MODEL_FILE)
    
    # 2. Process in Chunks
    print("   ?봽 Processing in Vectorized Chunks...")
    
    # Output File
    header = False
    
    # Diamond Columns
    d_cols = ["q_id", "s_id", "pident", "evalue", "bitscore"]
    
    processed_count = 0
    
    # Use chunksize for Diamond Hits
    for df_chunk in pd.read_csv(DIAMOND_OUT, sep="\t", names=d_cols, chunksize=500000):
        # 2.1 Clean IDs
        # q_id is Test (might differ from fasta header, but Diamond input was fasta)
        # s_id is Train (e.g. sp|P12345|NAME)
        df_chunk['q_id'] = clean_id_vectorized(df_chunk['q_id'])
        df_chunk['s_id'] = clean_id_vectorized(df_chunk['s_id'])
        
        # 2.2 Merge to get Candidate Terms
        # Left Join on s_id == id
        merged = df_chunk.merge(df_terms, left_on='s_id', right_on='id', how='left')
        
        # Drop rows where no terms found (NaN term)
        merged = merged.dropna(subset=['term'])
        
        if merged.empty:
            continue
            
        # 2.3 Calculate Features
        # pident: 0-100 -> 0-1
        merged['pident'] = merged['pident'] / 100.0
        
        # log_evalue
        merged['log_evalue'] = -np.log10(merged['evalue'] + 1e-50)
        
        # term_freq (Map)
        merged['term_freq'] = merged['term'].map(df_freq).fillna(0.0)
        
        # 2.4 Prepare DMatrix
        # Features: pident, bitscore, log_evalue, term_freq
        X = merged[["pident", "bitscore", "log_evalue", "term_freq"]]
        dtest = xgb.DMatrix(X)
        
        # 2.5 Predict
        preds = model.predict(dtest)
        
        # 2.6 Filter & Write
        merged['score'] = preds
        
        # Filter low scores
        final_df = merged[merged['score'] > 0.001][['q_id', 'term', 'score']]
        
        if not final_df.empty:
            final_df.to_csv(OUTPUT_FILE, sep="\t", index=False, header=False, mode='a')
            
        processed_count += len(df_chunk)
        print(f"   ??Processed {processed_count:,} hits...")
        
    print(f"?럦 Prediction Complete: {OUTPUT_FILE}")

if __name__ == "__main__":
    predict_fast()

