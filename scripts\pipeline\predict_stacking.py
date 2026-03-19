import pandas as pd
import numpy as np
import xgboost as xgb
import os
import sys
import collections
from tqdm import tqdm

# Config
TEST_FASTA = "./data/raw/test/testsuperset.fasta"
TRAIN_TERMS = "./data/raw/train/train_terms.tsv"
DIAMOND_DB = "train_data.dmnd"
DIAMOND_OUT = "test_stacking_hits.tsv"
MODEL_FILE = "xgboost_model.json"
TERM_COUNTS_FILE = "./data/derived/term_counts.tsv"
OUTPUT_FILE = "./results/final_submission/submission_Stacking_XGB.tsv"

# Make sure output dir exists
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

def run_diamond_test_vs_train():
    if not os.path.exists(DIAMOND_DB):
        print("??Diamond DB not found! Cannot run.")
        return False
        
    if not os.path.exists(DIAMOND_OUT):
        print("?룂 Running Diamond (Test vs Train)...")
        # Use absolute path for WSL
        cmd_run = f"/root/miniconda3/bin/diamond blastp -q {TEST_FASTA} -d {DIAMOND_DB} -o {DIAMOND_OUT} --outfmt 6 qseqid sseqid pident evalue bitscore --sensitive -k 50" 
        # Increased K to 50 for Test to get more candidates
        os.system(cmd_run)
    else:
        print(f"??Diamond hits found ({DIAMOND_OUT}). Skipping run.")
    return True

def clean_id(raw_id):
    if '|' in raw_id:
        try:
            return raw_id.split('|')[1]
        except:
            return raw_id
    return raw_id

def predict():
    print("?? Starting Refined Stacking Prediction...")
    
    # 1. Load Resources
    print("   ?뱿 Loading Maps (Terms & Counts)...")
    
    # GT Map (Subject -> Terms)
    # We use Train Terms to map the 'Hits' (which are train proteins) to their GO terms
    s_id_to_terms = collections.defaultdict(set)
    df_terms = pd.read_csv(TRAIN_TERMS, sep="\t", names=["id", "term", "aspect"])
    for pid, term in zip(df_terms['id'], df_terms['term']):
        s_id_to_terms[str(pid)].add(term)
        
    # Term Frequencies (Priors)
    df_counts = pd.read_csv(TERM_COUNTS_FILE, sep="\t")
    term_freq_map = dict(zip(df_counts['term'], df_counts['freq']))
    
    # Load Model
    print(f"   ?쭬 Loading XGBoost Model {MODEL_FILE}...")
    model = xgb.Booster()
    model.load_model(MODEL_FILE)
    
    # 2. Process Hits & Predict in Chunks
    print("   ?봽 Processing Hits & Predicting...")
    
    chunk_size = 50000 # Hits per chunk
    
    # Buffers
    rows = []
    
    # Output File
    f_out = open(OUTPUT_FILE, 'w')
    
    total_preds = 0
    
    with open(DIAMOND_OUT, 'r') as f:
        pbar = tqdm(desc="Predicting")
        
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 5: continue
            
            # Clean IDs
            # Query is Test, Subject is Train
            q_id = clean_id(parts[0])
            s_id = clean_id(parts[1])
            
            try:
                pident = float(parts[2]) / 100.0
                evalue = float(parts[3])
                bitscore = float(parts[4])
            except:
                continue
            
            log_evalue = -np.log10(evalue + 1e-50)
            
            # Get Candidate Terms from Subject
            candidates = s_id_to_terms.get(s_id, set())
            
            if not candidates:
                continue
                
            # Create Features for all candidates
            for term in candidates:
                term_freq = term_freq_map.get(term, 0.0)
                
                rows.append({
                    "q_id": q_id,
                    "term": term,
                    "pident": pident,
                    "bitscore": bitscore,
                    "log_evalue": log_evalue,
                    "term_freq": term_freq
                })
            
            # Batch Processing
            if len(rows) >= 500000: # Process batch of features
                process_batch(rows, model, f_out)
                total_preds += len(rows) # Approx
                rows = []
                pbar.update(chunk_size)
                
        # Final Flush
        if rows:
            process_batch(rows, model, f_out)
            
        pbar.close()
        
    f_out.close()
    print(f"??Prediction Complete.")
    print(f"?뱚 Output: {OUTPUT_FILE}")

def process_batch(rows, model, f_out):
    if not rows: return
    
    df = pd.DataFrame(rows)
    
    # Feature Order MUST match Training
    feature_cols = ["pident", "bitscore", "log_evalue", "term_freq"]
    
    dtest = xgb.DMatrix(df[feature_cols])
    preds = model.predict(dtest)
    
    # Write Results (Filter > 0.001)
    # Vectorized write formatting
    df['score'] = preds
    
    # Setup for fast write
    # We interpret score as probability
    
    mask = df['score'] > 0.001
    valid = df[mask]
    
    if valid.empty: return
    
    # Simple write (could be optimized further but this is decent)
    for _, row in valid.iterrows():
        f_out.write(f"{row['q_id']}\t{row['term']}\t{row['score']:.5f}\n")

if __name__ == "__main__":
    if run_diamond_test_vs_train():
        predict()

