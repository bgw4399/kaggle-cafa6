import pandas as pd
import numpy as np
import os
import sys
from tqdm import tqdm
import collections

# Config
TRAIN_FASTA = "./data/raw/train/train_sequences.fasta"
TRAIN_TERMS = "./data/raw/train/train_terms.tsv"
DIAMOND_DB = "train_data.dmnd"
DIAMOND_OUT = "train_stacking_hits.tsv"
OUT_FEATURES = "train_stacking_features.tsv"
TERM_COUNTS_FILE = "./data/derived/term_counts.tsv"

def run_diamond_train_vs_train():
    if not os.path.exists(DIAMOND_DB):
        print("?뵪 Building Diamond DB...")
        # Running inside WSL, so call diamond directly
        cmd_db = f"/root/miniconda3/bin/diamond makedb --in {TRAIN_FASTA} -d {DIAMOND_DB}"
        os.system(cmd_db)
        
    if not os.path.exists(DIAMOND_OUT):
        print("?룂 Running Diamond (Train vs Train)...")
        # Running inside WSL, so call diamond directly
        cmd_run = f"/root/miniconda3/bin/diamond blastp -q {TRAIN_FASTA} -d {DIAMOND_DB} -o {DIAMOND_OUT} --outfmt 6 qseqid sseqid pident evalue bitscore --sensitive -k 25" 
        os.system(cmd_run)
    else:
        print(f"??Diamond hits found ({DIAMOND_OUT}). Skipping run.")

def generate_features():
    print("?뱤 Generating Stacking Features...")
    
    # 1. Load Ground Truth
    print("   ?뱿 Loading Ground Truth...")
    # id -> set of terms
    gt = collections.defaultdict(set)
    df_terms = pd.read_csv(TRAIN_TERMS, sep="\t", names=["id", "term", "aspect"])
    
    # Pre-calculate Term Counts (Priors)
    term_counts = df_terms['term'].value_counts().to_dict()
    total_proteins = df_terms['id'].nunique()
    
    # Save Term Counts for Test time
    df_counts = pd.DataFrame(list(term_counts.items()), columns=['term', 'count'])
    df_counts['freq'] = df_counts['count'] / total_proteins
    df_counts.to_csv(TERM_COUNTS_FILE, sep="\t", index=False)
    print(f"   ?뮶 Saved Term Counts to {TERM_COUNTS_FILE}")

    # Build GT map
    # We only care about terms that appear in training
    for pid, term in tqdm(zip(df_terms['id'], df_terms['term']), total=len(df_terms), desc="Building GT"):
        gt[pid].add(term)
        
    # 2. Process Diamond Hits
    print("   ?봽 Processing Hits...")
    
    features = []
    # Columns: [q_id, term, pident, bitscore, log_evalue, term_freq, label]
    
    # We need to map sseqid -> terms because Diamond gives us Neighbor ID
    # In Train-vs-Train, sseqid is also in train_terms.
    # So we construct s_id_to_terms map
    s_id_to_terms = gt # It matches exactly for Train-vs-Train
    
    chunk_size = 100000
    rows = []
    
    def clean_id(raw_id):
        # Format: sp|ID|NAME -> ID
        # Or just ID
        if '|' in raw_id:
            try:
                return raw_id.split('|')[1]
            except:
                return raw_id
        return raw_id
    
    with open(DIAMOND_OUT, 'r') as f:
        for line in tqdm(f, desc="Parsing Hits"):
            parts = line.strip().split('\t')
            if len(parts) < 5: continue
            
            # Clean IDs
            q_id = clean_id(parts[0])
            s_id = clean_id(parts[1])
            
            if q_id == s_id: continue # Remove Self Hits (Target Leakage)
            
            try:
                pident = float(parts[2]) / 100.0 # Normalize 0-1
                evalue = float(parts[3])
                bitscore = float(parts[4])
            except:
                continue
                
            # Log Evalue handling
            log_evalue = -np.log10(evalue + 1e-50)
            
            # Retrieve Terms from Neighbor (s_id)
            neighbor_terms = s_id_to_terms.get(s_id, set())
            
            # Ground Truth for Query (q_id)
            q_truth = gt.get(q_id, set())
            
            # For each transferred term, create a training sample
            for term in neighbor_terms:
                # Feature Vector
                term_freq = term_counts.get(term, 0) / total_proteins
                
                # Label: Is this term actually in Query's GT?
                label = 1 if term in q_truth else 0
                
                rows.append({
                    "pident": pident,
                    "bitscore": bitscore,
                    "log_evalue": log_evalue,
                    "term_freq": term_freq,
                    "label": label
                })
                
            if len(rows) > 1000000:
                 # Flush to list to avoid OOM
                 features.extend(rows)
                 rows = []
                 
    features.extend(rows)
    
    # 3. Create DataFrame
    print(f"   ?뵪 Building DataFrame ({len(features)} samples)...")
    df = pd.DataFrame(features)
    
    # Downsample Negatives if too huge?
    # Stacking often deals with imbalanced data. XGBoost scale_pos_weight can handle it.
    # But let's check size.
    
    print(f"   ?뮶 Saving to {OUT_FEATURES}...")
    df.to_csv(OUT_FEATURES, sep="\t", index=False)
    print("??Done.")

if __name__ == "__main__":
    run_diamond_train_vs_train()
    generate_features()

