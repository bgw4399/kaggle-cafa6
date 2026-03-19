import sys
import os
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

# Config
FILE_ESM = "./results/final_submission/final_esm_full.tsv"
FILE_PROT = "./results/final_submission/final_prott5_full.tsv"
OUTPUT_FILE = "./results/final_submission/submission_Final_Ensemble_Scientific.tsv"

# Optimal Weights from Validation
WEIGHT_ESM = 0.6
WEIGHT_PROT = 0.4

def load_scores(path, name):
    print(f"   📥 Loading {name} from {path}...")
    scores = defaultdict(dict)
    if not os.path.exists(path):
        print(f"❌ File not found: {path}")
        return scores
        
    with open(path, 'r') as f:
        for line in tqdm(f):
            p = line.strip().split('\t')
            if len(p)<3: continue
            pid, term, val = p[0], p[1], float(p[2])
            scores[pid][term] = val
    return scores

def main():
    print(f"🚀 Generating Final Ensemble (Scientific)...")
    print(f"   Weights: ESM={WEIGHT_ESM}, Prot={WEIGHT_PROT}")
    
    # 1. Load Data
    s_esm = load_scores(FILE_ESM, "ESM2-15B")
    s_prot = load_scores(FILE_PROT, "ProtT5-XL")
    
    # Check if loaded
    if not s_esm or not s_prot:
        print("❌ Missing input files. Training might not be complete.")
        return

    # 2. Merge
    print("   ⚗️ Merging Scores...")
    # Get all proteins
    all_pids = set(s_esm.keys()).union(s_prot.keys())
    
    # Stream write for memory efficiency?
    # No, we need to sort or process. But writing line by line is fine if we process by PID.
    
    merged_count = 0
    
    with open(OUTPUT_FILE, 'w') as f_out:
        for pid in tqdm(all_pids, desc="Merging"):
            # Get union of terms
            terms = set(s_esm.get(pid, {}).keys()).union(s_prot.get(pid, {}).keys())
            
            pid_scores = []
            for term in terms:
                v1 = s_esm.get(pid, {}).get(term, 0.0)
                v2 = s_prot.get(pid, {}).get(term, 0.0)
                
                final_score = (v1 * WEIGHT_ESM) + (v2 * WEIGHT_PROT)
                
                if final_score > 0.001:
                    pid_scores.append((term, final_score))
            
            # Sort by score descending (optional but good for submission)
            pid_scores.sort(key=lambda x: x[1], reverse=True)
            
            for term, score in pid_scores:
                f_out.write(f"{pid}\t{term}\t{score:.5f}\n")
                merged_count += 1
                
    print(f"🎉 Ensemble Complete!")
    print(f"   Total Predictions: {merged_count:,}")
    print(f"   Saved: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
