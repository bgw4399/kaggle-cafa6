import pandas as pd
import numpy as np
import os
import gc
from tqdm import tqdm

# =========================================================
# ?숋툘 SOTA Ensemble Config
# =========================================================
ESM_FILE = "./results/submission_378.tsv"
PROT_FILE = "./results/pred_prott5_resmlp_focal.tsv"
ANKH_FILE = "./results/pred_ankh_resmlp_focal.tsv"
OUTPUT_FILE = "./results/submission_SOTA_Ensemble_Final.tsv"

# Weights
W_ESM = 0.6
W_OTHERS = 0.4  # (Prot + Ankh) / 2

def load_to_dict(path):
    print(f"?뱰 Loading {os.path.basename(path)}...")
    data = {}
    with open(path, 'r') as f:
        for line in tqdm(f):
            parts = line.strip().split('\t')
            if len(parts) < 3: continue
            pid, term = parts[0], parts[1]
            score = float(parts[2])
            data[(pid, term)] = score
    return data

print("?? SOTA Rank Ensemble Start...")

# 1. Load All (High RAM usage but safest for Rank)
d_esm = load_to_dict(ESM_FILE)
d_prot = load_to_dict(PROT_FILE)
d_ankh = load_to_dict(ANKH_FILE)

# 2. Collect All Keys
all_keys = set(d_esm.keys()) | set(d_prot.keys()) | set(d_ankh.keys())
print(f"   Total Unique Predictions: {len(all_keys):,}")

# 3. Calculate Scores
# Normalizing scores by Rank is safer when distributions differ
# But here we use weighted sum of probabilities for simplicity first, 
# assuming models are calibrated (Sigmoid).
# If simple sum fails, we will try Rank.

print("   ?뽳툘 Calculating Weighted Sum...")
results = []
for k in tqdm(all_keys):
    s_esm = d_esm.get(k, 0.0)
    s_prot = d_prot.get(k, 0.0)
    s_ankh = d_ankh.get(k, 0.0)
    
    # Combined New Models
    s_new = (s_prot + s_ankh) / 2
    
    # Final Score
    final_score = (s_esm * W_ESM) + (s_new * W_OTHERS)
    
    if final_score > 0.01:
        results.append((k[0], k[1], final_score))

# 4. Save
print(f"   ?뮶 Saving {len(results):,} predictions...")
with open(OUTPUT_FILE, 'w') as f:
    for pid, term, score in tqdm(results):
        f.write(f"{pid}\t{term}\t{score:.5f}\n")

print(f"?럦 SOTA Ensemble Completed: {OUTPUT_FILE}")

