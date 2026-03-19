import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from collections import defaultdict

# =========================================================
# ?숋툘 Optimization Configuration
# =========================================================
INPUT_FILE = "./results/submission_378.tsv"
OUTPUT_FILE = "./results/submission_378_Optimized.tsv"

TRAIN_TERMS = "./data/raw/train/train_terms.tsv"
GAF_NEG_FILE = "./gaf_negative_preds.tsv"
OBO_FILE = "./data/raw/train/go-basic.obo"

# Tuning Params
FREQ_BOOST = 1.2       # Frequent terms score multiplier
RARE_PENALTY = 0.5     # Rare terms score multiplier
FREQ_THRESHOLD = 500   # Terms with >500 training samples are "Frequent"
RARE_THRESHOLD = 50    # Terms with <50 training samples are "Rare"

def load_train_counts():
    print("?뱴 Counting Training Terms...")
    df = pd.read_csv(TRAIN_TERMS, sep='\t')
    return df['term'].value_counts()

def optimize():
    print(f"?? Optimizing {INPUT_FILE}...")
    
    # 1. Load Frequency Data
    term_counts = load_train_counts()
    
    # 2. Load GAF Negative (Set for O(1) lookup)
    print("   ?㏏ Loading GAF Negative...")
    gaf_neg = set()
    if os.path.exists(GAF_NEG_FILE):
        with open(GAF_NEG_FILE, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    k = (parts[0], parts[1])
                    gaf_neg.add(k)
    print(f"     -> {len(gaf_neg):,} negative filters loaded.")

    # 3. Stream & Process
    print("   ??streaming and optimizing...")
    with open(INPUT_FILE, 'r') as f_in, open(OUTPUT_FILE, 'w') as f_out:
        for line in tqdm(f_in):
            parts = line.strip().split('\t')
            if len(parts) < 3: continue
            
            pid = parts[0]
            term = parts[1]
            try: score = float(parts[2])
            except: continue
            
            # (1) GAF Negative Filter
            if (pid, term) in gaf_neg:
                continue # Skip (Delete)
            
            # (2) Frequency Scaling
            count = term_counts.get(term, 0)
            
            if count >= FREQ_THRESHOLD:
                score *= FREQ_BOOST
                if score > 1.0: score = 1.0
            elif count <= RARE_THRESHOLD:
                score *= RARE_PENALTY
            
            # (3) Write if valid
            if score > 0.001:
                f_out.write(f"{pid}\t{term}\t{score:.5f}\n")

    print(f"?럦 Optimization Complete: {OUTPUT_FILE}")

if __name__ == "__main__":
    optimize()


