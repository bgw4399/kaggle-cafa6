import sys
import os
import pandas as pd
from tqdm import tqdm

FILE_BASE = "./results/submission_378.tsv"
OUTPUT_FILE = "./results/submission_Rare_Penalty.tsv"
TRAIN_TERMS = "./data/raw/train/train_terms.tsv"

# Config
RARE_THRESHOLD = 50   # If term appears < 50 times in training
PENALTY_FACTOR = 0.5  # Reduce score by 50%

def prune():
    print("?? Pruning Rare Terms (Negative Strategy)...")
    
    # 1. Load Train Counts
    print("   ?뱴 Loading Training Counts...")
    if not os.path.exists(TRAIN_TERMS):
        print("??Train terms file missing!")
        return
        
    df_train = pd.read_csv(TRAIN_TERMS, sep='\t')
    term_counts = df_train['term'].value_counts().to_dict()
    print(f"     Known Terms: {len(term_counts):,}")
    
    # 2. Stream and Penalty
    print(f"   ?뱣 Applying Penalty (Threshold={RARE_THRESHOLD}, Factor={PENALTY_FACTOR})...")
    
    with open(FILE_BASE, 'r') as f_in, open(OUTPUT_FILE, 'w') as f_out:
        penalized_count = 0
        
        for line in tqdm(f_in):
            parts = line.strip().split('\t')
            if len(parts) < 3: continue
            
            term = parts[1]
            try:
                score = float(parts[2])
            except:
                continue
            
            # Check Rare
            count = term_counts.get(term, 0)
            if count < RARE_THRESHOLD:
                score *= PENALTY_FACTOR
                penalized_count += 1
            
            if score > 0.001:
                f_out.write(f"{parts[0]}\t{term}\t{score:.5f}\n")
                
    print(f"?럦 Pruning Complete!")
    print(f"   Penalized Terms: {penalized_count:,}")
    print(f"   Saved: {OUTPUT_FILE}")

if __name__ == "__main__":
    prune()


