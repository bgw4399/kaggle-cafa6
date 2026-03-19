import sys
import os
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

FILE_INPUT = "./results/submission_378.tsv"
# We will create 2 versions to be safe
FILE_OUT_50 = "./results/submission_378_Top50.tsv"
FILE_OUT_75 = "./results/submission_378_Top75.tsv"

def filter_topk():
    print("?? applying Top-K Filtering to Best_378...")
    
    # 1. Load Data
    print("   ?뱿 Reading Data...")
    data = defaultdict(list)
    with open(FILE_INPUT, 'r') as f:
        for line in tqdm(f):
            parts = line.strip().split('\t')
            if len(parts) < 3: continue
            # Store (term, score)
            data[parts[0]].append((parts[1], float(parts[2])))
            
    # 2. Process
    print(f"   ?㎚ Processing {len(data):,} proteins...")
    
    with open(FILE_OUT_50, 'w') as f50, open(FILE_OUT_75, 'w') as f75:
        for pid, preds in tqdm(data.items()):
            # Sort by score descending
            preds.sort(key=lambda x: x[1], reverse=True)
            
            # Write Top 50
            for i, (term, score) in enumerate(preds):
                if score < 0.001: break # Minimum score cutoff
                
                line = f"{pid}\t{term}\t{score:.5f}\n"
                
                if i < 75:
                    f75.write(line)
                if i < 50:
                    f50.write(line)
                    
    print(f"?럦 Done!")
    print(f"   Saved: {FILE_OUT_50}")
    print(f"   Saved: {FILE_OUT_75}")

if __name__ == "__main__":
    filter_topk()

