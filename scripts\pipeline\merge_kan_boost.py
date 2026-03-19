import pandas as pd
import os
from tqdm import tqdm

# Config
FILE_BASE = "./results/final_submission/submission_Final_Repaired.tsv" # 0.378 Baseline
FILE_KAN = "./results/kan/submission_KAN_ESM.tsv" # 0.169 KAN
OUTPUT_FILE = "./results/final_submission/submission_KAN_Boosted.tsv"

def main():
    print("🚀 Boosting Baseline with KAN Signals...")
    
    # 1. Load KAN Scores into Memory (It's small, ~250MB)
    kan_scores = {}
    print(f"   📥 Loading KAN Predictions from {FILE_KAN}...")
    
    with open(FILE_KAN, 'r') as f:
        for line in tqdm(f, desc="Reading KAN"):
            parts = line.strip().split('\t')
            if len(parts) < 3: continue
            pid = parts[0]
            term = parts[1]
            try:
                score = float(parts[2])
            except:
                continue
                
            # Store
            kan_scores[(pid, term)] = score
            
    print(f"      Loaded {len(kan_scores):,} KAN predictions.")
    
    # 2. Stream Base File and Apply Max
    print(f"   🔄 Merging with Base {FILE_BASE}...")
    f_out = open(OUTPUT_FILE, 'w')
    
    # Track used
    used_kan_keys = set()
    
    updated_count = 0
    
    with open(FILE_BASE, 'r') as f:
        for line in tqdm(f, desc="Merging"):
            parts = line.strip().split('\t')
            if len(parts) < 3:
                f_out.write(line)
                continue
                
            pid = parts[0]
            term = parts[1]
            try:
                base_score = float(parts[2])
            except:
                f_out.write(line)
                continue
            
            key = (pid, term)
            
            if key in kan_scores:
                kan_score = kan_scores[key]
                used_kan_keys.add(key)
                
                # Logic: MAX Merge
                if kan_score > base_score:
                    f_out.write(f"{pid}\t{term}\t{kan_score:.5f}\n")
                    updated_count += 1
                else:
                    f_out.write(line)
            else:
                f_out.write(line)
                
    # 3. Inject Remaining KAN (Novel Predictions)
    print("   ➕ Injecting Novel KAN Predictions...")
    injected_count = 0
    for key, score in tqdm(kan_scores.items(), desc="Injecting"):
        if key not in used_kan_keys:
            pid, term = key
            f_out.write(f"{pid}\t{term}\t{score:.5f}\n")
            injected_count += 1
            
    f_out.close()
    
    print(f"✅ Boost Complete.")
    print(f"   Boosted (KAN > Base): {updated_count:,}")
    print(f"   Injected (KAN Only): {injected_count:,}")
    print(f"📁 Output: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
