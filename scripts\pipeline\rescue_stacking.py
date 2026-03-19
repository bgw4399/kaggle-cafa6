import pandas as pd
import os
import sys
from tqdm import tqdm
import collections

# Config
FILE_BASE = "./results/final_submission/submission_Final_Repaired.tsv" # The 0.378 Baseline (High Coverage)
FILE_STACKING = "./results/final_submission/submission_Stacking_XGB.tsv" # The 0.22 Stacking (High Precision)
OUTPUT_FILE = "./results/final_submission/submission_Hybrid_Rescue.tsv"

def main():
    print("🚑 Rescue Mission: Merging Stacking Precision with DL Coverage...")
    
    # 1. Load Stacking Scores accurately
    # Stacking file is 836MB, might fit in RAM?
    # 30M lines -> Approx 1.5GB RAM for Dict. feasible.
    print(f"   📥 Loading Stacking Scores from {FILE_STACKING}...")
    stacking_scores = {}
    
    try:
        # Check size first
        size_mb = os.path.getsize(FILE_STACKING) / (1024*1024)
        print(f"      Size: {size_mb:.1f} MB. Loading into memory...")
        
        with open(FILE_STACKING, 'r') as f:
            for line in tqdm(f, desc="Reading Stacking"):
                parts = line.strip().split('\t')
                if len(parts) < 3: continue
                pid = parts[0]
                term = parts[1]
                score = float(parts[2])
                stacking_scores[(pid, term)] = score
                
        print(f"      ✅ Loaded {len(stacking_scores):,} Stacking Predictions.")
        
    except MemoryError:
        print("      ❌ OOM! Cannot load Stacking file. Aborting Rescue.")
        return

    # 2. Stream Base File and Merge
    print(f"   🔄 Streaming Base File {FILE_BASE} and Overriding...")
    
    f_out = open(OUTPUT_FILE, 'w')
    
    # Track which stacking hits were used
    used_stacking_keys = set()
    
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
            
            if key in stacking_scores:
                # OVERRIDE with Stacking Score
                # Stacking score is calibrated probability (0-1)
                # Usually we trust it more. 
                # Strategy: Max? Average? Validated Stacking AUC 0.77 implies it's good.
                # Let's use Stacking Score directly.
                new_score = stacking_scores[key]
                f_out.write(f"{pid}\t{term}\t{new_score:.5f}\n")
                used_stacking_keys.add(key)
            else:
                # Keep Base Score (DL prediction for non-homology)
                f_out.write(line)
                
    # 3. Add Missing Stacking Hits (if any)
    # Stacking might have found something DL missed? Unlikely but possible.
    print("   ➕ Adding any missing Stacking Hits...")
    added_count = 0
    for key, score in tqdm(stacking_scores.items(), desc="Injecting"):
        if key not in used_stacking_keys:
            pid, term = key
            f_out.write(f"{pid}\t{term}\t{score:.5f}\n")
            added_count += 1
            
    f_out.close()
    
    print(f"✅ Rescue Complete.")
    print(f"   Used Overrides: {len(used_stacking_keys):,}")
    print(f"   Injected New: {added_count:,}")
    print(f"📁 Output: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
