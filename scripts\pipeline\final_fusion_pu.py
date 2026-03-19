import pandas as pd
import os
from tqdm import tqdm

# Config
FILE_BASE = "./results/final_submission/submission_Final_Repaired.tsv" # The 0.378 Champion
FILE_PU = "./results/pu_learning/submission_PU_ESM.tsv" # The PU Challenger
OUTPUT_FILE = "./results/final_submission/submission_Final_Fusion_PU.tsv"

def main():
    print("🚀 Final Fusion: Baseline + PU Learning...")
    
    # 1. Load PU Scores (Challenger)
    print(f"   📥 Loading PU Predictions from {FILE_PU}...")
    pu_scores = {}
    
    with open(FILE_PU, 'r') as f:
        for line in tqdm(f, desc="Reading PU"):
            parts = line.strip().split('\t')
            if len(parts) < 3: continue
            pid = parts[0]
            term = parts[1]
            try:
                score = float(parts[2])
            except: continue
            
            # Optimization: Only keep significant PU scores to save RAM
            if score > 0.1: 
                pu_scores[(pid, term)] = score
                
    print(f"      Loaded {len(pu_scores):,} significant PU predictions (>0.1).")
    
    # 2. Stream Base File and Apply Max
    print(f"   🔄 Merging with Baseline {FILE_BASE}...")
    f_out = open(OUTPUT_FILE, 'w')
    
    updated_count = 0
    injected_count = 0
    used_pu_keys = set()
    
    with open(FILE_BASE, 'r') as f:
        # Header check
        pos = f.tell()
        line = f.readline()
        if "Term" not in line and "term" not in line:
            f.seek(pos)
        else:
            # If base has header, we might want to skip it or write it?
            # Standard CAFA format has NO header.
            # But previous steps might have added one?
            # Let's check if line looks like header
            if "Term" in line: pass # Skip header
            else: f.seek(pos)
            
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
            
            final_score = base_score
            
            if key in pu_scores:
                pu_score = pu_scores[key]
                used_pu_keys.add(key)
                
                if pu_score > base_score:
                    final_score = pu_score
                    updated_count += 1
            
            # Thresholding to keep file clean
            if final_score > 0.001:
                f_out.write(f"{pid}\t{term}\t{final_score:.5f}\n")
                
    # 3. Inject Novel PU Predictions (That were not in Base)
    print("   ➕ Injecting Novel PU Predictions...")
    for key, score in tqdm(pu_scores.items(), desc="Injecting"):
        if key not in used_pu_keys:
            pid, term = key
            # Only inject if score is reasonably high (Precision Control)
            # PU can be noisy, so we trust it if it's confident
            if score > 0.3: 
                f_out.write(f"{pid}\t{term}\t{score:.5f}\n")
                injected_count += 1
                
    f_out.close()
    
    print(f"✅ Fusion Complete.")
    print(f"   Boosted (PU > Base): {updated_count:,}")
    print(f"   Injected (PU Only): {injected_count:,}")
    print(f"📁 Output: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
