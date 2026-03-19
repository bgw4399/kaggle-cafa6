import sys
import os
from tqdm import tqdm

# Config
FILE_BASE = "./results/final_submission/submission_Final_Repaired.tsv"  # The 0.378 Champion
FILE_DIAMOND = "./temp_diamond.tsv"                     # The Homology Hits
OUTPUT_FILE = "./results/final_submission/submission_Homology_Dominance.tsv"

THRESHOLD = 0.8 # Trust Diamond if score >= 0.8

def main():
    print("🚀 Homology Dominance Strategy Starting...")
    print(f"   Base File: {FILE_BASE}")
    print(f"   Homology File: {FILE_DIAMOND}")
    print(f"   Threshold: Score >= {THRESHOLD} -> Force to 1.0")

    # 1. Load Diamond High Confidence Hits
    print("   📥 Loading High-Confidence Diamond Hits...")
    diamond_hits = {} # (id, term) -> score
    count_dia = 0
    
    with open(FILE_DIAMOND, 'r') as f:
        for line in tqdm(f, desc="Reading Diamond"):
            parts = line.strip().split('\t')
            if len(parts) < 3: continue
            
            pid = parts[0]
            term = parts[1]
            try:
                score = float(parts[2])
            except:
                continue
                
            if score >= THRESHOLD:
                # Force to 1.0 (or keep raw score if it's high enough)
                # Let's force to 1.0 to be decisive. Or 0.99?
                # If score is 0.99, keep 0.99. If 1.0, keep 1.0.
                # Actually, simply using the Diamond score is safer than forcing 1.0 blindly?
                # But user wants "Dominance".
                # If Diamond says 0.8, and DL says 0.2.
                # If we use 0.8, it dominates.
                # Let's use the exact Diamond score.
                diamond_hits[(pid, term)] = score
                count_dia += 1
                
    print(f"   ✅ Loaded {count_dia:,} High-Confidence Hits.")
    
    # 2. Process Base File and Override
    print("   🔄 Processing Base File & Overriding...")
    
    # We will stream the base file and check against diamond_hits.
    # Note: If Diamond has a term NOT in Base, we should ADD it.
    # But Base usually has all terms (if coverage is good).
    # If Base misses it, we should append it.
    # To do this efficiently, we track which diamond hits were used.
    
    used_diamond_keys = set()
    f_out = open(OUTPUT_FILE, 'w')
    
    with open(FILE_BASE, 'r') as f:
        for line in tqdm(f, desc="Applying Dominance"):
            parts = line.strip().split('\t')
            if len(parts) < 3: 
                f_out.write(line)
                continue
                
            pid = parts[0]
            term = parts[1]
            
            # Check overlap
            if (pid, term) in diamond_hits:
                # OVERRIDE
                d_score = diamond_hits[(pid, term)]
                f_out.write(f"{pid}\t{term}\t{d_score:.5f}\n")
                used_diamond_keys.add((pid, term))
            else:
                # Keep original
                f_out.write(line)
                
    # 3. Add Missing Diamond Hits (New Discoveries)
    print("   ➕ Adding Missing High-Confidence Terms...")
    count_added = 0
    for key, score in tqdm(diamond_hits.items(), desc="Injecting"):
        if key not in used_diamond_keys:
            pid, term = key
            f_out.write(f"{pid}\t{term}\t{score:.5f}\n")
            count_added += 1
            
    f_out.close()
    print(f"✅ Done! Overridden: {len(used_diamond_keys):,}, Added: {count_added:,}")
    print(f"📁 Output: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
