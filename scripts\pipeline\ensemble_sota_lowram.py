import pandas as pd
import numpy as np
import os
import gc
from tqdm import tqdm

# =========================================================
# ?숋툘 SOTA Ensemble Config (Low RAM)
# =========================================================
ESM_FILE = "./results/submission_378.tsv"
PROT_FILE = "./results/pred_prott5_resmlp_focal.tsv"
ANKH_FILE = "./results/pred_ankh_resmlp_focal.tsv"
OUTPUT_FILE = "./results/submission_SOTA_Ensemble_Final.tsv"
TEMP_FILE = "./results/temp_ensemble_full.tsv"

# Final Weights
# ESM: 0.6
# Others: 0.4 (split equally -> 0.2 each)
W_ESM = 0.6
W_PROT = 0.2
W_ANKH = 0.2

def stream_processed(input_file, output_handle, weight, desc):
    if not os.path.exists(input_file):
        print(f"?좑툘 Missing file: {input_file}")
        return
        
    print(f"   ?뙄 Streaming {desc} (w={weight})...")
    with open(input_file, 'r') as f:
        for line in tqdm(f):
            parts = line.strip().split('\t')
            if len(parts) < 3: continue
            
            try:
                pid = parts[0]
                term = parts[1]
                score = float(parts[2])
                
                # Apply weight immediately
                weighted_score = score * weight
                
                # Write to temp
                output_handle.write(f"{pid}\t{term}\t{weighted_score:.6f}\n")
            except:
                continue

print("?? SOTA Ensemble (Low RAM) Start...")

# 1. Stream all weighted scores to a single file
with open(TEMP_FILE, 'w') as f_out:
    stream_processed(ESM_FILE, f_out, W_ESM, "ESM (Original)")
    stream_processed(PROT_FILE, f_out, W_PROT, "ProtT5")
    stream_processed(ANKH_FILE, f_out, W_ANKH, "Ankh")

print("   ?뮶 Aggregating (Summing weighted scores)...")

# 2. Chunk processing for Aggregation
chunk_size = 5_000_000
final_results = {}

# We cannot iterate effectively if not sorted, BUT using a dict for aggregation 
# works if memory allows. If 40M unique pairs -> ~4GB RAM. It might be tight but doable.
# If previous script crashed, it was likely holding 3 dicts (12GB+). 
# Holding 1 dict is 3x better.

# To be safer: read chunk, group, then merge to main dict.
# Actually, if the file is not sorted, 'Id'/'Term' can appear anywhere.
# The safest Low-RAM way is to use system SORT first, but Windoes/WSL sort can be tricky with temp space.
# We will try the Dict approach first (it's 1/3rd the RAM of the previous attempt).

for chunk in tqdm(pd.read_csv(TEMP_FILE, sep='\t', names=['Id', 'Term', 'Score'], header=None, chunksize=chunk_size)):
    # Sum scores for same ID-Term within chunk
    grouped = chunk.groupby(['Id', 'Term'])['Score'].sum()
    
    for (pid, term), score in grouped.items():
        if (pid, term) in final_results:
            final_results[(pid, term)] += score
        else:
            final_results[(pid, term)] = score
            
    # Explicit garbage collection
    del chunk
    del grouped
    gc.collect()

# Clean temp
if os.path.exists(TEMP_FILE):
    os.remove(TEMP_FILE)

# 3. Save
print(f"   ?뮶 Saving Final Output to {OUTPUT_FILE}...")
with open(OUTPUT_FILE, 'w') as f:
    for (pid, term), score in tqdm(final_results.items()):
        if score > 0.001: # Filter very low scores
            f.write(f"{pid}\t{term}\t{score:.5f}\n")

print(f"?럦 Done! Saved: {OUTPUT_FILE}")

