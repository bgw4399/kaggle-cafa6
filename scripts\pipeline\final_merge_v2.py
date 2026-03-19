import pandas as pd
import numpy as np
import os
import gc
from tqdm import tqdm

# =========================================================
# ?숋툘 [Config] New Strategy
# =========================================================
# Input Files
DL_FILE = './results/pred_esm2_15B_FINAL.tsv'          # New Model (Structured)
DIAMOND_FILE = './results/submission_diamond_taxon_filtered.tsv' # Homology (High Precision)
GAF_NEG_FILE = './gaf_negative_preds.tsv'              # Negative Filter
TEST_IDS_FILE = './data/embeddings/testsuperset_ids.npy' # Target IDs

# Output
OUTPUT_FILE = './results/submission_ESM15B_Ensemble_Weighted.tsv'

# Weights (Ensemble Strategy)
# Model is 50%, Diamond is 50%.
# If a term is in both, Score = 0.5*Model + 0.5*Diamond
# If only in one, Score = 0.5*Score (Penalized implicitly? No, we should assume missing = 0)
W_DL = 0.5
W_DIA = 0.5

# Chunk Size for Low RAM
CHUNK_SIZE = 5_000_000

# =========================================================

def clean_id_str(pid):
    if isinstance(pid, bytes): pid = pid.decode('utf-8')
    pid = str(pid).strip().replace('>', '')
    if '|' in pid: 
        parts = pid.split('|')
        pid = parts[1] if len(parts) >= 2 else pid
    return pid

def main():
    print("?? Final Merge V2: ESM2-15B + Diamond (Weighted Ensemble)...")
    
    # Check Inputs
    if not os.path.exists(DL_FILE):
        print(f"?슚 Error: Model file not found: {DL_FILE}")
        print("   Did training finish? If not, please wait or use BCE/ASL file.")
        return

    # 1. Load Valid IDs
    print(f"   ?뱥 Loading Valid Test IDs...")
    if os.path.exists(TEST_IDS_FILE):
        raw_ids = np.load(TEST_IDS_FILE)
        valid_test_ids = set([clean_id_str(pid) for pid in raw_ids])
        del raw_ids
        print(f"     -> {len(valid_test_ids):,} valid IDs loaded.")
    else:
        print("   ?좑툘 Test IDs file not found. Skipping ID filtering (Risky).")
        valid_test_ids = None

    # 2. Dictionary-based Accumulation (Low RAM optimized?)
    # Weighted Average requires summing scores.
    # To do this chunk-wise is tricky if split across chunks.
    # But usually files are grouped by ID.
    
    # Better approach for Low RAM Weighted Average:
    # 1. Read Diamond into Memory (It's usually smaller/sparse).
    # 2. Read DL in Chunks.
    #    For each chunk, look up Diamond score, compute average, write to disk.
    #    Remove processed Diamond entries from memory.
    # 3. After DL is done, write remaining Diamond entries (multiplied by weight).
    
    # EXCEPT: Diamond file can be large too.
    # Let's stick to the "Dump all to Temp, then Aggregate" strategy.
    # It's robust.
    
    TEMP_FILE = './results/temp_weighted_raw.tsv'
    print("   ??Streaming Weighted Scores to Temp File...")
    
    with open(TEMP_FILE, 'w') as f_out:
        # (1) Process DL File
        print(f"     Processing DL (Weight {W_DL}): {DL_FILE}")
        with open(DL_FILE, 'r') as f_in:
            for line in tqdm(f_in):
                parts = line.strip().split('\t')
                if len(parts) < 3: continue
                pid = clean_id_str(parts[0])
                if valid_test_ids and pid not in valid_test_ids: continue
                
                term = parts[1]
                score = float(parts[2])
                
                # Apply Weight Immediately
                weighted_score = score * W_DL
                if weighted_score > 0.001:
                    f_out.write(f"{pid}\t{term}\t{weighted_score:.5f}\n")

        # (2) Process Diamond File
        if os.path.exists(DIAMOND_FILE):
            print(f"     Processing Diamond (Weight {W_DIA}): {DIAMOND_FILE}")
            with open(DIAMOND_FILE, 'r') as f_in:
                for line in tqdm(f_in):
                    parts = line.strip().split('\t')
                    if len(parts) < 3: continue
                    pid = clean_id_str(parts[0])
                    if valid_test_ids and pid not in valid_test_ids: continue
                    
                    term = parts[1]
                    score = float(parts[2])
                    
                    # Apply Weight Immediately
                    weighted_score = score * W_DIA
                    if weighted_score > 0.001:
                        f_out.write(f"{pid}\t{term}\t{weighted_score:.5f}\n")
                        
        # (3) GAF Positive? (Optional - Usually 1.0)
        # If we include GAF, we should probably weight it high or max it?
        # Let's skip strict GAF enforcement for now to keep it purely probabilistic, 
        # OR add it as a separate step (Post-process Max).
        # User didn't strictly ask for GAF, but previous script had it.
        # Let's add it as 1.0 (Override).
        
    print("   ?뮶 Sorting & Aggregating (Summing Weights)...")
    
    # 3. Aggregate (SUM)
    # We wrote (Score * Weight) to file.
    # Now valid score = Sum(Weighted_Scores).
    # Since we have at most 2 entries per (ID, Term) [one from DL, one from Diamond],
    # Sum is correct.
    
    final_results = {}
    
    # Read Temp File in Chunks
    for chunk in tqdm(pd.read_csv(TEMP_FILE, sep='\t', header=None, names=['Id', 'Term', 'Score'], chunksize=CHUNK_SIZE)):
        # Sum scores for same ID-Term
        grouped = chunk.groupby(['Id', 'Term'])['Score'].sum()
        
        for (pid, term), score in grouped.items():
            current = final_results.get((pid, term), 0.0)
            final_results[(pid, term)] = current + score

    # Remove Temp
    if os.path.exists(TEMP_FILE):
        os.remove(TEMP_FILE)
        
    # 4. GAF Negative Filter
    if os.path.exists(GAF_NEG_FILE):
        print("   ?㏏ Applying GAF Negative Filter...")
        with open(GAF_NEG_FILE, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 2: continue
                pid = clean_id_str(parts[0])
                term = parts[1]
                if (pid, term) in final_results:
                    del final_results[(pid, term)]

    # 5. Save
    print(f"   ?뮶 Final Saving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        for (pid, term), score in tqdm(final_results.items()):
            # Clip to 1.0 just in case
            if score > 1.0: score = 1.0
            if score >= 0.01:
                f.write(f"{pid}\t{term}\t{score:.5f}\n")

    print(f"?럦 Final Merge V2 Completed: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()


