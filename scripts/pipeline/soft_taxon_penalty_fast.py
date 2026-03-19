import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import os

# Optimized Taxonomy Filter
# Faster ID loading (set lookup)
# Vectorized/Chunked processing where possible, or just optimized loop

def clean_id_str(pid):
    if isinstance(pid, bytes):
        pid = pid.decode("utf-8")
    return str(pid).strip().replace(">", "").split("|")[1] if "|" in str(pid) else str(pid).strip()

def load_mapping(npy_ids, npy_tax):
    print(f"   ?뱿 Loading Mapping: {npy_ids}")
    raw_ids = np.load(npy_ids, allow_pickle=True).reshape(-1)
    # Fast clean
    clean_ids = [clean_id_str(x) for x in raw_ids]
    
    tax = np.load(npy_tax, allow_pickle=True).reshape(-1)
    
    return dict(zip(clean_ids, tax))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_in", default="./results/final_submission/submission_Final_Hybrid.tsv")
    ap.add_argument("--pred_out", default="./results/final_submission/submission_Final_Hybrid_Taxon_Filtered.tsv")
    
    # Paths (on External Drive now)
    base_ext = "./data/embeddings"
    ap.add_argument("--train_terms", default="./data/raw/train/train_terms.tsv")
    ap.add_argument("--train_ids_npy", default=f"{base_ext}/esm2_15B/train_sequences_ids.npy")
    ap.add_argument("--train_taxon_npy", default=f"{base_ext}/taxonomy/train_species_idx.npy")
    
    ap.add_argument("--test_ids_npy", default=f"{base_ext}/testsuperset_ids.npy")
    ap.add_argument("--test_taxon_npy", default=f"{base_ext}/taxonomy/test_species_idx.npy")
    
    ap.add_argument("--penalty", type=float, default=0.5, help="Multiplier for invalid terms (0.5 means 50% penalty)")
    ap.add_argument("--restore_threshold", type=float, default=0.9, help="Keep score if > this value even if invalid")
    
    args = ap.parse_args()
    
    print(f"?? Taxonomy Filter (Fast) Starting...")
    print(f"   Input: {args.pred_in}")
    print(f"   Output: {args.pred_out}")
    print(f"   Penalty: {args.penalty}, Restore: {args.restore_threshold}")

    # 1. Load Species Mappings
    # Train: ID -> Taxon
    train_id2sp = load_mapping(args.train_ids_npy, args.train_taxon_npy)
    # Test: ID -> Taxon
    test_id2sp = load_mapping(args.test_ids_npy, args.test_taxon_npy)
    
    # 2. Build Valid Terms per Species (from Train)
    print("   ?뱴 Building Valid Terms per Species...")
    sp2terms = {}
    df = pd.read_csv(args.train_terms, sep="\t", usecols=["EntryID", "term"])
    
    # Iterate DataFrame efficiently
    # Pre-fetch ID to Taxon dict for speed
    # Convert DF to list of tuples
    data = zip(df["EntryID"], df["term"])
    
    unknown_ids = 0
    for pid, term in tqdm(data, total=len(df), desc="Mapping Terms"):
        pid = str(pid).strip()
        if pid in train_id2sp:
            sp = train_id2sp[pid]
            if sp not in sp2terms: sp2terms[sp] = set()
            sp2terms[sp].add(term)
        else:
            unknown_ids += 1
            
    print(f"   ??Mapped {len(sp2terms)} species. Included {len(df)} annotations. (Skipped {unknown_ids} unknown IDs)")

    # 3. Filter Predictions
    # Use buffered read/write for speed
    print("   ?㏏ Filtering Predictions...")
    
    # Pre-calculate penalty needed? No, simple float mul.
    
    count_processed = 0
    count_penalized = 0
    
    with open(args.pred_in, "r") as fin, open(args.pred_out, "w") as fout:
        # Check header? Assuming NO header in CAFA file usually.
        # But if first line is textual, skip.
        
        # Buffer processing? Line by line is safe for memory.
        for line in tqdm(fin, desc="Filtering"):
            parts = line.strip().split("\t")
            if len(parts) < 3: continue
            
            # Check if header
            if parts[0].startswith("EntryID") or parts[0].lower().startswith("id"):
                continue # Skip header if present
                
            pid = parts[0]
            term = parts[1]
            try:
                score = float(parts[2])
            except:
                continue # Skip bad lines
                
            if score <= 0: continue
            
            # Check Taxonomy
            # Pid -> Species
            # CAUTION: Test IDs in file might need cleaning?
            # Usually strict match.
            
            # fast lookup
            clean_pid = pid
            
            # Apply Logic
            final_score = score
            
            if clean_pid in test_id2sp:
                sp = test_id2sp[clean_pid]
                
                # If we know valid terms for this species
                if sp in sp2terms:
                    valid_set = sp2terms[sp]
                    if term not in valid_set:
                        # Invalid!
                        # Apply penalty unless very high confidence
                        if score < args.restore_threshold:
                            final_score = score * args.penalty
                            count_penalized += 1
            
            # Write
            if final_score > 0.001:
                fout.write(f"{pid}\t{term}\t{final_score:.5f}\n")
            
            count_processed += 1

    print(f"\n??Done. Processed {count_processed:,} lines.")
    print(f"   ?뱣 Penalized {count_penalized:,} invalid terms.")
    print(f"   ?뱚 Saved to: {args.pred_out}")

if __name__ == "__main__":
    main()


