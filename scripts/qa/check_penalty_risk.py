import pandas as pd
import numpy as np
import sys

# Compare two files line by line (assuming same order or sortable)
# But files are HUGE (1.6GB).
# We'll check the "Penalty" logic simulation on SOTA file directly.

def check_sota_penalty():
    print("?쉻 Checking if SOTA hits would be killed by Taxon Filter...")
    
    # Load required maps
    train_terms_path = "./data/raw/train/train_terms.tsv"
    train_ids_npy = "data/embeddings/esm2_15B/train_sequences_ids.npy"
    train_taxon_npy = "data/embeddings/taxonomy/train_species_idx.npy"
    test_ids_npy = "data/embeddings/testsuperset_ids.npy"
    test_taxon_npy = "data/embeddings/taxonomy/test_species_idx.npy"
    
    # SOTA file
    sota_path = "./results/submission_SOTA_Kingdom_Max_371.tsv"

    print("   Loading Taxonomies...", flush=True)
    try:
        # Train Species -> Terms
        train_species = np.load(train_taxon_npy).reshape(-1)
        raw_train_ids = np.load(train_ids_npy)
        train_ids = [str(x).replace(">", "").strip() for x in raw_train_ids.reshape(-1)]
        id2sp = dict(zip(train_ids, train_species))
        
        sp2terms = {}
        df_train = pd.read_csv(train_terms_path, sep="\t")
        for pid, term in zip(df_train["EntryID"], df_train["term"]):
            pid = str(pid).strip()
            if pid in id2sp:
                sp = id2sp[pid]
                if sp not in sp2terms: sp2terms[sp] = set()
                sp2terms[sp].add(str(term).strip())
                
        # Test Species
        test_species = np.load(test_taxon_npy).reshape(-1)
        raw_test_ids = np.load(test_ids_npy)
        test_ids = [str(x).replace(">", "").strip() for x in raw_test_ids.reshape(-1)]
        test_id2sp = dict(zip(test_ids, test_species))
        
    except Exception as e:
        print(f"??Failed to load maps: {e}")
        return

    print("   Scanning SOTA file for potential penalties...", flush=True)
    total = 0
    penalized = 0
    high_score_penalized = 0
    
    try:
        with open(sota_path, "r") as f:
            for i, line in enumerate(f):
                if i > 5000000: break # Sample first 5M
                parts = line.strip().split("\t")
                if len(parts) < 3: continue
                
                pid = parts[0].replace(">", "")
                term = parts[1]
                score = float(parts[2])
                
                if pid not in test_id2sp: continue
                sp = test_id2sp[pid]
                
                if i < 5:
                    print(f"DEBUG: PID: {pid} | SP: {sp} | In Terms Map: {sp in sp2terms}")
                    if sp in sp2terms:
                         print(f"   Terms count for {sp}: {len(sp2terms[sp])}")
                    else:
                         print(f"   Shape of sp: {type(sp)}")

                
                # Logic: If species in train (known) AND term NOT in known set -> Penalty
                if sp in sp2terms:
                    if term not in sp2terms[sp]:
                        # This would be penalized
                        total += 1
                        penalized += 1
                        if score > 0.8: # High confidence hit
                            high_score_penalized += 1
                else:
                    # Species not in train, usually safe or handled differently
                    pass
    except Exception as e:
        print(e)
        
    print(f"?뵇 Result (Sample 5M lines):")
    print(f"   Total SOTA lines checking out-of-distribution: {total}")
    print(f"   Penalized (Would be reduced): {penalized} ({penalized/total*100:.1f}%)")
    print(f"   High Confidence (>0.8) Penalized: {high_score_penalized}")
    
    if high_score_penalized > 0:
        print("?좑툘 CRITICAL: The filter is attacking High Confidence SOTA predictions!")
    else:
        print("??High confidence SOTA predictions seem safe.")

if __name__ == "__main__":
    check_sota_penalty()


