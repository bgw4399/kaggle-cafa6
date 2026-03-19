import pandas as pd
import collections
import os
from tqdm import tqdm

# Config
PRED_FILE = "./results/final_submission/submission_Final_Repaired.tsv" # Base file (0.31/0.378)
OUTPUT_FILE = "./results/final_submission/submission_Taxon_Corrected.tsv"
TRAIN_TAXON = "./data/raw/train/train_taxonomy.tsv"
TRAIN_TERMS = "./data/raw/train/train_terms.tsv"
DIAMOND_HITS = "train_stacking_hits.tsv" # Generated earlier

PENALTY_FACTOR = 0.1
RESTORE_THRESHOLD = 0.95 # Only very high confidence survives violation

def main():
    print("?뙼 Taxonomy-Aware Filtering (Homology Inference)...")
    
    # 1. Load Train Taxonomy
    print("   ?뱿 Loading Train Taxonomy...")
    train_id_to_species = {}
    try:
        df_tax = pd.read_csv(TRAIN_TAXON, sep='\t')
        for pid, tax in zip(df_tax['EntryID'], df_tax['TaxonomyID']):
            train_id_to_species[str(pid)] = int(tax)
    except:
        df_tax = pd.read_csv(TRAIN_TAXON, sep='\t', header=None, names=['EntryID', 'TaxonomyID'])
        for pid, tax in zip(df_tax['EntryID'], df_tax['TaxonomyID']):
            train_id_to_species[str(pid)] = int(tax)
            
    # 2. Build Allowed Terms per Species
    print("   ?뱴 Building KB (Species -> Terms)...")
    # Species -> Set(Terms)
    species_allowed = collections.defaultdict(set)
    df_terms = pd.read_csv(TRAIN_TERMS, sep='\t')
    for pid, term in zip(df_terms['EntryID'], df_terms['term']):
        pid = str(pid)
        if pid in train_id_to_species:
            sp = train_id_to_species[pid]
            species_allowed[sp].add(term)
            
    # 3. Infer Test Species from Diamond Hits
    print("   ?빑截?Inferring Test Species from Diamond Hits...")
    # TestID -> Suggested Species
    test_species_map = {}
    
    # We take the Top Hit for each query
    # File format: qseqid, sseqid, pident, ...
    # We generated this file in Stacking step.
    
    count_inferred = 0
    with open(DIAMOND_HITS, 'r') as f:
        for line in tqdm(f, desc="Parsing Hits"):
            parts = line.strip().split('\t')
            if len(parts) < 2: continue
            
            # format: sp|ID|... or just ID
            q_raw = parts[0]
            s_raw = parts[1]
            
            # Clean IDs
            def clean(x):
                if '|' in x: return x.split('|')[1]
                return x
            
            q_id = clean(q_raw)
            s_id = clean(s_raw)
            
            if q_id in test_species_map: continue # Provide first hit (best)
            
            if s_id in train_id_to_species:
                sp = train_id_to_species[s_id]
                test_species_map[q_id] = sp
                count_inferred += 1
                
    print(f"      Inferred Species for {count_inferred:,} test proteins.")
    
    # 4. Apply Filter
    print("   ?㏏ Applying Taxonomy Filter...")
    
    f_out = open(OUTPUT_FILE, 'w')
    count_penalized = 0
    total = 0
    
    # We stream the prediction file
    with open(PRED_FILE, 'r') as f:
        # Check header
        pos = f.tell()
        line = f.readline()
        if "Term" not in line and "term" not in line:
            f.seek(pos) # No header
            
        for line in tqdm(f, desc="Filtering"):
            parts = line.strip().split('\t')
            if len(parts) < 3:
                f_out.write(line)
                continue
                
            pid = parts[0]
            term = parts[1]
            try:
                score = float(parts[2])
            except:
                f_out.write(line)
                continue
            
            # Filter Logic
            keep_score = score
            
            if pid in test_species_map:
                sp = test_species_map[pid]
                if sp in species_allowed:
                    # Check if term is allowed
                    if term not in species_allowed[sp]:
                        # Violation!
                        # Check threshold
                        if score < RESTORE_THRESHOLD:
                            keep_score = score * PENALTY_FACTOR
                            count_penalized += 1
                            
            if keep_score > 0.001:
                f_out.write(f"{pid}\t{term}\t{keep_score:.5f}\n")
            total += 1
            
    f_out.close()
    print(f"??Filter Complete.")
    print(f"   Total Predictions: {total:,}")
    print(f"   Penalized Violations: {count_penalized:,}")
    print(f"?뱚 Output: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

