import pandas as pd
import numpy as np
import sys
from collections import defaultdict
from tqdm import tqdm
import os

# Files
INPUT_FILE = "./results/submission_ESM15B_Ensemble_Weighted_Repaired.tsv"
GAF_FILE = "./gaf_positive_preds.tsv"
OUTPUT_FILE = "./results/submission_ESM15B_Ensemble_Weighted_Final.tsv"
OBO_FILE = "./data/raw/train/go-basic.obo"

def load_obo_parents(path):
    print(f"Loading OBO from {path}...")
    parents = defaultdict(set)
    term = None
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line == '[Term]':
                term = None
            elif line.startswith('id: '):
                term = line[4:].split()[0]
            elif line.startswith('is_a: ') and term:
                parent = line[6:].split()[0]
                parents[term].add(parent)
            elif line.startswith('relationship: part_of ') and term:
                parent = line[22:].split()[0]
                parents[term].add(parent)
    return parents

def main():
    print("?? Starting GAF Positive Injection & Final Propagation...")
    
    # 1. Load Repaired File
    if not os.path.exists(INPUT_FILE):
        print(f"?슚 Input file not found: {INPUT_FILE}")
        return

    print("   ?뱰 Reading Repaired Submission...")
    # Load as simple dataframe
    # This might be large, but we need random access for fast injection.
    try:
        df = pd.read_csv(INPUT_FILE, sep='\t', header=None, names=['id', 'term', 'score'], dtype={'score': 'float32', 'id': 'str', 'term': 'str'})
        print(f"     -> Loaded {len(df):,} rows.")
    except Exception as e:
        print(f"     -> Error loading input: {e}")
        return

    # 2. Load GAF Positives
    print("   ?뱰 Reading GAF Positives...")
    gaf_pairs = set()
    if os.path.exists(GAF_FILE):
        with open(GAF_FILE, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    pid = parts[0].strip()
                    term = parts[1].strip()
                    gaf_pairs.add((pid, term))
        print(f"     -> Loaded {len(gaf_pairs):,} GAF positive pairs.")
    else:
        print("     -> ?좑툘 GAF File not found. Skipping injection.")
        gaf_pairs = set()

    if not gaf_pairs:
        print("Nothing to inject. Exiting.")
        return

    # 3. Inject and Propagate
    # We need to process by protein to propagate efficiently.
    parents_map = load_obo_parents(OBO_FILE)
    
    print("   ??processing per protein (Vectorized)...")
    
    unique_ids = df['id'].unique()
    
    # Identify proteins that need update
    # Filter GAF pairs to only those proteins present in the submission? 
    # OR adds new rows? GAF positives might be new terms!
    # If GAF positive is for a protein in test set, we must include it.
    
    # Let's map GAF pairs by ID
    gaf_by_id = defaultdict(set)
    for pid, term in gaf_pairs:
        gaf_by_id[pid].add(term)
        
    final_dfs = []
    
    # We process in chunks of proteins
    # But wait, rewriting the whole file is expensive.
    # Are we just modifying scores?
    
    # Strategy:
    # 1. Iterate proteins in input df.
    # 2. If protein has GAF hits:
    #    a. Pivot to matrix.
    #    b. Set score=1.0 for GAF terms.
    #    c. Propagate 1.0 upwards.
    #    d. Save.
    # 3. If no GAF hits, just save as is (pass through).
    
    with open(OUTPUT_FILE, 'w') as f_out:
        batch_size = 2000
        for i in tqdm(range(0, len(unique_ids), batch_size)):
            batch_ids = unique_ids[i : i+batch_size]
            batch_set = set(batch_ids)
            
            # Check if any protein in this batch has GAF hits
            has_gaf = any(pid in gaf_by_id for pid in batch_ids)
            
            sub_df = df[df['id'].isin(batch_set)]
            
            if not has_gaf:
                # Fast path: Write directly
                sub_df.to_csv(f_out, sep='\t', header=False, index=False, float_format='%.5f')
                continue
                
            # Slow path: Pivot, Inject, Propagate
            pivoted = sub_df.pivot(index='id', columns='term', values='score').fillna(0)
            
            # Inject 1.0
            # Ensure columns exist
            # For each protein in batch, get its GAF terms
            # Add columns if missing
            
            # Collect all needed columns first
            needed_cols = set(pivoted.columns)
            for pid in batch_ids:
                if pid in gaf_by_id:
                    needed_cols.update(gaf_by_id[pid])
            
            # Add missing columns
            for c in needed_cols:
                if c not in pivoted.columns:
                    pivoted[c] = 0.0
            
            # Set values
            for pid in batch_ids:
                if pid in gaf_by_id:
                    for term in gaf_by_id[pid]:
                        if term in pivoted.columns: # Should be there now
                            pivoted.at[pid, term] = 1.0
                            
            # Propagate (Parent >= Child)
            # Only need to propagate if we injected 1.0?
            # Yes. 1.0 is the max, so it will push parents to 1.0.
            
            cols = list(pivoted.columns)
            col_map = {c: i for i, c in enumerate(cols)}
            vals = pivoted.values.copy()
            
            # Build relations
            relations = []
            for child in cols:
                if child in parents_map:
                    for p in parents_map[child]:
                        if p in col_map:
                            relations.append((col_map[child], col_map[p]))
            
            # Propagate up
            for _ in range(12): # 12 passes usually enough
                changed = False
                for c_idx, p_idx in relations:
                    np.maximum(vals[:, p_idx], vals[:, c_idx], out=vals[:, p_idx])
            
            # Output
            repaired_df = pd.DataFrame(vals, index=pivoted.index, columns=cols)
            melted = repaired_df.stack().reset_index()
            melted.columns = ['id', 'term', 'score']
            melted = melted[melted['score'] >= 0.01]
            melted.to_csv(f_out, sep='\t', header=False, index=False, float_format='%.5f')

    print(f"?럦 Final GAF Integrated File Saved: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()


