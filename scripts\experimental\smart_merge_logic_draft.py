import pandas as pd
import numpy as np
import sys
from collections import defaultdict

# Config
FILE_378 = "./results/submission_378.tsv"
FILE_371 = "./results/submission_SOTA_Kingdom_Max_371.tsv"
OBO_FILE = "./data/raw/train/go-basic.obo"
OUTPUT_FILE = "./results/submission_ensemble_repaired.tsv"

def load_obo(path):
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
    
    print(f"Loaded {len(parents)} terms with parents.")
    return parents

def propagate_scores(df, parents_map):
    """
    Ensure Parent Score >= Child Score.
    This works by iterating. Since max depth is small (<20 usually), restricted iteration works.
    Better: topologically sorted, but simple iteration is easier to implement for quick script.
    """
    print("Reparing ontology structure (Propagation)...")
    
    # Convert to dict for fast access: proteins[id][term] = score
    # This might be memory intensive.
    # Alternative: Group by ID.
    
    # Let's process protein by protein (or chunks) to save memory
    # But for a script, let's try pandas groupby apply? Slow.
    
    # Memory Efficient Approach:
    # 1. Get all involved terms.
    # 2. Build local graph of terms in the file.
    # 3. Sort terms by depth (or simplistically, just iterate 3-4 times which covers most depth).
    
    # ACTUALLY, simpler:
    # Most violations are local.
    # We can iterate: score[parent] = max(score[parent], score[child])
    # Repeat this pass 15 times (max depth of GO is ~16-20).
    
    terms = df['term'].unique()
    # Pre-calculate parent-child list for terms present in DF
    # List of (child, parent) tuples
    
    relevant_relations = []
    # Filter parent map to only terms in df
    # optimization: set of terms
    term_set = set(terms)
    
    for child, parents in parents_map.items():
        if child in term_set:
            for p in parents:
                if p in term_set:
                    relevant_relations.append((child, p))
    
    print(f"Found {len(relevant_relations)} relevant parent-child relations.")
    
    # To vectorize, we can't easily.
    # But we can iterate over relations.
    # For each relation (child, parent):
    #   df.loc[parent] = max(df.loc[parent], df.loc[child]) -> Hard in long format.
    
    # Pivot is too big (50k proteins * 10k terms = 500M cells).
    
    # Strategy C:
    # Sort relations by 'level' if possible?
    # Or just use the raw dataframe and merge itself?
    
    # 1. Rename columns to parent/child
    # 2. Merge on ID.
    
    for i in range(12): # 12 passes should propagate deep enough
        print(f"  Pass {i+1}...")
        changes = 0
        
        # We need to propagate: For every (child, parent) pair,
        # Score(Parent) = Max(Score(Parent), Score(Child))
        
        # This is hard to do efficiently in Pandas Long format without blowing up memory.
        # But maybe we don't need to FULLY repair 371?
        # Just repairing the main root terms and their immediate children might be enough?
        # No, user wants good results.
        
        # Let's use a dictionary approach on a per-protein basis.
        # It's Python loop but if we group by Protein ID it's okay.
        pass
        
    return df

def smart_merge():
    # 1. Load Ontology
    parents = load_obo(OBO_FILE)
    
    # 2. Load Data (Optimized)
    # We will process protein-by-protein to allow sophisticated repair without RAM explosion
    
    print("Reading files in chunks...")
    # Read both files, sort by ID to stream
    # Takes effort.
    
    # Alternative: Just load all if < 64GB RAM.
    # 20M rows * 3 cols * 8 bytes ~ 480MB. Python overhead -> 5GB. Fine.
    
    print("Loading 378 (Base)...")
    df1 = pd.read_csv(FILE_378, sep='\t', header=None, names=['id', 'term', 'score'], 
                      dtype={'score': 'float32', 'id': 'str', 'term': 'str'})
    
    print("Loading 371 (To Repair)...")
    df2 = pd.read_csv(FILE_371, sep='\t', header=None, names=['id', 'term', 'score'], 
                      dtype={'score': 'float32', 'id': 'str', 'term': 'str'})
    
    # Repair 371 logic
    # Group df2 by ID
    print("Repairing 371...")
    
    # Convert to list of dicts?
    # Or operate on pivot?
    # 50000 proteins. Pivot is feasible if sparse?
    # Pandas SparseDtype.
    
    # Let's try a simpler heuristic repair:
    # Just fix the ROOT terms and Level 1 terms which are the massive blockers.
    # Or, trust that 378 has structure.
    
    # HYBRID STRATEGY:
    # Final_Score = (Score_378 + Score_371) / 2
    # IF Score_378 > Score_371: Trust 378 (it has structure) -> Average is safe (pulls down 371 error).
    # IF Score_371 >> Score_378 (e.g. 0.9 vs 0.0): This is the "Structure Violation" case usually. 
    #   (Child=0.9, Parent in 378=0.9, Parent in 371=0.0).
    #   If we average, we get Child=0.9, Parent=0.45. VIOLATION!
    
    # So we MUST repair 371 BEFORE averaging.
    
    # Let's implement a simplified repair in-memory:
    # 1. Pivot `df2` (sparse).
    # 2. Iterate relations.
    pass

if __name__ == "__main__":
    # This is a placeholder for the script I will provide to the user.
    # The actual full script is complex.
    pass


