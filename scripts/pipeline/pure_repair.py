import sys
import os
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

FILE_INPUT = "./results/submission_378.tsv"
FILE_OUTPUT = "./results/submission_378_Pure_Repair.tsv"
OBO_FILE = "./data/raw/train/go-basic.obo"
GAF_NEG_FILE = "./gaf_negative_preds.tsv"
GAF_POS_FILE = "./gaf_positive_preds.tsv"

def load_obo(path):
    print("?뱴 Loading OBO...")
    parents = defaultdict(set)
    with open(path, 'r') as f:
        term = None
        for line in f:
            line = line.strip()
            if line.startswith('id: '): term = line[4:].split()[0]
            elif line.startswith('is_a: ') and term:
                parent = line[6:].split(' ! ')[0]
                parents[term].add(parent)
    return parents

def repair_and_filter():
    print("?? Running Pure Repair on Best_378...")
    
    # 1. Load OBO
    parents_map = load_obo(OBO_FILE)
    
    # 2. Load GAF Filters
    print("   ?㏏ Loading GAF Filters...")
    gaf_neg = set()
    if os.path.exists(GAF_NEG_FILE):
        with open(GAF_NEG_FILE) as f:
            for line in f:
                p = line.strip().split('\t')
                if len(p)>=2: gaf_neg.add((p[0], p[1]))
                
    gaf_pos = {} # (id, term) -> 1.0
    if os.path.exists(GAF_POS_FILE):
        with open(GAF_POS_FILE) as f:
            for line in f:
                p = line.strip().split('\t')
                if len(p)>=2: gaf_pos[(p[0], p[1])] = 1.0
                
    # 3. Process by Protein (Grouped) to Propagate
    # Streaming isn't enough for efficient prop, need chunking by ID
    # But 378 is sorted? If not, we use dict buffer.
    
    print("   ?뱿 Reading Data...")
    data = defaultdict(dict) # pid -> {term: score}
    
    with open(FILE_INPUT, 'r') as f:
        for line in tqdm(f, desc="Loading"):
            p = line.strip().split('\t')
            if len(p) < 3: continue
            pid, term, score = p[0], p[1], float(p[2])
            
            # GAF Negative Filter Immediate
            if (pid, term) in gaf_neg: continue
            
            data[pid][term] = score
            
    print(f"   ?㎚ Processing {len(data):,} proteins...")
    
    with open(FILE_OUTPUT, 'w') as f_out:
        for pid, preds in tqdm(data.items(), desc="Repairing"):
            
            # 1. Inject GAF Positives
            # (Note: Need to check if this protein has GAF pos. Optimizing lookup is hard without index, 
            # assuming GAF pos file is small or we pre-filtered. 
            # Actually, standard GAF injection: if pid in GAF pos list, set 1.0.
            # For simplicity, we skip GAF Pos injection here to be SAFE/PURE Repair first. 
            # Injecting 1.0 can cause propagation issues if not careful.)
            
            # 2. Propagate (Child <= Parent)
            # Maximize: Parent Score should be at least Child Score
            # We iterate multiple times or topological sort. 
            # Simple approach: Propagate upwards.
            
            # Get all terms involved
            active_terms = list(preds.keys())
            
            # Expansion (add parents if missing)? 
            # To be strict, CAFA rules say we should predict parents.
            # But adding too many parents increases file size 10x.
            # We only enforce consistency on EXISTING predictions + immediate parents if highly confident.
            
            # "Quick Repair": Just ensure if Parent exists, Score[Parent] = max(Score[Parent], Score[Child])
            # This is O(N*Depth).
            
            # Let's do a simplified constraint pass
            # Sort terms by depth? or simple loop
            changed = True
            loop = 0
            while changed and loop < 5:
                changed = False
                current_terms = list(preds.keys()) # snapshot
                for term in current_terms:
                    score = preds[term]
                    if term in parents_map:
                        for parent in parents_map[term]:
                            # If parent not in preds, should we add it?
                            # CAFA: Yes.
                            parent_score = preds.get(parent, 0.0)
                            if score > parent_score:
                                preds[parent] = score
                                changed = True
                loop += 1
                
            # Write
            for term, score in preds.items():
                if score > 0.001:
                    f_out.write(f"{pid}\t{term}\t{score:.5f}\n")
                    
    print(f"?럦 Pure Repair Complete: {FILE_OUTPUT}")

if __name__ == "__main__":
    repair_and_filter()


