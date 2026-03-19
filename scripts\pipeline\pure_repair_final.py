import sys
import os
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

# Config
FILE_INPUT = "./temp_local_ensemble.tsv"
FILE_OUTPUT = "./results/final_submission/submission_Final_Repaired.tsv"
OBO_FILE = "./data/raw/train/go-basic.obo"
GAF_POS_FILE = "gaf_positive_preds.tsv"
GAF_NEG_FILE = "gaf_negative_preds.tsv"

def load_obo(path):
    print("?뱴 Loading OBO...")
    parents_map = defaultdict(set)
    with open(path, 'r') as f:
        term = None
        for line in f:
            line = line.strip()
            if line.startswith('id: GO:'):
                term = line[4:].split()[0]
            elif line.startswith('is_a: ') and term:
                p = line[6:].split(' ! ')[0]
                parents_map[term].add(p)
                
    # Optimization: Transitive closure or just direct?
    # For "Parent >= Child", checking immediate parents repeatedly until convergence is enough.
    # But pre-calculating ancestors map might optionally speed up, but takes RAM.
    # Direct parents map is small. We'll use loop.
    return parents_map

def load_gaf_filters():
    print("   ?㏏ Loading GAF Filters...")
    gaf_neg = set()
    if os.path.exists(GAF_NEG_FILE):
        with open(GAF_NEG_FILE) as f:
            for line in f:
                p = line.strip().split('\t')
                if len(p)>=2: gaf_neg.add((p[0], p[1]))
                
    gaf_pos = defaultdict(set)
    if os.path.exists(GAF_POS_FILE):
        with open(GAF_POS_FILE) as f:
            for line in f:
                p = line.strip().split('\t')
                if len(p)>=2: gaf_pos[p[0]].add(p[1])
                
    return gaf_neg, gaf_pos

def parse_line(line):
    if not line: return None, None, 0.0
    p = line.strip().split('\t')
    if len(p) < 3: return None, None, 0.0
    return p[0], p[1], float(p[2])

def read_protein_block(f, pushback_line):
    scores = {}
    current_pid = None
    line = pushback_line
    if not line: line = f.readline()
        
    while line:
        pid, term, score = parse_line(line)
        if not pid: 
            line = f.readline()
            continue
        if current_pid is None: current_pid = pid
        if pid != current_pid: return current_pid, scores, line
        
        if term: scores[term] = score
        line = f.readline()
        
    return current_pid, scores, None

def main():
    print(f"?? Streaming Repair & Inject...")
    parents_map = load_obo(OBO_FILE)
    neg_set, pos_map = load_gaf_filters()
    
    f_in = open(FILE_INPUT, 'r')
    f_out = open(FILE_OUTPUT, 'w')
    
    line_buffer = None
    count = 0
    pbar = tqdm(desc="Processing Proteins")
    
    while True:
        pid, preds, line_buffer = read_protein_block(f_in, line_buffer)
        if not pid: break
        
        # 1. GAF Negative Filter
        # Remove terms in neg_set
        keys = list(preds.keys())
        for t in keys:
            if (pid, t) in neg_set:
                del preds[t]
        
        # 2. GAF Positive Injection
        # Force 1.0
        if pid in pos_map:
            for t in pos_map[pid]:
                preds[t] = 1.0
                
        # 3. Propagate (Quick Repair)
        # Ensure Parent >= Child
        # Iterate until stable (max 10 loops)
        changed = True
        loop = 0
        while changed and loop < 10:
            changed = False
            curr_terms = list(preds.keys())
            for t in curr_terms:
                score = preds[t]
                if t in parents_map:
                    for p in parents_map[t]:
                        p_score = preds.get(p, 0.0)
                        if score > p_score:
                            preds[p] = score
                            changed = True
            loop += 1
            
        # 4. Write
        # Sort desc
        final_items = sorted(preds.items(), key=lambda x: x[1], reverse=True)
        for t, s in final_items:
            if s > 0.001:
                f_out.write(f"{pid}\t{t}\t{s:.5f}\n")
                
        count += 1
        if count % 1000 == 0: pbar.update(1000)
        
    pbar.close()
    f_in.close()
    f_out.close()
    print(f"??Repair Complete: {FILE_OUTPUT}")

if __name__ == "__main__":
    main()

