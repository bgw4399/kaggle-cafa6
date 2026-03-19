import sys
import os
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

# Config
FILE_DL = "./results/final_submission/submission.tsv"
FILE_DIAMOND = "./temp_diamond.tsv"
OUTPUT_FILE = "./results/final_submission/submission_Final_Hybrid.tsv"
GAF_POS_FILE = "gaf_positive_preds.tsv"
GAF_NEG_FILE = "gaf_negative_preds.tsv"

WEIGHT_DL = 0.5
WEIGHT_DIAMOND = 0.5

def load_gaf_filters():
    print("   🧹 Loading GAF Filters...")
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
    print(f"🚀 Merging DL Ensemble + Diamond (Hybrid)...")
    print(f"   Weights: DL={WEIGHT_DL}, Diamond={WEIGHT_DIAMOND}")
    
    neg_set, pos_map = load_gaf_filters()
    
    f_dl = open(FILE_DL, 'r')
    f_diamond = open(FILE_DIAMOND, 'r')
    f_out = open(OUTPUT_FILE, 'w')
    
    line_dl = None
    line_diamond = None
    
    pbar = tqdm(desc="Merging")
    count = 0
    
    # Init
    pid_dl, block_dl, line_dl = read_protein_block(f_dl, line_dl)
    pid_dia, block_dia, line_diamond = read_protein_block(f_diamond, line_diamond)
    
    while pid_dl or pid_dia:
        target_pid = None
        final_scores = []
        
        # Determine strict processing order
        if pid_dl and pid_dia:
            if pid_dl == pid_dia:
                target_pid = pid_dl
                # Combine
                terms = set(block_dl.keys()).union(block_dia.keys())
                for t in terms:
                    s1 = block_dl.get(t, 0.0)
                    s2 = block_dia.get(t, 0.0)
                    fs = (s1 * WEIGHT_DL) + (s2 * WEIGHT_DIAMOND)
                    
                    # GAF Neg Filter
                    if (target_pid, t) in neg_set: continue
                    
                    if fs > 0.001: final_scores.append((t, fs))
                
                # Advance both
                pid_dl, block_dl, line_dl = read_protein_block(f_dl, line_dl)
                pid_dia, block_dia, line_diamond = read_protein_block(f_diamond, line_diamond)
                
            elif pid_dl < pid_dia:
                target_pid = pid_dl
                for t, s in block_dl.items():
                    fs = s * WEIGHT_DL
                    if (target_pid, t) in neg_set: continue
                    if fs > 0.001: final_scores.append((t, fs))
                pid_dl, block_dl, line_dl = read_protein_block(f_dl, line_dl)
                
            else: # pid_dl > pid_dia
                target_pid = pid_dia
                for t, s in block_dia.items():
                    fs = s * WEIGHT_DIAMOND
                    if (target_pid, t) in neg_set: continue
                    if fs > 0.001: final_scores.append((t, fs))
                pid_dia, block_dia, line_diamond = read_protein_block(f_diamond, line_diamond)
        
        elif pid_dl:
             target_pid = pid_dl
             for t, s in block_dl.items():
                 fs = s * WEIGHT_DL
                 if (target_pid, t) in neg_set: continue
                 if fs > 0.001: final_scores.append((t, fs))
             pid_dl, block_dl, line_dl = read_protein_block(f_dl, line_dl)
             
        elif pid_dia:
             target_pid = pid_dia
             for t, s in block_dia.items():
                 fs = s * WEIGHT_DIAMOND
                 if (target_pid, t) in neg_set: continue
                 if fs > 0.001: final_scores.append((t, fs))
             pid_dia, block_dia, line_diamond = read_protein_block(f_diamond, line_diamond)
             
        # GAF Positive Injection
        if target_pid in pos_map:
            # Create a dict for easy update
            s_dict = {t:s for t,s in final_scores}
            for t in pos_map[target_pid]:
                s_dict[t] = 1.0
            final_scores = list(s_dict.items())
            
        # Sort desc
        final_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Write
        for t, s in final_scores:
            f_out.write(f"{target_pid}\t{t}\t{s:.5f}\n")
            
        count += 1
        if count % 1000 == 0: pbar.update(1000)
    
    pbar.close()
    f_dl.close()
    f_diamond.close()
    f_out.close()
    print(f"✅ Hybrid Merge Complete: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
