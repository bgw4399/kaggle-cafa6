import argparse
import sys
import gc
import pandas as pd
import numpy as np
from scipy.stats import rankdata

def read_protein_block(f_handle):
    """
    Yields (pid, list_of_lines)
    Assumes file is sorted by pid.
    """
    current_pid = None
    buffer = []
    
    for line in f_handle:
        parts = line.strip().split("\t")
        if len(parts) < 3: continue
        
        pid = parts[0]
        # Skip header if present (though usually no header)
        if pid == "id" or pid.startswith("CAFA"): continue
        
        if current_pid is None:
            current_pid = pid
        
        if pid != current_pid:
            yield current_pid, buffer
            current_pid = pid
            buffer = []
        
        buffer.append(parts)
    
    if current_pid is not None and buffer:
        yield current_pid, buffer

def parse_block(block):
    # block is list of [pid, term, score]
    # return dict: term -> float(score)
    data = {}
    for p in block:
        try:
            data[p[1]] = float(p[2])
        except:
            pass
    return data

def get_ranks(scores_dict):
    # returns dict: term -> rank (0~1)
    # rankdata with method='average'
    terms = list(scores_dict.keys())
    scores = list(scores_dict.values())
    
    if not scores: return {}
    
    # rankdata returns 1..N
    # We want normalized 0..1? 
    # Usually: rank / len
    r = rankdata(scores, method='average')
    r = r / len(r)
    
    return dict(zip(terms, r))

def solve_ensemble(id_a, data_a, id_b, data_b, out_f):
    # data_a: dict term->score
    # score -> rank
    
    # Validation: IDs match?
    # The loop calls this only when aligned
    pid = id_a if id_a else id_b
    
    ranks_a = get_ranks(data_a) if data_a else {}
    ranks_b = get_ranks(data_b) if data_b else {}
    
    all_terms = set(ranks_a.keys()) | set(ranks_b.keys())
    
    results = []
    for t in all_terms:
        ra = ranks_a.get(t, 0.0) # If missing, rank 0
        rb = ranks_b.get(t, 0.0)
        
        # MEAN Rank
        avg_r = (ra + rb) / 2.0
        
        if avg_r > 0.001:
            results.append((t, avg_r))
            
    # Write
    # Optional: limit terms or threshold
    # But for rank ensemble, we just dump
    for t, r in results:
        out_f.write(f"{pid}\t{t}\t{r:.5f}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file1", required=True)
    parser.add_argument("--file2", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    
    print(f"🚀 Starting Rank Ensemble: {args.file1} + {args.file2}")
    
    with open(args.file1, "r") as f1, open(args.file2, "r") as f2, open(args.out, "w") as fout:
        
        gen1 = read_protein_block(f1)
        gen2 = read_protein_block(f2)
        
        try:
            curr1 = next(gen1) # (pid, block)
            pid1 = curr1[0]
        except StopIteration:
            curr1 = None
            pid1 = None
            
        try:
            curr2 = next(gen2)
            pid2 = curr2[0]
        except StopIteration:
            curr2 = None
            pid2 = None
            
        count = 0
        
        while curr1 is not None or curr2 is not None:
            # Determine which to process
            process_1 = False
            process_2 = False
            
            if curr1 is not None and curr2 is not None:
                if pid1 == pid2:
                    process_1 = True
                    process_2 = True
                elif pid1 < pid2:
                    process_1 = True
                else: # pid1 > pid2
                    process_2 = True
            elif curr1 is not None:
                process_1 = True
            else:
                process_2 = True
            
            # Extract data
            data1 = parse_block(curr1[1]) if process_1 else {}
            data2 = parse_block(curr2[1]) if process_2 else {}
            
            target_pid = pid1 if process_1 else pid2
            
            # Solve
            solve_ensemble(target_pid, data1, target_pid, data2, fout)
            count += 1
            if count % 10000 == 0:
                print(f"   Processed {count:,} proteins...", end="\r")
            
            # Advance generators
            if process_1:
                try:
                    curr1 = next(gen1)
                    pid1 = curr1[0]
                except StopIteration:
                    curr1 = None
                    pid1 = None
            
            if process_2:
                try:
                    curr2 = next(gen2)
                    pid2 = curr2[0]
                except StopIteration:
                    curr2 = None
                    pid2 = None
                    
    print(f"\n✅ Done! Total proteins: {count:,}")

if __name__ == "__main__":
    main()
