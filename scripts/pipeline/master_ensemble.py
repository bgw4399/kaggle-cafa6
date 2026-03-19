import sys
import os
from tqdm import tqdm

# Config
FILE_OLD = "./results/submission_378_Pure_Repair.tsv"
FILE_NEW = "./results/final_submission/submission_final_filtered.tsv"
OUTPUT_FILE = "./results/final_submission/submission_Master_Ensemble.tsv"

WEIGHT_OLD = 0.7
WEIGHT_NEW = 0.3

def parse_line(line):
    if not line: return None, None, 0.0
    p = line.strip().split('\t')
    if len(p) < 3: return None, None, 0.0
    # Skip header
    if p[0].startswith("Entry") or p[0].lower().startswith("id"): return None, None, 0.0
    
    try:
        return p[0], p[1], float(p[2])
    except:
        return None, None, 0.0

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
    print(f"?? Master Ensemble (The Merger)...")
    print(f"   Old Best (0.378): {FILE_OLD} (w={WEIGHT_OLD})")
    print(f"   New Scientific (0.32): {FILE_NEW} (w={WEIGHT_NEW})")
    
    try:
        f_old = open(FILE_OLD, 'r')
        f_new = open(FILE_NEW, 'r')
        f_out = open(OUTPUT_FILE, 'w')
    except Exception as e:
        print(f"??Error opening files: {e}")
        return
    
    line_old = None
    line_new = None
    
    # Init
    pid_old, block_old, line_old = read_protein_block(f_old, line_old)
    pid_new, block_new, line_new = read_protein_block(f_new, line_new)
    
    count = 0
    pbar = tqdm(desc="Merging")
    
    while pid_old or pid_new:
        target_pid = None
        final_scores = []
        
        # Merge Logic (Sorted IDs assumed? CAFA files usually sorted by ID, but distinct models might differ)
        # If files are NOT sorted identically, this streaming logic FAILS.
        # But commonly, evaluation files are grouped by protein.
        # Let's hope they are at least grouped.
        # If order differs: "pid_old < pid_new" implies string comparison.
        
        # Standard String Sort Handling
        # If we encounter a mismatch, we process the "smaller" ID first.
        
        use_old = False
        use_new = False
        
        if pid_old and pid_new:
            if pid_old == pid_new:
                target_pid = pid_old
                use_old = True; use_new = True
            elif pid_old < pid_new:
                target_pid = pid_old
                use_old = True
            else:
                target_pid = pid_new
                use_new = True
        elif pid_old:
            target_pid = pid_old
            use_old = True
        elif pid_new:
            target_pid = pid_new
            use_new = True
            
        # Collect Terms
        terms = set()
        if use_old: terms.update(block_old.keys())
        if use_new: terms.update(block_new.keys())
        
        for t in terms:
            s1 = block_old.get(t, 0.0) if use_old else 0.0
            s2 = block_new.get(t, 0.0) if use_new else 0.0
            
            fs = (s1 * WEIGHT_OLD) + (s2 * WEIGHT_NEW)
            
            if fs > 0.001: final_scores.append((t, fs))
            
        # Write
        # Sort by score desc for neatness (optional but good)
        final_scores.sort(key=lambda x: x[1], reverse=True)
        
        for t, s in final_scores:
            f_out.write(f"{target_pid}\t{t}\t{s:.5f}\n")
            
        # Advance
        if use_old: pid_old, block_old, line_old = read_protein_block(f_old, line_old)
        if use_new: pid_new, block_new, line_new = read_protein_block(f_new, line_new)
        
        count += 1
        if count % 1000 == 0: pbar.update(1000)
            
    pbar.close()
    f_old.close()
    f_new.close()
    f_out.close()
    print(f"??Master Merge Complete: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

