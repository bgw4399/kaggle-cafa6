import sys
import os
from tqdm import tqdm

# Files
FILE_ESM = "./results/final_submission/final_esm_full.tsv"
FILE_PROT = "./results/final_submission/final_prott5_full.tsv"
OUTPUT_FILE = "./results/final_submission/submission_Final_Scientific_Ensemble.tsv"

# Weights
W_ESM = 0.6
W_PROT = 0.4

def parse_line(line):
    if not line: return None, None, 0.0
    p = line.strip().split('\t')
    if len(p) < 3: return None, None, 0.0
    return p[0], p[1], float(p[2])

def read_protein_block(f, pushback_line):
    """
    Reads all lines for the next protein.
    Returns: (pid, {term: score}, next_pushback_line)
    """
    scores = {}
    current_pid = None
    
    # Process pushback first
    line = pushback_line
    if not line:
        line = f.readline()
        
    while line:
        pid, term, score = parse_line(line)
        if not pid: # Empty line or bad format
            line = f.readline()
            continue
            
        if current_pid is None:
            current_pid = pid
            
        if pid != current_pid:
            # New protein found, return current block and push this line back
            return current_pid, scores, line
        
        # Add to current block
        if term:
            scores[term] = score
            
        line = f.readline()
        
    # EOF
    return current_pid, scores, None

def main():
    print("🚀 Generating Final Ensemble (Low-RAM Stream)...")
    
    f_esm = open(FILE_ESM, 'r')
    f_prot = open(FILE_PROT, 'r')
    f_out = open(OUTPUT_FILE, 'w')
    
    line_esm = None
    line_prot = None
    
    count = 0
    pbar = tqdm(desc="Merging Proteins", unit="prot")
    
    # Priming
    pid_esm, block_esm, line_esm = read_protein_block(f_esm, line_esm)
    pid_prot, block_prot, line_prot = read_protein_block(f_prot, line_prot)
    
    while pid_esm or pid_prot:
        # Determine strict processing order (to handle potential misalignments)
        # If both present
        target_pid = None
        
        if pid_esm and pid_prot:
            if pid_esm == pid_prot:
                target_pid = pid_esm
                # Combine
                terms = set(block_esm.keys()).union(block_prot.keys())
                final_scores = []
                for t in terms:
                    s1 = block_esm.get(t, 0.0)
                    s2 = block_prot.get(t, 0.0)
                    fs = (s1 * W_ESM) + (s2 * W_PROT)
                    if fs > 0.001:
                        final_scores.append((t, fs))
                
                # Sort descending
                final_scores.sort(key=lambda x: x[1], reverse=True)
                
                # Write
                for t, s in final_scores:
                    f_out.write(f"{target_pid}\t{t}\t{s:.5f}\n")
                
                # Advance both
                pid_esm, block_esm, line_esm = read_protein_block(f_esm, line_esm)
                pid_prot, block_prot, line_prot = read_protein_block(f_prot, line_prot)
                
            elif pid_esm < pid_prot:
                # ESM behind (or Prot missing this ID)
                target_pid = pid_esm
                # Process ESM only
                final_scores = []
                for t, s in block_esm.items():
                    fs = s * W_ESM
                    if fs > 0.001: final_scores.append((t, fs))
                final_scores.sort(key=lambda x: x[1], reverse=True)
                for t, s in final_scores: f_out.write(f"{target_pid}\t{t}\t{s:.5f}\n")
                
                # Advance ESM
                pid_esm, block_esm, line_esm = read_protein_block(f_esm, line_esm)
                
            else: # pid_esm > pid_prot
                # Prot behind
                target_pid = pid_prot
                # Process Prot only
                final_scores = []
                for t, s in block_prot.items():
                    fs = s * W_PROT
                    if fs > 0.001: final_scores.append((t, fs))
                final_scores.sort(key=lambda x: x[1], reverse=True)
                for t, s in final_scores: f_out.write(f"{target_pid}\t{t}\t{s:.5f}\n")
                
                # Advance Prot
                pid_prot, block_prot, line_prot = read_protein_block(f_prot, line_prot)
        
        elif pid_esm:
             # Prot finished
             target_pid = pid_esm
             final_scores = []
             for t, s in block_esm.items():
                 fs = s * W_ESM
                 if fs > 0.001: final_scores.append((t, fs))
             final_scores.sort(key=lambda x: x[1], reverse=True)
             for t, s in final_scores: f_out.write(f"{target_pid}\t{t}\t{s:.5f}\n")
             pid_esm, block_esm, line_esm = read_protein_block(f_esm, line_esm)
             
        elif pid_prot:
             # ESM finished
             target_pid = pid_prot
             final_scores = []
             for t, s in block_prot.items():
                 fs = s * W_PROT
                 if fs > 0.001: final_scores.append((t, fs))
             final_scores.sort(key=lambda x: x[1], reverse=True)
             for t, s in final_scores: f_out.write(f"{target_pid}\t{t}\t{s:.5f}\n")
             pid_prot, block_prot, line_prot = read_protein_block(f_prot, line_prot)

        count += 1
        if count % 1000 == 0:
            pbar.update(1000)
            
    pbar.close()
    f_esm.close()
    f_prot.close()
    f_out.close()
    print(f"✅ Low-RAM Merge Complete: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
