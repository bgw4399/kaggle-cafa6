import os
import sys
import subprocess
from tqdm import tqdm

# Config
FILE_BASE = "./results/final_submission/submission_Final_Repaired.tsv"
FILE_STACKING = "./results/final_submission/submission_Stacking_XGB.tsv"
OUTPUT_FILE = "./results/final_submission/submission_Hybrid_Rescue.tsv"

TEMP_BASE_SORTED = "temp_base_sorted.tsv"
TEMP_STACKING_SORTED = "temp_stacking_sorted.tsv"

def sort_file(input_file, output_file):
    print(f"🔄 Sorting {input_file} -> {output_file}...")
    # Use system sort. -k1,1 (ID) -k2,2 (Term)
    # Using disk-based merge sort
    cmd = f"sort -k1,1 -k2,2 --parallel=4 -S 2G {input_file} > {output_file}"
    ret = os.system(cmd)
    if ret != 0:
        raise RuntimeError(f"Sort failed for {input_file}")
    print(f"✅ Sorted {output_file}")

def read_block(f, current_line):
    # Reads a block of lines with the same ID
    if not current_line:
        line = f.readline()
        if not line: return None, [], None
    else:
        line = current_line
        
    parts = line.strip().split('\t')
    if len(parts) < 2: return None, [], None
    
    current_pid = parts[0]
    block = []
    
    while True:
        parts = line.strip().split('\t')
        if len(parts) < 2: break
        
        pid = parts[0]
        if pid != current_pid:
            return current_pid, block, line # Return next line for next iteration
            
        # Parse Term/Score
        term = parts[1]
        try:
            score = float(parts[2])
        except:
            score = 0.0
            
        block.append((term, score))
        
        line = f.readline()
        if not line:
            return current_pid, block, None
            
    return current_pid, block, None

def merge_sorted():
    print("🚀 Starting Streaming Merge (Low RAM)...")
    
    f_base = open(TEMP_BASE_SORTED, 'r')
    f_stack = open(TEMP_STACKING_SORTED, 'r')
    f_out = open(OUTPUT_FILE, 'w')
    
    # Read first lines
    line_base = f_base.readline()
    line_stack = f_stack.readline()
    
    # Primitives to hold current block
    pid_base, block_base, next_line_base = read_block(f_base, line_base)
    pid_stack, block_stack, next_line_stack = read_block(f_stack, line_stack)
    
    count_override = 0
    count_inject = 0
    count_total = 0
    
    # Merge Loop
    pbar = tqdm(desc="Merging")
    
    while pid_base or pid_stack:
        target_pid = None
        
        # Determine which PID is smaller (lexicographically)
        if pid_base and pid_stack:
            if pid_base == pid_stack:
                target_pid = pid_base
                use_base = True
                use_stack = True
            elif pid_base < pid_stack:
                target_pid = pid_base
                use_base = True
                use_stack = False
            else:
                target_pid = pid_stack
                use_base = False
                use_stack = True
        elif pid_base:
            target_pid = pid_base
            use_base = True
            use_stack = False
        else:
            target_pid = pid_stack
            use_base = False
            use_stack = True
            
        # Process Target PID
        base_terms = dict(block_base) if use_base else {}
        stack_terms = dict(block_stack) if use_stack else {}
        
        # 1. Add all Base Terms (Override if in Stack)
        all_terms = set(base_terms.keys()) | set(stack_terms.keys())
        
        for term in all_terms:
            final_score = 0.0
            
            # Priority: Stacking > Base
            if term in stack_terms:
                final_score = stack_terms[term]
                if term in base_terms:
                    count_override += 1
                else:
                    count_inject += 1
            elif term in base_terms:
                final_score = base_terms[term]
                
            f_out.write(f"{target_pid}\t{term}\t{final_score:.5f}\n")
            count_total += 1
            
        # Advance
        if use_base:
            pid_base, block_base, next_line_base = read_block(f_base, next_line_base)
        if use_stack:
            pid_stack, block_stack, next_line_stack = read_block(f_stack, next_line_stack)
            
        if count_total % 100000 == 0:
            pbar.update(100000)
            
    f_base.close()
    f_stack.close()
    f_out.close()
    
    print(f"\n✅ Merge Complete.")
    print(f"   Total Predictions: {count_total:,}")
    print(f"   Overridden: {count_override:,}")
    print(f"   Injected: {count_inject:,}")
    
    # Cleanup
    os.remove(TEMP_BASE_SORTED)
    os.remove(TEMP_STACKING_SORTED)

def main():
    # 1. Sort files
    if not os.path.exists(TEMP_STACKING_SORTED):
        # We assume clean stacking file exists
        sort_file(FILE_STACKING, TEMP_STACKING_SORTED)
        
    if not os.path.exists(TEMP_BASE_SORTED):
        sort_file(FILE_BASE, TEMP_BASE_SORTED)
        
    # 2. Merge
    merge_sorted()

if __name__ == "__main__":
    main()
