import sys
import os
import csv
from tqdm import tqdm

# ==========================================
# [Disk/RAM Safe Stream Merge]
# ==========================================
FILE_OLD = "./results/submission_378.tsv"
WEIGHT_OLD = 0.7

FILE_NEW = "./results/final_submission/submission_final_filtered.tsv"
WEIGHT_NEW = 0.3

OUTPUT_FILE = "./results/final_submission/submission_Master_Ensemble.tsv"

# Temporary sorted files
TEMP_OLD = "temp_sorted_old.tsv"
TEMP_NEW = "temp_sorted_new.tsv"

def sort_file(input_file, output_file):
    print(f"?봽 Sorting {input_file} -> {output_file}...")
    if not os.path.exists(input_file):
        print(f"??Input file missing: {input_file}")
        return False
        
    # Linux sort command: Sort by Column 1 (ID), then Column 2 (Term)
    # -k1,1: Key 1 start to end
    # -k2,2: Key 2 start to end
    # -T .: Use current directory for temp (ensures output partition is used)
    # -S 1G: Use 1GB buffer for sort (adjustable)
    cmd = f"sort -k1,1 -k2,2 -S 1G -T . \"{input_file}\" > \"{output_file}\""
    
    ret = os.system(cmd)
    if ret != 0:
        print("??Sort command failed. Do you have 'sort' installed and enough disk space?")
        return False
    return True

def parse_line(line):
    # Returns id, term, score
    parts = line.strip().split('\t')
    if len(parts) < 3: return None, None, 0.0
    
    try:
        s = float(parts[2])
        if s > 1.0: s = 1.0 # Clip
    except:
        s = 0.0
    return parts[0], parts[1], s

def stream_merge():
    print("?? Starting Stream Merge...")
    
    f1 = open(TEMP_OLD, 'r')
    f2 = open(TEMP_NEW, 'r')
    f_out = open(OUTPUT_FILE, 'w')
    
    line1 = f1.readline()
    line2 = f2.readline()
    
    id1, term1, score1 = parse_line(line1) if line1 else (None, None, 0.0)
    id2, term2, score2 = parse_line(line2) if line2 else (None, None, 0.0)
    
    count = 0
    pbar = tqdm(desc="Merging Lines", unit="lines")
    
    while id1 or id2:
        # Determine strict order: ID then Term
        use_1 = False
        use_2 = False
        
        # Comparison Key construction
        k1 = (id1, term1) if id1 else ("~", "~") # "~" is larger than any ID
        k2 = (id2, term2) if id2 else ("~", "~")
        
        if k1 == k2:
            # Match! Merge.
            use_1 = True
            use_2 = True
            final_score = (score1 * WEIGHT_OLD) + (score2 * WEIGHT_NEW)
        elif k1 < k2:
            # Item 1 is earlier
            use_1 = True
            final_score = score1 * WEIGHT_OLD # Only in old
        else:
            # Item 2 is earlier
            use_2 = True
            final_score = score2 * WEIGHT_NEW # Only in new
            
        # Write if significant
        if final_score > 0.001:
            target_id = id1 if use_1 else id2
            target_term = term1 if use_1 else term2
            f_out.write(f"{target_id}\t{target_term}\t{final_score:.5f}\n")
            count += 1
            
        # Advance pointers
        if use_1:
            line1 = f1.readline()
            id1, term1, score1 = parse_line(line1) if line1 else (None, None, 0.0)
            
        if use_2:
            line2 = f2.readline()
            id2, term2, score2 = parse_line(line2) if line2 else (None, None, 0.0)
            
        if count % 100000 == 0: pbar.update(100000)
            
    pbar.close()
    f1.close()
    f2.close()
    f_out.close()
    print(f"??Final Output: {OUTPUT_FILE} ({count:,} rows)")

def main():
    try:
        # 1. Sort inputs
        if not sort_file(FILE_OLD, TEMP_OLD): return
        if not sort_file(FILE_NEW, TEMP_NEW): return
        
        # 2. Merge
        stream_merge()
        
    except KeyboardInterrupt:
        print("\n?썞 Interrupted.")
    finally:
        # Cleanup
        if os.path.exists(TEMP_OLD): os.remove(TEMP_OLD)
        if os.path.exists(TEMP_NEW): os.remove(TEMP_NEW)
        print("?㏏ Cleanup temp files.")

if __name__ == "__main__":
    main()

