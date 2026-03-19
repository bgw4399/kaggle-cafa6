import sys
import os
from tqdm import tqdm

INPUT_FILE = "./results/temp_ensemble_sorted.tsv"
OUTPUT_FILE = "./results/submission_SOTA_Ensemble_Final.tsv"

def aggregate():
    print(f"?? Aggregating Sorted Stream: {INPUT_FILE} -> {OUTPUT_FILE}")
    
    if not os.path.exists(INPUT_FILE):
        print(f"??Input file not found: {INPUT_FILE}")
        return

    current_key = None
    current_score = 0.0
    
    # We use manual buffering for speed
    with open(INPUT_FILE, 'r') as f_in, open(OUTPUT_FILE, 'w') as f_out:
        # Using tqdm is tricky with line-by-line file size unknown, but we can guess
        # 3.7GB -> approx 75M lines?
        pbar = tqdm(total=75_000_000, desc="Aggregating", unit="line")
        
        for line in f_in:
            parts = line.split('\t') # fast split
            if len(parts) < 3: continue
            
            pid = parts[0]
            term = parts[1]
            try:
                score = float(parts[2])
            except:
                continue
                
            key = (pid, term)
            
            if key == current_key:
                current_score += score
            else:
                # Write previous
                if current_key is not None:
                    if current_score > 0.001:
                        f_out.write(f"{current_key[0]}\t{current_key[1]}\t{current_score:.5f}\n")
                
                # Reset
                current_key = key
                current_score = score
            
            pbar.update(1)
            
        # Write last
        if current_key is not None:
             if current_score > 0.001:
                f_out.write(f"{current_key[0]}\t{current_key[1]}\t{current_score:.5f}\n")
        
        pbar.close()

    print(f"?럦 Aggregation Complete!")

if __name__ == "__main__":
    aggregate()

