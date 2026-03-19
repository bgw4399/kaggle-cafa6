import pandas as pd
import numpy as np
from tqdm import tqdm
import os

# Config
INPUT_FILE = "./results/submission_ESM15B_Ensemble_Weighted_Repaired.tsv"
OUTPUT_FILE = "./results/submission_ESM15B_TopK_Fixed.tsv"

# Top-K per protein (match Best_378's ~105 predictions per protein)
TOP_K = 120

# Minimum score threshold
MIN_SCORE = 0.01

def main():
    print(f"?? Top-K Filter: Reducing predictions per protein to {TOP_K}...")
    print(f"   Input: {INPUT_FILE}")
    
    if not os.path.exists(INPUT_FILE):
        print(f"??Input file not found!")
        return
    
    print("   ?뱰 Reading input (chunks)...")
    
    # Process in chunks to handle large file
    chunk_size = 5_000_000
    
    with open(OUTPUT_FILE, 'w') as f_out:
        # We need to group by protein ID, so we can't just stream.
        # But we can accumulate per protein and flush periodically.
        
        protein_buffer = {}  # id -> [(term, score), ...]
        
        for chunk in tqdm(pd.read_csv(INPUT_FILE, sep='\t', header=None, 
                                       names=['id', 'term', 'score'], 
                                       chunksize=chunk_size)):
            
            for _, row in chunk.iterrows():
                pid = row['id']
                if pid not in protein_buffer:
                    protein_buffer[pid] = []
                protein_buffer[pid].append((row['term'], row['score']))
            
            # Memory management: flush proteins that are unlikely to get more data
            # (This is tricky since file isn't sorted by protein)
            # For simplicity, just hold all in memory. 2GB file -> ~10GB RAM?
            # Let's try it. If OOM, we need a different approach.
        
        print("   ?뵩 Applying Top-K filter...")
        
        for pid, terms in tqdm(protein_buffer.items()):
            # Sort by score descending
            sorted_terms = sorted(terms, key=lambda x: x[1], reverse=True)
            
            # Take top K
            top_terms = sorted_terms[:TOP_K]
            
            # Write
            for term, score in top_terms:
                if score >= MIN_SCORE:
                    f_out.write(f"{pid}\t{term}\t{score:.5f}\n")
    
    print(f"?럦 Done! Output: {OUTPUT_FILE}")
    
    # Verify output
    import subprocess
    result = subprocess.run(['ls', '-lh', OUTPUT_FILE], capture_output=True, text=True)
    print(result.stdout)

if __name__ == "__main__":
    main()

