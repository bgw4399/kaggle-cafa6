import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import os

# Input Files
FILE_378 = "./results/submission_378.tsv"
FILE_371 = "./results/submission_SOTA_Kingdom_Max_371.tsv"

# Output File
OUTPUT_FILE = "./results/submission_378_371_Merged.tsv"

# Merge Strategy: MAX (take maximum score for each protein-term pair)
# This preserves high-confidence predictions from both sources

def main():
    print("?? Merging submission_378.tsv and submission_SOTA_Kingdom_Max_371.tsv...")
    print(f"   Strategy: MAX(score_378, score_371)")
    
    # Dictionary to store scores: {(protein_id, term): max_score}
    scores = defaultdict(float)
    
    # Load File 1 (378)
    print(f"\n?뱰 Loading {FILE_378}...")
    chunk_size = 5_000_000
    for chunk in tqdm(pd.read_csv(FILE_378, sep='\t', header=None, 
                                   names=['id', 'term', 'score'], 
                                   chunksize=chunk_size)):
        for _, row in chunk.iterrows():
            key = (row['id'], row['term'])
            scores[key] = max(scores[key], row['score'])
    
    print(f"   -> Loaded {len(scores):,} unique (protein, term) pairs")
    
    # Load File 2 (371) and merge
    print(f"\n?뱰 Loading and merging {FILE_371}...")
    new_pairs = 0
    updated_pairs = 0
    for chunk in tqdm(pd.read_csv(FILE_371, sep='\t', header=None, 
                                   names=['id', 'term', 'score'], 
                                   chunksize=chunk_size)):
        for _, row in chunk.iterrows():
            key = (row['id'], row['term'])
            old_score = scores[key]
            new_score = max(old_score, row['score'])
            if old_score == 0:
                new_pairs += 1
            elif new_score > old_score:
                updated_pairs += 1
            scores[key] = new_score
    
    print(f"   -> New pairs from 371: {new_pairs:,}")
    print(f"   -> Updated pairs (371 had higher score): {updated_pairs:,}")
    print(f"   -> Total unique pairs: {len(scores):,}")
    
    # Write output
    print(f"\n?뮶 Writing merged file...")
    with open(OUTPUT_FILE, 'w') as f:
        for (pid, term), score in tqdm(scores.items()):
            f.write(f"{pid}\t{term}\t{score:.5f}\n")
    
    print(f"\n?럦 Done! Output: {OUTPUT_FILE}")
    
    # Check output size
    import subprocess
    result = subprocess.run(['ls', '-lh', OUTPUT_FILE], capture_output=True, text=True)
    print(result.stdout)

if __name__ == "__main__":
    main()

