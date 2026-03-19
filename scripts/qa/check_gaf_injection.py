import pandas as pd
import sys

FILE_FINAL = "./results/submission_ESM15B_Ensemble_Weighted_Final.tsv"
GAF_FILE = "./gaf_positive_preds.tsv"

def check_gaf():
    print("?? Checking GAF Injection...")
    
    # 1. Load GAF Positives
    gaf_pairs = []
    try:
        with open(GAF_FILE, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    gaf_pairs.append((parts[0], parts[1]))
    except:
        print("GAF File not found.")
        return

    print(f"Loaded {len(gaf_pairs)} GAF pairs.")
    if not gaf_pairs: return

    # 2. Check a sample of them in Final File
    # Loading full file is slow, so let's grep a few specific IDs?
    # Or just load the file since we need to verify scores.
    # Let's load top 100 GAF pairs and check if they exist with score 1.0
    
    check_subset = gaf_pairs[:100]
    check_ids = set([p[0] for p in check_subset])
    
    print(f"Loading Final File (filtering for {len(check_ids)} IDs)...")
    
    found = {}
    try:
        # Optimization: use chunking to find specific IDs
        chunk_size = 1_000_000
        for chunk in pd.read_csv(FILE_FINAL, sep='\t', header=None, names=['id', 'term', 'score'], chunksize=chunk_size):
            # Check for our target IDs
            rel = chunk[chunk['id'].isin(check_ids)]
            for _, row in rel.iterrows():
                found[(row['id'], row['term'])] = row['score']
            
            # If we found all, break? No, might span chunks.
    except FileNotFoundError:
        print("Final file not found yet.")
        return

    # 3. Verify
    print("Verifying Scores...")
    correct = 0
    missing = 0
    incorrect_score = 0
    
    for pid, term in check_subset:
        if (pid, term) in found:
            score = found[(pid, term)]
            if score >= 0.99:
                correct += 1
            else:
                incorrect_score += 1
                print(f"??Incorrect Score: {pid} {term} = {score:.4f} (Expected 1.0)")
        else:
            missing += 1
            # print(f"??Missing Pair: {pid} {term}")

    print(f"??Correct (1.0): {correct}")
    print(f"??Incorrect Score: {incorrect_score}")
    print(f"??Missing in File: {missing} (Maybe filtered out?)")

if __name__ == "__main__":
    check_gaf()

