import numpy as np
import json
import sys

def check():
    print("?뵮 Checking Species Index Alignment...")
    
    try:
        train_idx = np.load("data/embeddings/taxonomy/train_species_idx.npy").reshape(-1)
        test_idx = np.load("data/embeddings/taxonomy/test_species_idx.npy").reshape(-1)
        
        train_uniq = np.unique(train_idx)
        test_uniq = np.unique(test_idx)
        
        print(f"   Train Unique Indices: {len(train_uniq)}")
        print(f"   Test Unique Indices: {len(test_uniq)}")
        
        intersect = np.intersect1d(train_uniq, test_uniq)
        print(f"   Intersection: {len(intersect)}")
        
        if len(intersect) < 10:
            print("?좑툘 WARNING: Very low intersection! Indices might be mismatched.")
        else:
            print("??Indices overlap seems okay.")

        # Check vocab
        with open("data/embeddings/taxonomy/species_vocab.json", "r") as f:
            vocab = json.load(f)
            print(f"   Vocab Size: {len(vocab)}")
            
            # Check 1073
            # Vocab is likely "TaxonID": Index or Index: "TaxonID"
            # It's usually a dict.
            keys = list(vocab.keys())[:5]
            vals = list(vocab.values())[:5]
            print(f"   Vocab Sample Keys: {keys}")
            print(f"   Vocab Sample Values: {vals}")
            
            # Check max index
            max_idx = max(vocab.values()) if vocab else 0
            print(f"   Max Index in Vocab: {max_idx}")
            
            # Find what 1073 is
            target = 1073
            # If values are indices
            target = 1073
            found_k = None
            for k, v in vocab.items():
                if v == target:
                    found_k = k
                    break
            print(f"   Index 1073 corresponds to TaxonID: {found_k}")
            
    except Exception as e:
        print(f"??Error: {e}")

if __name__ == "__main__":
    check()

