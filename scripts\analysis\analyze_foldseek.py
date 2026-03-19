import pandas as pd
import sys
from collections import defaultdict

FILE_FS = "./results/sorted_foldseek.tsv"
FILE_378 = "./results/submission_378.tsv"
OBO_FILE = "./data/raw/train/go-basic.obo"

# Parent-Child check
PARENT = "GO:0005575"
CHILDREN = ["GO:0005829", "GO:0005886"]

def analyze_foldseek():
    print(f"Loading {FILE_FS}...")
    try:
        df = pd.read_csv(FILE_FS, sep='\t', header=None, names=['id', 'term', 'score'], 
                         dtype={'score': 'float32', 'id': 'str', 'term': 'str'})
        print(f"Rows: {len(df):,}")
    except Exception as e:
        print(f"Error loading: {e}")
        return

    n_proteins = df['id'].nunique()
    print(f"Unique Proteins: {n_proteins:,}")
    
    # Score distribution
    print(f"Score Stats:\n{df['score'].describe()}")
    
    # Check Consistency
    print("\nChecking Consistency (Sample)...")
    # Join with itself to check parent-child? No, pivot.
    sample_ids = df['id'].unique()[:1000]
    df_sample = df[df['id'].isin(sample_ids)]
    pivot = df_sample.pivot(index='id', columns='term', values='score').fillna(0)
    
    if PARENT in pivot.columns:
        valid_children = [c for c in CHILDREN if c in pivot.columns]
        for c in valid_children:
            bad = pivot[pivot[c] > pivot[PARENT] + 1e-4]
            if len(bad) > 0:
                print(f"??Violation: {c} > {PARENT} in {len(bad)}/{len(pivot)} cases.")
            else:
                print(f"??Consistent: {c} <= {PARENT}")
    else:
        print(f"?좑툘 Parent {PARENT} not found in sample. Might be missing root terms?")
        
    # Compare with 378 (Coverage)
    print(f"\nComparing with {FILE_378} (First 1M rows)...")
    df378 = pd.read_csv(FILE_378, sep='\t', header=None, names=['id', 'term', 'score'], nrows=1000000)
    ids_378 = set(df378['id'].unique())
    ids_fs = set(df['id'].unique())
    
    common = ids_378.intersection(ids_fs)
    print(f"Proteins in 378 sample: {len(ids_378)}")
    print(f"Proteins in Foldseek intersecting 378 sample: {len(common)}")
    
    # Check if Foldseek has proteins NOT in 378?
    # Usually 378 covers all Targets.
    # If Foldseek covers fewer, it's sparse.
    
    # Important: Does Foldseek have High Confidence?
    high_conf = len(df[df['score'] > 0.9])
    print(f"High Confidence Predictions (>0.9): {high_conf:,} ({high_conf/len(df)*100:.1f}%)")

if __name__ == "__main__":
    analyze_foldseek()


