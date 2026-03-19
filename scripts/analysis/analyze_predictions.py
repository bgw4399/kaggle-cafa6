import pandas as pd
import numpy as np
import sys
import gc

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

FILES = {
    "SOTA_371": "./results/submission_SOTA_Kingdom_Max_371.tsv",
    "Best_378": "./results/submission_378.tsv",
    "Step1_256": "./results/submission_step1_aggregated.tsv"
}

def load_file_optimized(path):
    print(f"Loading {path}...", end=" ", flush=True)
    try:
        df = pd.read_csv(path, sep='\t', header=None, names=['id', 'term', 'score'], 
                         dtype={'score': 'float32', 'id': 'str', 'term': 'str'})
        print(f"Done. Rows: {len(df):,}")
        return df
    except Exception as e:
        print(f"\nFailed: {e}")
        return None

def analyze_single_file(name, path):
    print(f"\n?뱤 Analyzing {name}...")
    df = load_file_optimized(path)
    if df is None: return

    n_proteins = df['id'].nunique()
    preds_per_prot = df.groupby('id').size()
    score_mean = df['score'].mean()
    
    quantiles = df['score'].quantile([0.1, 0.5, 0.9]).to_dict()

    print(f"   Proteins: {n_proteins:,}")
    print(f"   Avg Preds/Prot: {preds_per_prot.mean():.1f}")
    print(f"   Score: Mean={score_mean:.3f}")
    print(f"   Score Quantiles: 10%={quantiles[0.1]:.3f}, 50%={quantiles[0.5]:.3f}, 90%={quantiles[0.9]:.3f}")
    
    # Histogram
    hist, bin_edges = np.histogram(df['score'], bins=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    print(f"   Score Dist (0-1, 0.2 bins): {hist}")

    del df
    gc.collect()

def compare_two_files(name1, path1, name2, path2):
    print(f"\n?뵕 Comparing {name1} vs {name2}...")
    df1 = load_file_optimized(path1)
    df2 = load_file_optimized(path2)
    
    if df1 is None or df2 is None: return

    # Vectorized similarity check
    # We want Jaccard index of the set of (id, term) pairs
    # Since these are too big to make set of strings, let's just use merge
    
    print("   Merging to find overlap...", end=" ", flush=True)
    # Inner join on id and term
    # To save memory, drop score first if not needed, or use it for correlation
    
    # Let's count intersection size via merge
    # We only care about id+term presence
    
    common = pd.merge(df1[['id', 'term']], df2[['id', 'term']], on=['id', 'term'], how='inner')
    intersection_count = len(common)
    
    len1 = len(df1)
    len2 = len(df2)
    union_count = len1 + len2 - intersection_count
    
    jaccard = intersection_count / union_count if union_count > 0 else 0
    
    print("Done.")
    print(f"   Jaccard Similarity: {jaccard:.4f}")
    print(f"   Intersection: {intersection_count:,}")
    print(f"   Unique to {name1}: {len1 - intersection_count:,}")
    print(f"   Unique to {name2}: {len2 - intersection_count:,}")

    del df1, df2, common
    gc.collect()

def main():
    # 1. Individual Analysis
    for name, path in FILES.items():
        analyze_single_file(name, path)
        gc.collect()

    # 2. Pairwise Comparison
    # Compare SOTA vs Best
    compare_two_files("SOTA_371", FILES["SOTA_371"], "Best_378", FILES["Best_378"])
    
    # Compare Best vs Step1
    compare_two_files("Best_378", FILES["Best_378"], "Step1_256", FILES["Step1_256"])

if __name__ == "__main__":
    main()

