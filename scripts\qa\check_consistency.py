import pandas as pd
import sys

FILE_371 = "./results/submission_SOTA_Kingdom_Max_371.tsv"

# Parent: Cellular Component (GO:0005575)
# Children: Cytosol (GO:0005829), Plasma Membrane (GO:0005886)
PARENT = "GO:0005575"
CHILDREN = ["GO:0005829", "GO:0005886"]

def check_consistency():
    print(f"Checking consistency in {FILE_371}...")
    
    # Read first 1M rows to get a sample
    df = pd.read_csv(FILE_371, sep='\t', header=None, names=['id', 'term', 'score'], 
                     dtype={'score': 'float32', 'id': 'str', 'term': 'str'}, nrows=2000000)
    
    print(f"Loaded {len(df)} rows.")
    
    # Pivot to get terms as columns
    sample_ids = df['id'].unique()[:1000] # Check first 1000 proteins
    df_sample = df[df['id'].isin(sample_ids)]
    
    pivot = df_sample.pivot(index='id', columns='term', values='score').fillna(0)
    
    violations = 0
    total_checks = 0
    
    if PARENT not in pivot.columns:
        print(f"??Parent term {PARENT} NOT FOUND in sample columns!")
        # If parent is missing but children are present, that's a 100% violation effectively (implied parent score 0)
        for child in CHILDREN:
            if child in pivot.columns:
                n_child = pivot[child].gt(0).sum()
                print(f"   But child {child} has {n_child} non-zero entries.")
        return

    for child in CHILDREN:
        if child not in pivot.columns: continue
        
        # Check: Child > Parent
        # We allow small float error, so check if Child > Parent + 1e-5
        bad = pivot[pivot[child] > pivot[PARENT] + 1e-4]
        n_bad = len(bad)
        n_total = len(pivot)
        
        print(f"Checking {child} (Child) vs {PARENT} (Parent):")
        print(f"  Violations: {n_bad} / {n_total} proteins ({n_bad/n_total*100:.1f}%)")
        
        if n_bad > 0:
            print("  Example violation:")
            ex_id = bad.index[0]
            print(f"    ID: {ex_id}, Child: {bad.loc[ex_id, child]:.4f}, Parent: {bad.loc[ex_id, PARENT]:.4f}")

if __name__ == "__main__":
    check_consistency()

