import pandas as pd
import sys

FILE_STEP1 = "./results/submission_step1_aggregated.tsv"
PARENT = "GO:0005575" # Cellular Component
CHILDREN = ["GO:0005829", "GO:0005886"] # Cytosol, Plasma Mem

def analyze_step1():
    print(f"Loading {FILE_STEP1}...")
    try:
        # Step 1 likely has no header or specific columns?
        # User said "submission_step1_aggregated.tsv"
        df = pd.read_csv(FILE_STEP1, sep='\t', header=None, names=['id', 'term', 'score'], nrows=2000000)
    except Exception as e:
        print(f"Error: {e}")
        return

    print("Checking Parent-Child Consistency...")
    pivot = df[df['id'].isin(df['id'].unique()[:1000])].pivot(index='id', columns='term', values='score').fillna(0)
    
    if PARENT not in pivot.columns:
        print(f"??Parent {PARENT} NOT FOUND in first 2M rows sample.")
        # Check children
        has_child = False
        for c in CHILDREN:
            if c in pivot.columns:
                print(f"   But Child {c} IS found (Max score: {pivot[c].max()})")
                has_child = True
        if has_child:
            print("   CONCLUSION: Structural Violation Confirmed. Root terms missing.")
    else:
        print(f"Parent {PARENT} found. Checking logical violations...")
        for c in CHILDREN:
            if c in pivot.columns:
                bad = pivot[pivot[c] > pivot[PARENT] + 1e-4]
                print(f"   {c} > Parent: {len(bad)} violations")

if __name__ == "__main__":
    analyze_step1()

