import pandas as pd
import sys

FILE_DIA = "./results/submission_diamond_taxon_filtered.tsv"
PARENT = "GO:0005575" # Cellular Component
children = ["GO:0005829", "GO:0005886"]

def analyze_diamond():
    print(f"Loading {FILE_DIA}...")
    try:
        # Load sample
        df = pd.read_csv(FILE_DIA, sep='\t', header=None, names=['id', 'term', 'score'], nrows=2000000)
    except FileNotFoundError:
        print(f"File not found: {FILE_DIA}")
        return
    except Exception as e:
        print(f"Error: {e}")
        return

    print("Checking Parent-Child Consistency...")
    # Pivot sample
    # Optimization: Filter for relevant terms first
    relevant_terms = set(children + [PARENT])
    df_rel = df[df['term'].isin(relevant_terms)]
    
    if df_rel.empty:
        print("Sample contains none of the checked terms.")
        return
    
    pivot = df_rel.pivot(index='id', columns='term', values='score').fillna(0)
    
    if PARENT not in pivot.columns:
        print(f"??Parent {PARENT} NOT FOUND in sample.")
        has_child = False
        for c in children:
            if c in pivot.columns:
                print(f"   But Child {c} IS found (Max: {pivot[c].max()})")
                has_child=True
        if has_child:
            print("   CONCLUSION: Diamond output is also missing roots (Structural Violation).")
    else:
        print("Parent found. Checking violations...")
        for c in children:
            if c in pivot.columns:
                bad = pivot[pivot[c] > pivot[PARENT] + 1e-4]
                if len(bad) > 0:
                    print(f"   Violation: {c} > Parent in {len(bad)} cases.")
                else:
                    print(f"   Consistent: {c} <= Parent")

if __name__ == "__main__":
    analyze_diamond()

