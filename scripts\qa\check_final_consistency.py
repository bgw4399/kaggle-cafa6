import pandas as pd
import sys

# D Drive file
FILE_FINAL = "./results/submission_ESM15B_Ensemble_Weighted_Repaired.tsv"

PARENT = "GO:0005575" # Cellular Component
CHILDREN = ["GO:0005829", "GO:0005886"] # Cytosol, Plasma Mem

def check_final():
    print(f"Loading {FILE_FINAL}...")
    try:
        # Load sample 2M rows
        df = pd.read_csv(FILE_FINAL, sep='\t', header=None, names=['id', 'term', 'score'], nrows=2000000)
    except FileNotFoundError:
        print(f"File not found: {FILE_FINAL}")
        return
    except Exception as e:
        print(f"Error: {e}")
        return

    print("Checking Consistency...")
    # Filter for relevant terms
    rel = df[df['term'].isin(CHILDREN + [PARENT])]
    if rel.empty:
        print("Sample has no relevant terms.")
        return
        
    pivot = rel.pivot(index='id', columns='term', values='score').fillna(0)
    
    if PARENT not in pivot.columns:
        print(f"??Parent {PARENT} STILL MISSING. Fix Failed.")
        return
    
    # Check Logic
    print(f"Parent {PARENT} found. Checking Logic...")
    violations = 0
    total = 0
    
    for c in CHILDREN:
        if c in pivot.columns:
            # Allow small float error
            bad = pivot[pivot[c] > pivot[PARENT] + 0.01] 
            total += len(pivot)
            violations += len(bad)
            
            if len(bad) > 0:
                print(f"   {c}: {len(bad)}/{len(pivot)} violations.")
                print(f"   Example: {bad.index[0]} Child={bad.iloc[0][c]:.4f} Parent={bad.iloc[0][PARENT]:.4f}")
            else:
                print(f"   ??{c}: 0 Violations.")

    if violations == 0:
        print("\n?럦 SUCCESS: All Structure Issues Resolved!")
    else:
        print(f"\n?좑툘 WARNING: {violations} Violations found.")

if __name__ == "__main__":
    check_final()

