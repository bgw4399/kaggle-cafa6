import pandas as pd
import sys
import gc

# Define interesting GO terms
# Tail related
GO_TAIL_MORPH = "GO:0036342" # post-anal tail morphogenesis
GO_TAIL_DEV = "GO:0060349"   # tail development

# Mouse/Sensory related (often expanded in mice)
GO_SMELL = "GO:0007608"      # sensory perception of smell
GO_PHEROMONE = "GO:0050911"  # detection of pheromone

INTERESTING_TERMS = {
    GO_TAIL_MORPH: "Post-anal tail morphogenesis",
    GO_TAIL_DEV: "Tail development",
    GO_SMELL: "Sensory perception of smell",
    GO_PHEROMONE: "Detection of pheromone",
    "GO:0005737": "Cytoplasm (Control)",
    "GO:0005634": "Nucleus (Control)"
}

FILES = {
    "Best_378": "./results/submission_378.tsv",
    "SOTA_371": "./results/submission_SOTA_Kingdom_Max_371.tsv"
}

def load_file(path):
    print(f"Loading {path}...", end=" ", flush=True)
    try:
        # Load only necessary columns
        df = pd.read_csv(path, sep='\t', header=None, names=['id', 'term', 'score'], 
                         dtype={'score': 'float32', 'id': 'str', 'term': 'str'})
        print(f"Done. Rows: {len(df):,}")
        return df
    except Exception as e:
        print(f"\nFailed to load {path}: {e}")
        return None

def analyze_terms(df, name):
    print(f"\nAnalyzing terms for {name}...")
    
    # 1. Total score per term (Sum of probabilities)
    # This gives an idea of "how much" this function is predicted across the dataset
    term_sums = df.groupby('term')['score'].sum().sort_values(ascending=False)
    
    # 2. Count of high-conf predictions (> 0.5)
    high_conf = df[df['score'] > 0.5].groupby('term').size()
    
    return term_sums, high_conf

def main():
    dfs = {}
    term_stats = {}
    
    for name, path in FILES.items():
        df = load_file(path)
        if df is not None:
            dfs[name] = df
            term_stats[name] = analyze_terms(df, name)
            del df
            gc.collect() # Free memory
    
    if len(dfs) < 2 and len(term_stats) < 2:
        print("Need both files to compare.")
        return

    # Compare
    print("\n" + "="*40)
    print("COMPARISON: Best_378 vs SOTA_371")
    print("="*40)
    
    stats1 = term_stats["Best_378"] # (Sums, HighConfCounts)
    stats2 = term_stats["SOTA_371"]
    
    # 1. Check Interesting Terms
    print("\n>>> Specific Biological Checks (Human Tail / Mouse Distinction)")
    print(f"{'GO Term':<12} {'Description':<30} {'Best_378 (Sum)':<15} {'SOTA_371 (Sum)':<15} {'Diff':<10}")
    print("-" * 90)
    
    for term, desc in INTERESTING_TERMS.items():
        val1 = stats1[0].get(term, 0.0)
        val2 = stats2[0].get(term, 0.0)
        diff = val1 - val2
        print(f"{term:<12} {desc:<30} {val1:>10.2f} {val2:>15.2f} {diff:>10.2f}")

    # 2. Top differences
    print("\n>>> Top Differences in Total Prediction Score (Absolute Diff)")
    # Align the two series
    all_terms = stats1[0].index.union(stats2[0].index)
    s1 = stats1[0].reindex(all_terms, fill_value=0)
    s2 = stats2[0].reindex(all_terms, fill_value=0)
    
    diff = (s1 - s2)
    abs_diff = diff.abs().sort_values(ascending=False)
    
    print(f"{'GO Term':<12} {'Best_378':<15} {'SOTA_371':<15} {'Difference (378 - 371)':<25}")
    print("-" * 70)
    for term in abs_diff.head(20).index:
        v1 = s1[term]
        v2 = s2[term]
        d = v1 - v2
        print(f"{term:<12} {v1:>10.2f} {v2:>15.2f} {d:>20.2f}")
        
    # 3. Missing predictions analysis
    # Which terms are heavily predicted in one but almost absent in the other?
    print("\n>>> Terms present in Best_378 but missing/low in SOTA_371 (Top 10)")
    # Filter where SOTA is low (< 10 score sum) but Best is high
    only_in_378 = s1[(s2 < 10) & (s1 > 100)].sort_values(ascending=False)
    for term in only_in_378.head(10).index:
        print(f"{term}: 378={s1[term]:.1f}, 371={s2[term]:.1f}")

    print("\n>>> Terms present in SOTA_371 but missing/low in Best_378 (Top 10)")
    only_in_371 = s2[(s1 < 10) & (s2 > 100)].sort_values(ascending=False)
    for term in only_in_371.head(10).index:
        print(f"{term}: 378={s1[term]:.1f}, 371={s2[term]:.1f}")

if __name__ == "__main__":
    main()

