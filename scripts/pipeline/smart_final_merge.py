import pandas as pd
import numpy as np
import sys
from collections import defaultdict, deque
import time

# Config
FILE_378 = "./results/submission_378.tsv"
FILE_371 = "./results/submission_SOTA_Kingdom_Max_371.tsv"
OBO_FILE = "./data/raw/train/go-basic.obo"
OUTPUT_FILE = "./results/submission_ensemble_repaired.tsv"
CHUNK_SIZE = 5000  # Number of proteins per chunk

def load_obo_dag(path):
    print(f"Loading OBO from {path}...")
    parents = defaultdict(set)
    term = None
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line == '[Term]':
                term = None
            elif line.startswith('id: '):
                term = line[4:].split()[0]
            elif line.startswith('is_a: ') and term:
                parent = line[6:].split()[0]
                parents[term].add(parent) # child -> parent
            elif line.startswith('relationship: part_of ') and term:
                parent = line[22:].split()[0]
                parents[term].add(parent)
    
    # Topological sort (Leaves first)
    print("Topological sort...")
    visited = set()
    order = []
    
    # We want to process children before parents.
    # Standard TopoSort on "Parent -> Child" graph gives Parent first.
    # We have "Child -> Parent" edges.
    # TopoSort on "Child -> Parent" graph gives Child first (Leaf first).
    
    # Build adjacency for sorting
    adj = parents # adj[u] = {v} means u -> v (u is child of v)
    
    # Helper for DFS
    temp_mark = set()
    perm_mark = set()
    
    def visit(n):
        if n in perm_mark: return
        if n in temp_mark: return # Cycle?
        
        temp_mark.add(n)
        for m in adj.get(n, []):
            visit(m)
        
        temp_mark.remove(n)
        perm_mark.add(n)
        order.append(n)
        
    all_nodes = list(adj.keys())
    for n in all_nodes:
        if n not in perm_mark:
            visit(n)
            
    # Order is now [Parent, ... Child] because recursive call finishes parent first?
    # Wait: DFS post-order on Child->Parent graph:
    # visit(Child):
    #   visit(Parent)... finishes Parent
    # finishes Child.
    # Output: Parent, ... Child.
    # We want LEAVES FIRST. So we just reverse this list? No.
    # We want to update Parent using Child. So Child must be processed BEFORE Parent?
    # No, to propagate UP (Child -> Parent), we check Child. Since Child score is fixed (it's a leaf), we update Parent.
    # Then Parent is updated. Then Grandparent uses updated Parent.
    # So we need Child FIRST.
    # DFS post-order gives [Parent, Child] (Parent finishes first).
    # Wait.
    # call visit(Child)
    #   call visit(Parent) -> finishes Parent.
    # finishes Child.
    # List: Parent, Child.
    # So iterating this list: Parent (no change), Child (check Parent? No).
    # We iterate list:
    #   Node: Parent. Update Parent's parents.
    #   Node: Child. Update Child's parents. (Parent is already processed? BAD).
    # We need child processed BEFORE parent.
    # So we need reverse of the Topological Sort?
    # List was [Parent, Child].
    # Reversed: [Child, Parent].
    # Process Child: update Parent.
    # Process Parent: update Grandparent.
    # Perfect.
    
    return order, parents

def process_chunk(chunk_ids, df1_all, df2_all, sort_order, parent_map):
    # Filter for this chunk
    d1 = df1_all[df1_all['id'].isin(chunk_ids)]
    d2 = df2_all[df2_all['id'].isin(chunk_ids)]
    
    # Pivot 371 (Requires repair)
    p2 = d2.pivot(index='id', columns='term', values='score').fillna(0)
    
    # Add missing columns (parents that might not be in 371 but are in ontology)
    # Optimization: Only add columns that are actually in sort_order and might be touched.
    # For simplicity, we assume robust pandas reindex.
    
    # To propagate, we need all terms in columns? 
    # Yes. But reindexing with 40k GO terms is heavy.
    # Only reindex with terms in sort_order that are relevant?
    # Let's just use the terms present + their parents?
    
    # Fast approach:
    # Iterate `sort_order`. If term is in columns, update its parents.
    # Parents might not be in columns. If so, create them.
    
    # Actually, initializing *all* columns is safer for vectorized ops.
    # But 40k cols x 5k rows = 200M float32 = 800MB. Fine.
    
    all_terms = list(set(p2.columns) | set(sort_order))
    # Filter to only GO terms?
    
    # Let's trust sort_order is comprehensive for the graph.
    # Only keep terms in p2 + structure.
    # p2 = p2.reindex(columns=sort_order, fill_value=0.0).astype(np.float32) 
    # ^ This might be huge if sort_order has all 40k terms.
    
    # Let's do a dictionary based update for sparse efficiency? 
    # No, vectorization is 100x faster.
    
    # Reindex only with terms from 371 + their ancestors.
    # (Simplified: Just reindex with all unique terms in both files + parents map keys)
    
    # Let's just assume we only care about terms expressed in the data + direct parents.
    # Iterating the full graph is safest.
    
    # Optimization: Filter sort_order to only nodes reachable from current leaves?
    # Skip for this script. We just run full graph. 800MB per chunk is fine.
    
    # Ensure all columns exist
    p2 = p2.reindex(columns=[t for t in sort_order if t in set(sort_order)], fill_value=0.0) # wait, just all sort_order
    # If sort_order has 40k items, p2 is 800MB.
    
    vals = p2.values
    cols = p2.columns.tolist()
    col_map = {c: i for i, c in enumerate(cols)}
    
    # Vectorized Propagation
    # Iterate Child -> Parent (Leaf First)
    # sort_order is [Parent ... Child]. Reversed is [Child ... Parent]
    # We agreed above: Post-order DFS yields [Parent, Child].
    # So we strictly want REVERSE of that list.
    
    # Wait, simple check:
    # A -> B (A is child of B).
    # DFS(A): visit(B). B finishes. A finishes. Order: [B, A].
    # Reverse: [A, B].
    # Iteration:
    #   Process A: Update B. B_score = max(B_score, A_score).
    #   Process B: Update C. ...
    # Correct.
    
    rev_order = list(reversed(sort_order))
    
    for child in rev_order:
        if child not in col_map: continue
        c_idx = col_map[child]
        
        # Get score vector for child
        c_vec = vals[:, c_idx]
        
        # If Child is all 0, skip
        if c_vec.max() <= 0: continue
        
        # Update parents
        parents = parent_map.get(child, set())
        for p in parents:
            if p not in col_map: continue
            p_idx = col_map[p]
            # In-place max
            vals[:, p_idx] = np.maximum(vals[:, p_idx], c_vec)
            
    # p2 is now Repaired.
    # Now merge with 378.
    
    p1 = d1.pivot(index='id', columns='term', values='score').fillna(0)
    p1 = p1.reindex(columns=p2.columns, fill_value=0.0) # Align columns
    
    # Weighted Ensemble
    # 0.5 * 378 + 0.5 * 371(Repaired)
    p_final = (p1 * 0.5) + (p2 * 0.5)
    
    # Filter low scores to reduce output size
    # p_final[p_final < 0.01] = np.nan
    # stack() drops NaNs.
    
    return p_final

def main():
    print("Initializing...")
    topo_order, parents = load_obo_dag(OBO_FILE)
    print(f"Graph terms: {len(topo_order)}")
    
    print("Loading dataframes...")
    df1 = pd.read_csv(FILE_378, sep='\t', header=None, names=['id', 'term', 'score'], 
                      dtype={'score': 'float32', 'id': 'str', 'term': 'str'},
                      usecols=[0,1,2])
    df2 = pd.read_csv(FILE_371, sep='\t', header=None, names=['id', 'term', 'score'], 
                      dtype={'score': 'float32', 'id': 'str', 'term': 'str'},
                      usecols=[0,1,2])
    
    unique_ids = sorted(list(set(df1['id']) | set(df2['id'])))
    print(f"Total Unique Proteins: {len(unique_ids)}")
    
    # Prepare output
    with open(OUTPUT_FILE, 'w') as f:
        pass # Clear file
        
    # Process in chunks
    for i in range(0, len(unique_ids), CHUNK_SIZE):
        chunk = unique_ids[i : i + CHUNK_SIZE]
        print(f"Processing chunk {i}-{i+len(chunk)}...")
        
        res_df = process_chunk(chunk, df1, df2, topo_order, parents)
        
        # Melt and save
        out = res_df.stack().reset_index()
        out.columns = ['id', 'term', 'score']
        
        # Filter low scores
        out = out[out['score'] >= 0.001]
        
        # Append to file
        out.to_csv(OUTPUT_FILE, sep='\t', header=False, index=False, mode='a', float_format='%.3f')
        
    print(f"Done. Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()


