import sqlite3
import pandas as pd
import os
from tqdm import tqdm

# Config
DB_FILE = "ensemble_merge.db"
FILE_ESM = "./results/final_submission/final_esm_full.tsv"
FILE_PROT = "./results/final_submission/final_prott5_full.tsv"
OUTPUT_FILE = "./results/final_submission/submission_Final_Scientific_Ensemble.tsv"

W_ESM = 0.6
W_PROT = 0.4

def main():
    print("🚀 robust Merge via SQLite (Disk-Based)...")
    
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)
        
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    
    # 1. Create Table (Single table for all preds)
    # columns: pid, term, score, source_weight
    cur.execute("""
        CREATE TABLE preds (
            pid TEXT,
            term TEXT,
            score REAL,
            weight REAL
        )
    """)
    conn.commit()
    
    # turn off disk sync for speed
    cur.execute("PRAGMA synchronous = OFF")
    cur.execute("PRAGMA journal_mode = MEMORY")
    
    # 2. Ingest ESM
    print("   📥 Ingesting ESM2-15B...")
    chunk_size = 100000
    with open(FILE_ESM, 'r') as f:
        batch = []
        for line in tqdm(f, desc="Reading ESM"):
            p = line.strip().split('\t')
            if len(p) < 3: continue
            # (pid, term, score, weight)
            batch.append((p[0], p[1], float(p[2]), W_ESM))
            
            if len(batch) >= chunk_size:
                cur.executemany("INSERT INTO preds VALUES (?,?,?,?)", batch)
                batch = []
        if batch:
            cur.executemany("INSERT INTO preds VALUES (?,?,?,?)", batch)
    conn.commit()
    
    # 3. Ingest ProtT5
    print("   📥 Ingesting ProtT5-XL...")
    with open(FILE_PROT, 'r') as f:
        batch = []
        for line in tqdm(f, desc="Reading ProtT5"):
            p = line.strip().split('\t')
            if len(p) < 3: continue
            batch.append((p[0], p[1], float(p[2]), W_PROT))
            
            if len(batch) >= chunk_size:
                cur.executemany("INSERT INTO preds VALUES (?,?,?,?)", batch)
                batch = []
        if batch:
            cur.executemany("INSERT INTO preds VALUES (?,?,?,?)", batch)
    conn.commit()
    
    # 4. Indexing (Critical for Group By)
    print("   🗂️ Indexing (This may take time)...")
    cur.execute("CREATE INDEX idx_pid_term ON preds(pid, term)")
    conn.commit()
    
    # 5. Aggregate and Export
    print("   ⚗️ Aggregating and Exporting...")
    # SQL: SUM(score * 1.0) because we already stored 'score' * 'weight'? 
    # No, we stored 'score' and 'weight'. So SUM(score * weight).
    # Wait, if we stored (score, W_ESM), and for the same (pid, term) we have (s1, 0.6) and (s2, 0.4).
    # SUM(s1*0.6) + SUM(s2*0.4) is correct.
    # What if a term exists only in ESM? SUM(s1*0.6). Correct (Prot part is 0).
    
    # We can use pandas to read sql chunk by chunk and write csv
    
    query = """
        SELECT pid, term, SUM(score * weight) as final_score
        FROM preds
        GROUP BY pid, term
        HAVING final_score > 0.001
        ORDER BY pid, final_score DESC
    """
    
    # Streaming export
    with open(OUTPUT_FILE, 'w') as f_out:
        # Cursor iterator
        cur.execute(query)
        while True:
            # Fetch many
            rows = cur.fetchmany(chunk_size)
            if not rows: break
            
            for r in rows:
                # r = (pid, term, score)
                f_out.write(f"{r[0]}\t{r[1]}\t{r[2]:.5f}\n")
                
    print(f"✅ Ensemble Complete: {OUTPUT_FILE}")
    
    # Cleanup
    conn.close()
    try:
        os.remove(DB_FILE)
        print("   🗑️ Removed temp DB.")
    except:
        pass

if __name__ == "__main__":
    main()
