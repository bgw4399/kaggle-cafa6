import sqlite3
import pandas as pd
import os
from tqdm import tqdm

# === ?ㅼ젙 ===
FILE_OLD = "./results/submission_378_Pure_Repair.tsv"
WEIGHT_OLD = 0.7

FILE_NEW = "./results/final_submission/submission_final_filtered.tsv"
WEIGHT_NEW = 0.3

OUTPUT_FILE = "./results/final_submission/submission_Master_Ensemble.tsv"
DB_FILE = "ensemble_temp_local.db"
CHUNK_SIZE = 500_000

def init_db():
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("PRAGMA synchronous = OFF")
    c.execute("PRAGMA journal_mode = WAL")
    c.execute("PRAGMA cache_size = 100000")
    
    c.execute('''CREATE TABLE preds (
                    id TEXT,
                    term TEXT,
                    score REAL
                )''')
    conn.commit()
    return conn

def ingest_file(conn, filename, weight):
    print(f"?뱿 Ingesting {filename} (Weight: {weight})...")
    reader = pd.read_csv(filename, sep="\t", names=["id", "term", "score"], header=None, chunksize=CHUNK_SIZE)
    
    count = 0
    c = conn.cursor()
    
    for chunk in tqdm(reader, desc=f"Loading {filename}"):
        chunk["score"] = pd.to_numeric(chunk["score"], errors='coerce').fillna(0.0)
        # Clip scores > 1.0 (observed in 378 file)
        chunk["score"] = chunk["score"].clip(upper=1.0)
        chunk["score"] = chunk["score"] * weight
        
        data = chunk.to_records(index=False).tolist()
        c.executemany("INSERT INTO preds (id, term, score) VALUES (?, ?, ?)", data)
        count += len(data)
        
    conn.commit()
    print(f"   ??Inserted {count:,} rows.")

def merge_and_export(conn):
    print("?봽 Aggregating and Exporting...")
    c = conn.cursor()
    
    # Group By ID, Term and Sum Scores
    # Filtering > 0.001 happens here
    query = '''
        SELECT id, term, SUM(score) as final_score
        FROM preds
        GROUP BY id, term
        HAVING final_score > 0.001
        ORDER BY id
    '''
    
    c.execute(query)
    
    with open(OUTPUT_FILE, "w") as f:
        # write line by line
        count = 0
        for row in tqdm(c, desc="Writing Output"):
            # row: (id, term, score)
            f.write(f"{row[0]}\t{row[1]}\t{row[2]:.5f}\n")
            count += 1
            
    print(f"??Exported {count:,} rows to {OUTPUT_FILE}")

def main():
    print("?? Master Ensemble (SQLite Low-RAM) Starting...")
    
    conn = init_db()
    
    try:
        # 1. Ingest Old
        ingest_file(conn, FILE_OLD, WEIGHT_OLD)
        
        # 2. Ingest New
        ingest_file(conn, FILE_NEW, WEIGHT_NEW)
        
        # 3. Create Index for faster grouping?
        # Actually creating index on 200M rows takes time. 
        # But sorting group by needs it.
        print("Indexer running... (This takes a moment)")
        conn.execute("CREATE INDEX idx_all ON preds(id, term)")
        conn.commit()
        
        # 4. Merge
        merge_and_export(conn)
        
    finally:
        conn.close()
        # Clean up
        if os.path.exists(DB_FILE):
            os.remove(DB_FILE)
            print("?㏏ Temp DB cleaned up.")

if __name__ == "__main__":
    main()

