import sqlite3
import pandas as pd
import os
import sys
from tqdm import tqdm

# Config
FILE_BASE = "./results/final_submission/submission_Final_Repaired.tsv"
FILE_STACKING = "./results/final_submission/submission_Stacking_XGB.tsv"
OUTPUT_FILE = "./results/final_submission/submission_Hybrid_Rescue.tsv"
DB_FILE = "./artifacts/sqlite/rescue.db"

def main():
    print("?쉻 Rescue Mission: SQLite Edition (Ultra-Safe)...")
    
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)
        
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    
    # 1. Create Tables
    print("   ?뵪 Creating Database Tables...")
    cur.execute("CREATE TABLE IF NOT EXISTS predictions (id TEXT, term TEXT, score REAL, source INTEGER, PRIMARY KEY (id, term))")
    # source: 0=Base, 1=Stacking
    conn.commit()
    
    # 2. Ingest Base File (Source 0)
    print(f"   ?뱿 Ingesting Base File {FILE_BASE}...")
    # Use chunked read
    chunk_size = 1000000
    count = 0
    with pd.read_csv(FILE_BASE, sep='\t', names=['id', 'term', 'score'], chunksize=chunk_size, header=None) as reader:
        for chunk in tqdm(reader, desc="Base Ingest"):
            chunk['source'] = 0
            # Normalize scores
            chunk['score'] = chunk['score'].astype(float)
            
            # Using INSERT OR IGNORE just in case, though duplicates shouldn't exist in base
            data = list(chunk.itertuples(index=False, name=None))
            cur.executemany("INSERT OR IGNORE INTO predictions VALUES (?, ?, ?, ?)", data)
            count += len(chunk)
            
    conn.commit()
    print(f"      ??Ingested {count:,} Base Predictions.")
    
    # 3. Ingest Stacking File (Source 1)
    # This acts as an UPDATE or INSERT
    # Logic: If exists (from Base), UPDATE score and set source=1
    # If not exists, INSERT
    print(f"   ?뱿 Ingesting Stacking File {FILE_STACKING}...")
    
    with pd.read_csv(FILE_STACKING, sep='\t', names=['id', 'term', 'score'], chunksize=chunk_size, header=None) as reader:
        for chunk in tqdm(reader, desc="Stacking Ingest"):
            chunk['score'] = chunk['score'].astype(float)
            data = list(chunk[['id', 'term', 'score']].itertuples(index=False, name=None))
            
            # UPSERT Logic
            # "INSERT OR REPLACE" will replace the row (including source 0 -> source 1 effectively if we set source=1)
            # Actually we just want the new score.
            # We can use INSERT OR REPLACE INTO predictions (id, term, score, source) VALUES (?, ?, ?, 1)
            
            cur.executemany("INSERT OR REPLACE INTO predictions (id, term, score, source) VALUES (?, ?, ?, 1)", 
                            [(r[0], r[1], r[2]) for r in data])
            
    conn.commit()
    print("      ??Stacking Merged into Database.")
    
    # 4. Export
    print(f"   ?뱾 Exporting to {OUTPUT_FILE}...")
    
    # We select all
    cur.execute("SELECT id, term, score FROM predictions")
    
    with open(OUTPUT_FILE, 'w') as f:
        while True:
            rows = cur.fetchmany(1000000)
            if not rows: break
            
            for row in rows:
                f.write(f"{row[0]}\t{row[1]}\t{row[2]:.5f}\n")
                
    conn.close()
    
    # Cleanup DB if successful
    try:
        os.remove(DB_FILE)
    except:
        pass
        
    print(f"??Rescue Complete via SQLite.")
    print(f"?뱚 Output: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

