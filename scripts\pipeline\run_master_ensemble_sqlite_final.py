import sqlite3
import pandas as pd
import os
from tqdm import tqdm

# ==========================================
# [?ㅼ젙] ?뚯씪 ?대쫫??蹂몄씤 ?섍꼍??留욊쾶 ?섏젙?섏꽭??
# ==========================================

# 1. 0.378???뚯씪 (Old Best)
# ?? "./results/submission_378.tsv" ?먮뒗 ?대떦 ?뚯씪???덈? 寃쎈줈
FILE_OLD = "./results/submission_378.tsv" 
WEIGHT_OLD = 0.7

# 2. 0.32???뚯씪 (New Scientific Filtered)
# ?? "./results/final_submission/submission_final_filtered.tsv"
FILE_NEW = "./results/final_submission/submission_final_filtered.tsv"
WEIGHT_NEW = 0.3

# 3. 寃곌낵 ??ν븷 ?뚯씪 ?대쫫
OUTPUT_FILE = "./results/final_submission/submission_Master_Ensemble.tsv"

# ?꾩떆 DB ?뚯씪 (?먮룞 ??젣??
DB_FILE = "ensemble_temp_final.db"
CHUNK_SIZE = 500_000  # 硫붾え由??덉쟾???꾪빐 50留?以꾩뵫 泥섎━

def init_db():
    # 湲곗〈 DB ?뚯씪 ??젣
    if os.path.exists(DB_FILE):
        try: os.remove(DB_FILE)
        except: pass
        
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    # ?띾룄 理쒖쟻???듭뀡
    c.execute("PRAGMA synchronous = OFF")
    c.execute("PRAGMA journal_mode = WAL")
    c.execute("PRAGMA cache_size = 100000") # 100MB 罹먯떆
    
    # ?뚯씠釉??앹꽦
    c.execute('''CREATE TABLE IF NOT EXISTS preds (
                    id TEXT,
                    term TEXT,
                    score REAL
                )''')
    conn.commit()
    return conn

def ingest_file(conn, filename, weight):
    print(f"?뱿 Loading & Ingesting: {filename} (Weight: {weight})")
    
    if not os.path.exists(filename):
        print(f"??Error: File not found: {filename}")
        return False

    # Pandas濡?泥?겕 ?⑥쐞 ?쎄린 (硫붾え由??덉빟)
    try:
        reader = pd.read_csv(filename, sep="\t", names=["id", "term", "score"], header=None, chunksize=CHUNK_SIZE)
    except Exception as e:
        print(f"??Error reading file: {e}")
        return False
    
    c = conn.cursor()
    total_rows = 0
    
    for chunk in tqdm(reader, desc=f"Imgerting {filename}"):
        # ?곗씠???뺤젣
        chunk["score"] = pd.to_numeric(chunk["score"], errors='coerce').fillna(0.0)
        
        # [以묒슂] 1.0 ?섎뒗 ?먯닔(1.003 ?? ?섎씪?닿린
        chunk["score"] = chunk["score"].clip(upper=1.0)
        
        # 媛以묒튂 ?곸슜
        chunk["score"] = chunk["score"] * weight
        
        # DB ?쎌엯
        data = chunk.to_records(index=False).tolist()
        c.executemany("INSERT INTO preds (id, term, score) VALUES (?, ?, ?)", data)
        total_rows += len(data)
        
    conn.commit()
    print(f"   ??Processed {total_rows:,} rows.")
    return True

def merge_and_export(conn):
    print("?봽 Merging scores in Database (Disk-based)...")
    c = conn.cursor()
    
    # ?몃뜳???앹꽦 (?띾룄 ?μ긽)
    print("   Building Index (This may take a minute)...")
    c.execute("CREATE INDEX IF NOT EXISTS idx_id_term ON preds(id, term)")
    conn.commit()
    
    # 吏묎퀎 荑쇰━ (Sum) 諛?0.001 誘몃쭔 ?쒓굅
    query = '''
        SELECT id, term, SUM(score) as final_score
        FROM preds
        GROUP BY id, term
        HAVING final_score > 0.001
    '''
    
    print(f"?뮶 Exporting to {OUTPUT_FILE}...")
    c.execute(query)
    
    with open(OUTPUT_FILE, "w") as f:
        count = 0
        for row in tqdm(c, desc="Writing Output"):
            # row: (id, term, score)
            f.write(f"{row[0]}\t{row[1]}\t{row[2]:.5f}\n")
            count += 1
            
    print(f"??Master Ensemble Completed! Total Predictions: {count:,}")

def main():
    print("?? Master Ensemble (Safe SQLite Ver.) Starting...")
    conn = init_db()
    
    try:
        # ?뚯씪 1 泥섎━
        if not ingest_file(conn, FILE_OLD, WEIGHT_OLD): return
        
        # ?뚯씪 2 泥섎━
        if not ingest_file(conn, FILE_NEW, WEIGHT_NEW): return
        
        # 蹂묓빀 諛????
        merge_and_export(conn)
        
    except Exception as e:
        print(f"??Unexpected Error: {e}")
        
    finally:
        conn.close()
        # ?꾩떆 ?뚯씪 ??젣
        if os.path.exists(DB_FILE):
            os.remove(DB_FILE)
            print("?㏏ Cleanup Done.")

if __name__ == "__main__":
    main()

