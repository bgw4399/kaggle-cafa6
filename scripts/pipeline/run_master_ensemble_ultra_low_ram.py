import sqlite3
import pandas as pd
import os
import gc
from tqdm import tqdm

# ==========================================
# [珥덉??ъ뼇 紐⑤뱶] ?ㅼ젙
# ==========================================
FILE_OLD = "./results/submission_378.tsv" 
WEIGHT_OLD = 0.7

FILE_NEW = "./results/final_submission/submission_final_filtered.tsv"
WEIGHT_NEW = 0.3

OUTPUT_FILE = "./results/final_submission/submission_Master_Ensemble.tsv"
DB_FILE = "ensemble_ultra_light.db"
CHUNK_SIZE = 50000  # 泥?겕 ?ш린 ???異뺤냼 (5留?以?

def init_db():
    if os.path.exists(DB_FILE):
        try: os.remove(DB_FILE)
        except: pass
        
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    # ?뮙 硫붾え由??덉빟 ?앺뙋???ㅼ젙 ?뮙
    c.execute("PRAGMA synchronous = OFF")
    c.execute("PRAGMA journal_mode = DELETE") # WAL 紐⑤뱶 ?댁젣 (硫붾え由??덉빟)
    c.execute("PRAGMA cache_size = 10000")    # 罹먯떆瑜?40MB濡??쒗븳
    c.execute("PRAGMA temp_store = FILE")     # ?꾩떆 ?곗씠?곕? 媛뺤젣濡??붿뒪?ъ뿉 ???(RAM ?ъ슜 ?덊븿)
    
    c.execute('''CREATE TABLE IF NOT EXISTS preds (
                    id TEXT,
                    term TEXT,
                    score REAL
                )''')
    conn.commit()
    return conn

def ingest_file(conn, filename, weight):
    print(f"?뱿 Ingesting: {filename} (Weight: {weight})")
    if not os.path.exists(filename):
        print(f"??File not found: {filename}")
        return False

    c = conn.cursor()
    try:
        # chunksize瑜?以꾩뿬??硫붾え由??쇳겕 媛먯냼
        reader = pd.read_csv(filename, sep="\t", names=["id", "term", "score"], header=None, chunksize=CHUNK_SIZE)
        
        count = 0
        for chunk in tqdm(reader, desc=f"Loading {filename}"):
            chunk["score"] = pd.to_numeric(chunk["score"], errors='coerce').fillna(0.0)
            chunk["score"] = chunk["score"].clip(upper=1.0) # ?먯닔 蹂댁젙
            chunk["score"] = chunk["score"] * weight
            
            data = chunk.to_records(index=False).tolist()
            c.executemany("INSERT INTO preds (id, term, score) VALUES (?, ?, ?)", data)
            count += len(data)
            
            # 紐낆떆??硫붾え由??댁젣
            del data
            del chunk
            
        conn.commit()
        print(f"   ??Processed {count:,} rows.")
        return True
        
    except Exception as e:
        print(f"??Error during ingestion: {e}")
        return False

def merge_and_export(conn):
    print("?봽 Merging on Disk (No Indexing to save RAM)...")
    c = conn.cursor()
    
    # ?몃뜳???앹꽦 ?④퀎瑜??앸왂?⑸땲?? (?몃뜳???앹꽦 ??硫붾え由??ㅽ뙆?댄겕 諛쒖깮 媛??
    # ???GROUP BY媛 ?붿뒪??湲곕컲 ?뺣젹???섑뻾?섍쾶 ?⑸땲??
    
    query = '''
        SELECT id, term, SUM(score) as final_score
        FROM preds
        GROUP BY id, term
        HAVING final_score > 0.001
    '''
    
    print(f"?뮶 Streaming Query Result to {OUTPUT_FILE}...")
    
    # fetchmany濡?議고쉶?섏뿬 硫붾え由??덉빟
    c.execute(query)
    
    with open(OUTPUT_FILE, "w") as f:
        count = 0
        while True:
            rows = c.fetchmany(10000) # 1留?媛쒖뵫 媛?몄삤湲?
            if not rows: break
            
            lines = []
            for row in rows:
                lines.append(f"{row[0]}\t{row[1]}\t{row[2]:.5f}\n")
            f.writelines(lines)
            count += len(rows)
            
            if count % 1000000 == 0:
                print(f"   ... Written {count:,} rows")
            
    print(f"??Completed! Total Predictions: {count:,}")

def main():
    print("?? Master Ensemble (Ultra Low-RAM Ver) Starting...")
    conn = init_db()
    
    try:
        if ingest_file(conn, FILE_OLD, WEIGHT_OLD) and ingest_file(conn, FILE_NEW, WEIGHT_NEW):
            merge_and_export(conn)
            
    except Exception as e:
        print(f"??Fatal Error: {e}")
        
    finally:
        conn.close()
        if os.path.exists(DB_FILE):
            try: os.remove(DB_FILE)
            except: pass
            print("?㏏ Cleanup Done.")

if __name__ == "__main__":
    main()

