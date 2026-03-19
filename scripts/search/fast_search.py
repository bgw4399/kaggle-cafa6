import os
import shutil
import glob
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

# Config
DB_DIR = "partial_dbs"
RES_DIR = "search_results"
FOLDSEEK_BIN = "/root/miniconda3/bin/foldseek"
TARGET_DB = "./artifacts/search_db/swissprot_db"

# Speed optimized: 3 parallel, reduced sensitivity
PARALLEL_SEARCHES = 3
SENSITIVITY = "7.5"  # Faster than 9.5, still accurate

def search_chunk(chunk_idx):
    """Run search for a single chunk"""
    tsv_out = os.path.join(RES_DIR, f"result_{chunk_idx}.tsv")
    out_db_path = os.path.join(DB_DIR, f"part_{chunk_idx}")
    
    # Skip if already done
    if os.path.exists(tsv_out) and os.path.getsize(tsv_out) > 1000:
        return (chunk_idx, "skip")
    
    # Check if DB exists
    if not os.path.exists(f"{out_db_path}.dbtype"):
        return (chunk_idx, "no_db")
    
    aln_db = os.path.join(RES_DIR, f"aln_{chunk_idx}")
    tmp_dir = os.path.join(RES_DIR, f"tmp_{chunk_idx}")
    
    shutil.rmtree(tmp_dir, ignore_errors=True)
    for f in glob.glob(f"{aln_db}*"):
        try: os.remove(f)
        except: pass
    
    os.makedirs(tmp_dir, exist_ok=True)
    
    # CPU search with reduced sensitivity for speed
    cmd_search = [
        FOLDSEEK_BIN, "search", 
        out_db_path, TARGET_DB, aln_db, tmp_dir,
        "-a",
        "-s", SENSITIVITY,
        "--threads", "4"
    ]
    
    try:
        subprocess.run(cmd_search, check=True, capture_output=True, timeout=600)
    except:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return (chunk_idx, "search_fail")
    
    fmt = "query,target,fident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits"
    cmd_conv = [FOLDSEEK_BIN, "convertalis", out_db_path, TARGET_DB, aln_db, tsv_out, "--format-output", fmt, "--threads", "4"]
    
    try:
        subprocess.run(cmd_conv, check=True, capture_output=True, timeout=120)
    except:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return (chunk_idx, "convert_fail")
    
    shutil.rmtree(tmp_dir, ignore_errors=True)
    for f in glob.glob(f"{aln_db}*"):
        try: os.remove(f)
        except: pass
    
    return (chunk_idx, "success")

def main():
    print(f"?? Fast Parallel Search (s={SENSITIVITY}, {PARALLEL_SEARCHES} workers)")
    
    os.makedirs(RES_DIR, exist_ok=True)
    
    # Priority chunks first
    priority = [4, 7, 9, 10, 11]
    all_chunks = list(range(113))
    remaining = [c for c in all_chunks if c not in priority]
    ordered = priority + remaining
    
    success = 0
    skipped = 0
    failed = []
    no_db = []
    
    with ProcessPoolExecutor(max_workers=PARALLEL_SEARCHES) as executor:
        futures = {executor.submit(search_chunk, i): i for i in ordered}
        
        for future in as_completed(futures):
            idx = futures[future]
            try:
                _, result = future.result()
                if result == "success":
                    success += 1
                    print(f"??{idx}")
                elif result == "skip":
                    skipped += 1
                    print(f"??{idx}")
                elif result == "no_db":
                    no_db.append(idx)
                    print(f"?좑툘 {idx}: No DB")
                else:
                    failed.append(idx)
                    print(f"??{idx}")
            except Exception as e:
                failed.append(idx)
                print(f"??{idx}: {e}")
    
    print(f"\n?럦 Done! Success:{success} Skip:{skipped}")
    if no_db: print(f"No DB: {no_db}")
    if failed: print(f"Failed: {failed}")

if __name__ == "__main__":
    main()

