import os
import shutil
import glob
import subprocess

# Config
DB_DIR = "partial_dbs"
RES_DIR = "search_results"
FOLDSEEK_BIN = "/root/miniconda3/bin/foldseek"
TARGET_DB = "./artifacts/search_db/swissprot_db_pad"

def search_chunk(chunk_idx):
    """Run GPU-accelerated search for a single chunk"""
    tsv_out = os.path.join(RES_DIR, f"result_{chunk_idx}.tsv")
    out_db_path = os.path.join(DB_DIR, f"part_{chunk_idx}")
    
    # Skip if already done
    if os.path.exists(tsv_out) and os.path.getsize(tsv_out) > 1000:
        print(f"??Chunk {chunk_idx}: Already done")
        return "skip"
    
    # Check if DB exists
    if not os.path.exists(f"{out_db_path}.dbtype"):
        print(f"?좑툘 Chunk {chunk_idx}: No DB file")
        return "no_db"
    
    print(f"?뵊 Chunk {chunk_idx}: Searching with GPU...")
    
    aln_db = os.path.join(RES_DIR, f"aln_{chunk_idx}")
    tmp_dir = os.path.join(RES_DIR, f"tmp_{chunk_idx}")
    
    # Cleanup
    shutil.rmtree(tmp_dir, ignore_errors=True)
    for f in glob.glob(f"{aln_db}*"):
        try: os.remove(f)
        except: pass
    
    os.makedirs(tmp_dir, exist_ok=True)
    
    # Search with GPU enabled, full sensitivity
    cmd_search = [
        FOLDSEEK_BIN, "search", 
        out_db_path, TARGET_DB, aln_db, tmp_dir,
        "-a",
        "--gpu", "1",      # GPU acceleration (correct flag)
        "-s", "9.5",       # Full sensitivity
        "--threads", "12"  # Use all threads
    ]
    
    try:
        result = subprocess.run(cmd_search, check=True, capture_output=True, text=True, timeout=600)
        print(f"   Search done")
    except subprocess.CalledProcessError as e:
        print(f"??Chunk {chunk_idx}: Search failed")
        print(f"   {e.stderr[:300] if e.stderr else 'No error message'}")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return "search_fail"
    except subprocess.TimeoutExpired:
        print(f"??Chunk {chunk_idx}: Timeout")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return "timeout"
    
    # Convert
    fmt = "query,target,fident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits"
    cmd_conv = [
        FOLDSEEK_BIN, "convertalis", 
        out_db_path, TARGET_DB, aln_db, tsv_out, 
        "--format-output", fmt,
        "--threads", "12"
    ]
    
    try:
        subprocess.run(cmd_conv, check=True, capture_output=True, timeout=120)
    except:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return "convert_fail"
    
    # Cleanup
    shutil.rmtree(tmp_dir, ignore_errors=True)
    for f in glob.glob(f"{aln_db}*"):
        try: os.remove(f)
        except: pass
    
    print(f"??Chunk {chunk_idx}: Done!")
    return "success"

def main():
    print("?? GPU-Accelerated FoldSeek Search")
    print("   GPU: RTX 3060 Ti")
    print("   Sensitivity: 9.5 (full accuracy)")
    print()
    
    os.makedirs(RES_DIR, exist_ok=True)
    
    # Priority chunks first, then the rest
    priority = [4, 7, 9, 10, 11]
    all_chunks = list(range(113))
    remaining = [c for c in all_chunks if c not in priority]
    ordered_chunks = priority + remaining
    
    success = 0
    skipped = 0
    failed = []
    no_db = []
    
    for i in ordered_chunks:
        result = search_chunk(i)
        if result == "success":
            success += 1
        elif result == "skip":
            skipped += 1
        elif result == "no_db":
            no_db.append(i)
        else:
            failed.append(i)
    
    print(f"\n{'='*50}")
    print(f"?럦 Complete!")
    print(f"   Success: {success}")
    print(f"   Skipped: {skipped}")
    if no_db:
        print(f"   No DB: {no_db}")
    if failed:
        print(f"   Failed: {failed}")

if __name__ == "__main__":
    main()

