import os
import shutil
import glob
import subprocess
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Configuration
FASTA_FILE = "./data/raw/test/testsuperset.fasta"
CHUNK_SIZE = 2000
TEMP_DIR_PREFIX = "temp_pdb_chunk_"
DB_DIR = "partial_dbs"
RES_DIR = "search_results"
API_URL = "https://alphafold.ebi.ac.uk/api/prediction/{}"
FOLDSEEK_BIN = "/root/miniconda3/bin/foldseek"
TARGET_DB = "./artifacts/search_db/swissprot_db"

# Chunks to repair: 3 + [30-39]
MISSING_CHUNKS = [3, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]

def get_all_ids_from_fasta(fasta_path):
    ids = []
    print(f"?뱰 Reading IDs from {fasta_path}...")
    with open(fasta_path, "r") as f:
        for line in f:
            if line.startswith(">"):
                parts = line.strip().split()
                header = parts[0].replace(">", "")
                if ":" in header:
                    protein_id = header.split(":")[1]
                else:
                    protein_id = header
                ids.append(protein_id)
    return list(dict.fromkeys(ids))  # Preserve order, remove duplicates

def download_single(pid, save_dir):
    save_path = os.path.join(save_dir, f"{pid}.pdb")
    if os.path.exists(save_path):
        return True

    try:
        res = requests.get(API_URL.format(pid), timeout=10)
        if res.status_code != 200: return False
        data = res.json()
        if not data or not isinstance(data, list): return False
        pdb_url = data[0].get('pdbUrl')
        if not pdb_url: return False
        pdb_res = requests.get(pdb_url, timeout=15)
        if pdb_res.status_code == 200:
            with open(save_path, "wb") as f:
                f.write(pdb_res.content)
            return True
    except:
        pass
    return False

def repair_chunk(chunk_idx, all_ids):
    print(f"\n{'='*50}")
    print(f"?뵩 Repairing Chunk {chunk_idx}")
    print(f"{'='*50}")
    
    start = chunk_idx * CHUNK_SIZE
    end = start + CHUNK_SIZE
    chunk_ids = all_ids[start:end]
    
    if not chunk_ids:
        print(f"?좑툘 No IDs for chunk {chunk_idx}, skipping...")
        return False
    
    print(f"   Target IDs: {len(chunk_ids)} ({start} - {end})")
    
    temp_dir = f"{TEMP_DIR_PREFIX}{chunk_idx}"
    out_db_path = os.path.join(DB_DIR, f"part_{chunk_idx}")
    
    # 1. Download PDBs
    os.makedirs(temp_dir, exist_ok=True)
    print("   ?뱿 Downloading PDBs...")
    with ThreadPoolExecutor(max_workers=30) as executor:
        futures = {executor.submit(download_single, pid, temp_dir): pid for pid in chunk_ids}
        for f in tqdm(as_completed(futures), total=len(chunk_ids), desc=f"Chunk {chunk_idx}"):
            pass
    
    # Check how many downloaded
    downloaded = len(glob.glob(os.path.join(temp_dir, "*.pdb")))
    print(f"   Downloaded: {downloaded}/{len(chunk_ids)}")
    
    if downloaded == 0:
        print(f"??No PDBs downloaded for chunk {chunk_idx}, skipping...")
        shutil.rmtree(temp_dir, ignore_errors=True)
        return False
    
    # 2. Remove old corrupt files
    for f in glob.glob(f"{out_db_path}*"):
        try: os.remove(f)
        except: pass
    
    # 3. Create DB
    print("   ?뾼截?Creating FoldSeek DB...")
    cmd = [FOLDSEEK_BIN, "createdb", temp_dir, out_db_path]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"??DB creation failed: {e.stderr.decode()}")
        shutil.rmtree(temp_dir, ignore_errors=True)
        return False
    
    # 4. Clean temp PDBs
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    # 5. Run Search
    print("   ?뵊 Running Search...")
    tsv_out = os.path.join(RES_DIR, f"result_{chunk_idx}.tsv")
    aln_db = os.path.join(RES_DIR, f"aln_{chunk_idx}")
    tmp_dir = os.path.join(RES_DIR, f"tmp_{chunk_idx}")
    
    # Clean previous
    shutil.rmtree(tmp_dir, ignore_errors=True)
    for f in glob.glob(f"{aln_db}*"):
        try: os.remove(f)
        except: pass
    if os.path.exists(tsv_out):
        os.remove(tsv_out)
    
    os.makedirs(tmp_dir, exist_ok=True)
    
    cmd_search = [FOLDSEEK_BIN, "search", out_db_path, TARGET_DB, aln_db, tmp_dir, "-a"]
    try:
        subprocess.run(cmd_search, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"??Search failed: {e.stderr.decode()[:500]}")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return False
    
    # 6. Convert to TSV
    fmt = "query,target,fident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits"
    cmd_conv = [FOLDSEEK_BIN, "convertalis", out_db_path, TARGET_DB, aln_db, tsv_out, "--format-output", fmt]
    try:
        subprocess.run(cmd_conv, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"??Convert failed: {e.stderr.decode()[:500]}")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return False
    
    # Cleanup
    shutil.rmtree(tmp_dir, ignore_errors=True)
    for f in glob.glob(f"{aln_db}*"):
        try: os.remove(f)
        except: pass
    
    print(f"??Chunk {chunk_idx} repaired successfully!")
    return True

def main():
    print("?? Starting Bulk Repair for Missing Chunks")
    print(f"   Chunks to repair: {MISSING_CHUNKS}")
    
    all_ids = get_all_ids_from_fasta(FASTA_FILE)
    print(f"   Total IDs in FASTA: {len(all_ids)}")
    
    os.makedirs(RES_DIR, exist_ok=True)
    
    success = 0
    failed = []
    
    for chunk_idx in MISSING_CHUNKS:
        if repair_chunk(chunk_idx, all_ids):
            success += 1
        else:
            failed.append(chunk_idx)
    
    print("\n" + "="*50)
    print("?럦 Bulk Repair Complete!")
    print(f"   Success: {success}/{len(MISSING_CHUNKS)}")
    if failed:
        print(f"   Failed: {failed}")
    print("="*50)

if __name__ == "__main__":
    main()

