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
DB_DIR = "partial_dbs"
RES_DIR = "search_results"
API_URL = "https://alphafold.ebi.ac.uk/api/prediction/{}"
FOLDSEEK_BIN = "/root/miniconda3/bin/foldseek"
TARGET_DB = "./artifacts/search_db/swissprot_db_pad"  # Use padded DB for GPU

# Missing DB chunks
MISSING_CHUNKS = [30, 31, 32, 33, 34, 35, 36, 37, 38, 39]

def get_all_ids_from_fasta(fasta_path):
    ids = []
    with open(fasta_path, "r") as f:
        for line in f:
            if line.startswith(">"):
                parts = line.strip().split()
                header = parts[0].replace(">", "")
                protein_id = header.split(":")[1] if ":" in header else header
                ids.append(protein_id)
    return list(dict.fromkeys(ids))

def download_single(pid, save_dir):
    save_path = os.path.join(save_dir, f"{pid}.pdb")
    if os.path.exists(save_path):
        return True
    try:
        res = requests.get(API_URL.format(pid), timeout=8)
        if res.status_code != 200: return False
        data = res.json()
        if not data: return False
        pdb_url = data[0].get('pdbUrl')
        if not pdb_url: return False
        pdb_res = requests.get(pdb_url, timeout=10)
        if pdb_res.status_code == 200:
            with open(save_path, "wb") as f:
                f.write(pdb_res.content)
            return True
    except: pass
    return False

def repair_chunk(chunk_idx, all_ids):
    print(f"\n?뵩 Repairing Chunk {chunk_idx}...")
    
    start = chunk_idx * CHUNK_SIZE
    end = start + CHUNK_SIZE
    chunk_ids = all_ids[start:end]
    
    if not chunk_ids:
        print(f"?좑툘 No IDs for chunk {chunk_idx}")
        return False
    
    temp_dir = f"temp_repair_{chunk_idx}"
    out_db_path = os.path.join(DB_DIR, f"part_{chunk_idx}")
    
    # Download
    os.makedirs(temp_dir, exist_ok=True)
    print(f"   ?뱿 Downloading {len(chunk_ids)} PDBs...")
    with ThreadPoolExecutor(max_workers=50) as executor:
        futures = {executor.submit(download_single, pid, temp_dir): pid for pid in chunk_ids}
        for f in tqdm(as_completed(futures), total=len(chunk_ids), leave=False):
            pass
    
    downloaded = len(glob.glob(os.path.join(temp_dir, "*.pdb")))
    if downloaded == 0:
        shutil.rmtree(temp_dir, ignore_errors=True)
        return False
    
    # Create DB
    for f in glob.glob(f"{out_db_path}*"):
        try: os.remove(f)
        except: pass
    
    print(f"   ?뾼截?Creating DB ({downloaded} PDBs)...")
    cmd = [FOLDSEEK_BIN, "createdb", temp_dir, out_db_path]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except:
        shutil.rmtree(temp_dir, ignore_errors=True)
        return False
    
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    # Search with GPU
    print(f"   ?뵊 Searching with GPU...")
    tsv_out = os.path.join(RES_DIR, f"result_{chunk_idx}.tsv")
    aln_db = os.path.join(RES_DIR, f"aln_{chunk_idx}")
    tmp_dir = os.path.join(RES_DIR, f"tmp_{chunk_idx}")
    
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)
    
    cmd_search = [FOLDSEEK_BIN, "search", out_db_path, TARGET_DB, aln_db, tmp_dir, "-a", "--gpu", "1", "-s", "9.5"]
    try:
        subprocess.run(cmd_search, check=True, capture_output=True)
    except:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return False
    
    fmt = "query,target,fident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits"
    cmd_conv = [FOLDSEEK_BIN, "convertalis", out_db_path, TARGET_DB, aln_db, tsv_out, "--format-output", fmt]
    try:
        subprocess.run(cmd_conv, check=True, capture_output=True)
    except:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return False
    
    shutil.rmtree(tmp_dir, ignore_errors=True)
    for f in glob.glob(f"{aln_db}*"):
        try: os.remove(f)
        except: pass
    
    print(f"??Chunk {chunk_idx} repaired!")
    return True

def main():
    print("?뵩 Repairing Missing DB Chunks: ", MISSING_CHUNKS)
    all_ids = get_all_ids_from_fasta(FASTA_FILE)
    os.makedirs(RES_DIR, exist_ok=True)
    
    success = 0
    for idx in MISSING_CHUNKS:
        if repair_chunk(idx, all_ids):
            success += 1
    
    print(f"\n?럦 Repair done! {success}/{len(MISSING_CHUNKS)}")

if __name__ == "__main__":
    main()

