import os
import shutil
import glob
import subprocess
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Configuration
FASTA_FILE = "./data/raw/test/testsuperset.fasta"
CHUNK_IDX = 3
CHUNK_SIZE = 2000
TEMP_DIR = f"temp_pdb_chunk_{CHUNK_IDX}"
DB_DIR = "partial_dbs"
RES_DIR = "search_results"
API_URL = "https://alphafold.ebi.ac.uk/api/prediction/{}"
FOLDSEEK_BIN = "/root/miniconda3/bin/foldseek"
TARGET_DB = "./artifacts/search_db/swissprot_db"

def get_ids_from_fasta(fasta_path):
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
    return list(set(ids))

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

def main():
    # 1. Get IDs for Chunk 3
    all_ids = get_ids_from_fasta(FASTA_FILE)
    start = CHUNK_IDX * CHUNK_SIZE
    end = start + CHUNK_SIZE
    chunk_ids = all_ids[start:end]
    
    print(f"?뵩 Repairing Chunk {CHUNK_IDX}")
    print(f"   Target IDs: {len(chunk_ids)} ({start} - {end})")

    # 2. Download
    os.makedirs(TEMP_DIR, exist_ok=True)
    print("   Downloading PDBs...")
    with ThreadPoolExecutor(max_workers=30) as executor:
        futures = {executor.submit(download_single, pid, TEMP_DIR): pid for pid in chunk_ids}
        for f in tqdm(as_completed(futures), total=len(chunk_ids)):
            pass
            
    # 3. Create DB
    out_db_path = os.path.join(DB_DIR, f"part_{CHUNK_IDX}")
    
    # Remove old corrupt files if exist
    for f in glob.glob(f"{out_db_path}*"):
        try: os.remove(f)
        except: pass
        
    print("   Creating FoldSeek DB...")
    cmd = [FOLDSEEK_BIN, "createdb", TEMP_DIR, out_db_path]
    subprocess.run(cmd, check=True)
    
    # 4. Search immediately
    print("   Running Search...")
    tsv_out = os.path.join(RES_DIR, f"result_{CHUNK_IDX}.tsv")
    aln_db = os.path.join(RES_DIR, f"aln_{CHUNK_IDX}")
    tmp_dir = os.path.join(RES_DIR, f"tmp_{CHUNK_IDX}")
    
    # Clean previous search tmp
    if os.path.exists(tmp_dir): shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)
    
    cmd_search = [FOLDSEEK_BIN, "search", out_db_path, TARGET_DB, aln_db, tmp_dir, "-a"]
    subprocess.run(cmd_search, check=True)
    
    fmt = "query,target,fident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits"
    cmd_conv = [FOLDSEEK_BIN, "convertalis", out_db_path, TARGET_DB, aln_db, tsv_out, "--format-output", fmt]
    subprocess.run(cmd_conv, check=True)
    
    print(f"??Chunk {CHUNK_IDX} Repaired and Searched!")
    
    # Cleanup
    shutil.rmtree(TEMP_DIR, ignore_errors=True)
    shutil.rmtree(tmp_dir, ignore_errors=True)

if __name__ == "__main__":
    main()

