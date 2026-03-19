import os
import shutil
import glob
import subprocess
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time

# =========================================================
# ?숋툘 Configuration
# =========================================================
FASTA_FILE = "./data/raw/test/testsuperset.fasta"
TEMP_DIR = "temp_pdb_chunk"  # Temporary folder for PDBs
DB_DIR = "partial_dbs"       # Folder for partial DBs
FINAL_DB = "test_query_db"   # Final merged DB name

CHUNK_SIZE = 2000            # Number of proteins to process at once
MAX_WORKERS = 30             # Download concurrency
# =========================================================

API_URL = "https://alphafold.ebi.ac.uk/api/prediction/{}"

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
        return True # Already exists

    try:
        # 1. Get metadata
        res = requests.get(API_URL.format(pid), timeout=10)
        if res.status_code != 200:
            return False
        data = res.json()
        if not data or not isinstance(data, list):
            return False
        
        # 2. Get PDB URL
        pdb_url = data[0].get('pdbUrl')
        if not pdb_url:
            return False

        # 3. Download
        pdb_res = requests.get(pdb_url, timeout=15)
        if pdb_res.status_code == 200:
            with open(save_path, "wb") as f:
                f.write(pdb_res.content)
            return True
    except:
        pass
    return False

def process_chunk(chunk_ids, chunk_idx):
    # 1. Create temp dir
    current_temp_dir = f"{TEMP_DIR}_{chunk_idx}"
    os.makedirs(current_temp_dir, exist_ok=True)
    
    # 2. Download PDBs
    print(f"\n[Chunk {chunk_idx}] Downloading {len(chunk_ids)} PDBs...")
    downloaded_count = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(download_single, pid, current_temp_dir): pid for pid in chunk_ids}
        for f in tqdm(as_completed(futures), total=len(chunk_ids), leave=False):
            if f.result():
                downloaded_count += 1
    
    if downloaded_count == 0:
        print(f"?좑툘 [Chunk {chunk_idx}] No PDBs downloaded. Skipping DB creation.")
        shutil.rmtree(current_temp_dir)
        return None

    # 3. Create Partial FoldSeek DB
    os.makedirs(DB_DIR, exist_ok=True)
    out_db_path = os.path.join(DB_DIR, f"part_{chunk_idx}")
    
    print(f"[Chunk {chunk_idx}] Creating FoldSeek DB with {downloaded_count} structures...")
    
    # foldseek createdb <input_dir> <output_db>
    cmd = ["foldseek", "createdb", current_temp_dir, out_db_path]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        success = True
    except subprocess.CalledProcessError:
        print(f"??[Chunk {chunk_idx}] FoldSeek createdb Failed!")
        success = False

    # 4. Delete Temp PDBs (Crucial for Disk Space)
    print(f"[Chunk {chunk_idx}] Cleaning up temp files...")
    shutil.rmtree(current_temp_dir)
    
    return out_db_path if success else None

def merge_dbs(db_paths):
    print("\n?뵕 Merging all partial databases...")
    # foldseek mergedbs <final_db> <input1> <input2> ...
    if not db_paths:
        print("??No databases to merge.")
        return

    cmd = ["foldseek", "mergedbs", FINAL_DB] + db_paths
    try:
        subprocess.run(cmd, check=True)
        print(f"??Final DB created: {FINAL_DB}")
        
        # Cleanup partial DBs
        print("?㏏ Removing partial DB files...")
        shutil.rmtree(DB_DIR)
        
    except subprocess.CalledProcessError as e:
        print(f"??Merge failed: {e}")

def main():
    # Check FoldSeek
    if shutil.which("foldseek") is None:
        print("??Error: 'foldseek' command not found. Please install it or check PATH.")
        return

    all_ids = get_ids_from_fasta(FASTA_FILE)
    total_ids = len(all_ids)
    print(f"?렞 Total Targets: {total_ids:,}")

    # Chunking
    chunks = [all_ids[i:i + CHUNK_SIZE] for i in range(0, total_ids, CHUNK_SIZE)]
    print(f"?벀 Total Chunks: {len(chunks)}")

    created_dbs = []

    for i, chunk in enumerate(chunks):
        # Resume Logic
        expected_db_path = os.path.join(DB_DIR, f"part_{i}")
        if os.path.exists(expected_db_path) and os.path.exists(expected_db_path + ".dbtype"):
            print(f"??[Chunk {i}] Already exists. Skipping.")
            created_dbs.append(expected_db_path)
            continue

        db_path = process_chunk(chunk, i)
        if db_path:
            created_dbs.append(db_path)
            
        # Optional: Print disk usage
        # os.system("df -h .")

    # Final Merge
    if created_dbs:
        merge_dbs(created_dbs)
    else:
        print("??valid databases created.")

if __name__ == "__main__":
    main()

