import os
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# =========================================================
# ?숋툘 [?ㅼ젙] API 湲곕컲 ?ㅼ슫濡쒕뱶 (媛???뺤떎??
# =========================================================
FASTA_FILE = "./data/raw/test/testsuperset.fasta"
OUTPUT_DIR = "test_pdb_dir"
FAIL_LOG_FILE = "failed_api_downloads.txt"

# AlphaFold API 二쇱냼
API_URL = "https://alphafold.ebi.ac.uk/api/prediction/{}"
MAX_WORKERS = 30  # ?숈떆 ?묒냽 ??(API??媛踰쇱썙??醫 ???믪뿬????
# =========================================================

def get_ids_from_fasta(fasta_path):
    ids = []
    print(f"?뱰 {fasta_path} ID 異붿텧 以?..")
    with open(fasta_path, "r") as f:
        for line in f:
            if line.startswith(">"):
                # >A0A0C5B5G6 9606 ?뺥깭 泥섎━
                parts = line.strip().split() # 怨듬갚?쇰줈 履쇨갔
                header = parts[0].replace(">", "")
                
                # TaxID:UniProtID ?뺥깭 ???
                if ":" in header:
                    protein_id = header.split(":")[1]
                else:
                    protein_id = header
                ids.append(protein_id)
    return list(set(ids))

def download_via_api(pid):
    """API瑜??듯빐 ?뺥솗???ㅼ슫濡쒕뱶 留곹겕瑜??살뼱?????""
    save_path = os.path.join(OUTPUT_DIR, f"{pid}.pdb")
    
    if os.path.exists(save_path):
        return ("skipped", pid, "Already exists")

    try:
        # 1. API??臾쇱뼱蹂닿린 ("?뚯씪 ?대뵪??")
        response = requests.get(API_URL.format(pid), timeout=10)
        
        if response.status_code != 200:
            return ("failed", pid, f"API Error {response.status_code}")
            
        data = response.json()
        if not data or not isinstance(data, list):
            return ("failed", pid, "No Data in AFDB")
            
        # 2. ?ㅼ슫濡쒕뱶 留곹겕 異붿텧 (媛??理쒖떊 踰꾩쟾 媛?몄샂)
        # data[0]['pdbUrl']??二쇱냼媛 ?ㅼ뼱?덉쓬
        download_url = data[0].get('pdbUrl')
        if not download_url:
            return ("failed", pid, "No PDB URL")
            
        # 3. 吏꾩쭨 ?ㅼ슫濡쒕뱶
        pdb_res = requests.get(download_url, timeout=15)
        if pdb_res.status_code == 200:
            with open(save_path, "wb") as f:
                f.write(pdb_res.content)
            return ("success", pid, "OK")
        else:
            return ("failed", pid, "Download Failed")

    except Exception as e:
        return ("error", pid, str(e))

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    target_ids = get_ids_from_fasta(FASTA_FILE)
    print(f"?? 珥?{len(target_ids):,}媛?API 湲곕컲 ?ㅼ슫濡쒕뱶 ?쒖옉...")

    failed_list = [] 
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(download_via_api, pid): pid for pid in target_ids}
        
        pbar = tqdm(as_completed(futures), total=len(target_ids), unit="pdb")
        success_cnt = 0
        
        for future in pbar:
            status, pid, msg = future.result()
            
            if status == "success":
                success_cnt += 1
            elif status != "skipped":
                failed_list.append(f"{pid}\t{msg}")
                # ?먮윭 硫붿떆吏 ?ㅼ떆媛??뺤씤
                pbar.set_postfix_str(f"Last: {pid} {msg[:15]}..")
    
    print("\n" + "="*40)
    print(f"??理쒖쥌 ?꾨즺!")
    print(f"   - ?깃났: {success_cnt:,}媛?)
    print(f"   - ?ㅽ뙣: {len(failed_list):,}媛?(DB???놁쓬)")
    print("="*40)
    
    if failed_list:
        with open(FAIL_LOG_FILE, "w") as f:
            for line in failed_list:
                f.write(line + "\n")

if __name__ == "__main__":
    main()
