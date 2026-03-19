import os
import glob
import subprocess
import shutil

def main():
    db_dir = "partial_dbs"
    target_db = "./artifacts/search_db/swissprot_db"
    res_dir = "search_results"
    
    os.makedirs(res_dir, exist_ok=True)
    
    # Get chunks sorted
    files = glob.glob(os.path.join(db_dir, "part_*.dbtype"))
    chunks = []
    for f in files:
        base = f.replace(".dbtype", "")
        # Extract index
        try:
            idx = int(os.path.basename(base).split('_')[1])
            chunks.append((idx, base))
        except:
            pass
            
    chunks.sort()
    
    print(f"?? found {len(chunks)} chunks to search.")
    
    foldseek_bin = "/root/miniconda3/bin/foldseek"
    
    for idx, base_path in chunks:
        tsv_out = os.path.join(res_dir, f"result_{idx}.tsv")
        
        if os.path.exists(tsv_out) and os.path.getsize(tsv_out) > 0:
            print(f"??Chunk {idx} already processed. Skipping.")
            continue
            
        print(f"?뵊 Processing Chunk {idx}...")
        
        aln_db = os.path.join(res_dir, f"aln_{idx}")
        tmp_dir = os.path.join(res_dir, f"tmp_{idx}")
        os.makedirs(tmp_dir, exist_ok=True)
        
        # 1. Search
        cmd_search = [foldseek_bin, "search", base_path, target_db, aln_db, tmp_dir, "-a"]
        try:
            subprocess.run(cmd_search, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"??Search failed for chunk {idx}")
            print(f"   Stdout: {e.stdout}")
            print(f"   Stderr: {e.stderr}")
            shutil.rmtree(tmp_dir, ignore_errors=True)
            continue
            
        # 2. Convert to TSV
        fmt = "query,target,fident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits"
        cmd_conv = [foldseek_bin, "convertalis", base_path, target_db, aln_db, tsv_out, "--format-output", fmt]
        
        try:
            subprocess.run(cmd_conv, check=True, capture_output=True, text=True)
            print(f"??Chunk {idx} Done.")
        except subprocess.CalledProcessError as e:
             print(f"??Convert failed for chunk {idx}")
             print(f"   Stdout: {e.stdout}")
             print(f"   Stderr: {e.stderr}")
        
        # Cleanup
        shutil.rmtree(tmp_dir, ignore_errors=True)
        # Optional: remove aln_db files to save space? 
        # aln_db is multiple files: aln_0, aln_0.dbtype...
        for f in glob.glob(f"{aln_db}*"):
            try: os.remove(f) 
            except: pass

    print("?럦 All chunks processed.")
    
    # Merge TSVs
    print("?뵕 Merging TSV files...")
    all_tsvs = glob.glob(os.path.join(res_dir, "result_*.tsv"))
    # Sort
    all_tsvs.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
    
    final_out = "./results/submission_foldseek_raw.tsv"
    with open(final_out, 'w') as fout:
        for tsv in all_tsvs:
            with open(tsv, 'r') as fin:
                shutil.copyfileobj(fin, fout)
                
    print(f"?뮶 Final merged file: {final_out}")

if __name__ == "__main__":
    main()

