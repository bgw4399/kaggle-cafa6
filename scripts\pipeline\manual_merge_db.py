import os
import glob
import subprocess

def merge():
    db_dir = "partial_dbs"
    final_db = "test_query_db"
    
    # Get all partial DB base names (files without extension inside db_dir)
    # FoldSeek DBs usually have a base file + .dbtype, .index etc.
    # We want just the base filename.
    # Globbing "part_*" could get part_0, part_0.dbtype, etc.
    # We filter for those that don't have an extension (or check for .dbtype and strip it)
    
    files = glob.glob(os.path.join(db_dir, "part_*"))
    # Filter: keep only those that have a corresponding .dbtype file
    dbs = []
    for f in files:
        if f.endswith(".dbtype"):
            continue
        if f.endswith(".index"):
            continue
        if f.endswith(".lookup"):
            continue
        if f.endswith(".source"):
            continue
        if f.endswith("_h"):
            continue
        if f.endswith("_ca"):
            continue
        if f.endswith("_ss"):
            continue
            
        # Check if it has a .dbtype sibling
        if os.path.exists(f + ".dbtype"):
            dbs.append(f)
            
    # Sort carefully: part_0, part_1... part_10...
    # split by '_' and sort by int
    def sort_key(x):
        try:
            return int(os.path.basename(x).split('_')[1])
        except:
            return 999999
            
    dbs.sort(key=sort_key)
    
    print(f"Found {len(dbs)} partial databases.")
    if not dbs:
        print("No databases found!")
        return

    # Create merge command
    cmd = ["/root/miniconda3/bin/foldseek", "mergedbs", final_db] + dbs
    
    print(f"Executing: {' '.join(cmd[:10])} ... ({len(cmd)} args)")
    
    try:
        subprocess.run(cmd, check=True)
        print("✅ Merge successful!")
    except Exception as e:
        print(f"❌ Merge failed: {e}")

if __name__ == "__main__":
    merge()
