import os
import glob
from tqdm import tqdm

# Config
FILE_BASE = "./results/final_submission/submission_Final_Repaired.tsv"
FILE_STACKING = "./results/final_submission/submission_Stacking_XGB.tsv"
OUTPUT_FILE = "./results/final_submission/submission_Hybrid_Rescue.tsv"
NUM_BUCKETS = 20 # 20 buckets should be enough (4GB / 20 = 200MB per bucket)
TEMP_DIR = "./temp_buckets"

def cleanup():
    if os.path.exists(TEMP_DIR):
        import shutil
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR, exist_ok=True)

def get_bucket(pid):
    # Simple hash based on last char or hash
    return hash(pid) % NUM_BUCKETS

def split_files():
    print("✂️ Splitting files into buckets...")
    
    # Open bucket handles
    base_handles = {i: open(f"{TEMP_DIR}/base_{i}.tsv", 'w') for i in range(NUM_BUCKETS)}
    stack_handles = {i: open(f"{TEMP_DIR}/stack_{i}.tsv", 'w') for i in range(NUM_BUCKETS)}
    
    # Split Base (4GB)
    print("   Streaming Base File...")
    with open(FILE_BASE, 'r') as f:
        for line in tqdm(f, desc="Split Base"):
            parts = line.strip().split('\t')
            if len(parts) < 2: continue
            pid = parts[0]
            b = get_bucket(pid)
            base_handles[b].write(line)
            
    # Split Stacking (800MB)
    print("   Streaming Stacking File...")
    with open(FILE_STACKING, 'r') as f:
        for line in tqdm(f, desc="Split Stack"):
            parts = line.strip().split('\t')
            if len(parts) < 2: continue
            pid = parts[0]
            b = get_bucket(pid)
            stack_handles[b].write(line)
            
    # Close
    for h in base_handles.values(): h.close()
    for h in stack_handles.values(): h.close()
    
    print("✅ Split complete.")

def process_buckets():
    print("🔗 Merging buckets...")
    
    f_out = open(OUTPUT_FILE, 'w')
    
    total_preds = 0
    
    for i in range(NUM_BUCKETS):
        b_base = f"{TEMP_DIR}/base_{i}.tsv"
        b_stack = f"{TEMP_DIR}/stack_{i}.tsv"
        
        # Load Stacking Bucket into Dict
        stack_scores = {}
        if os.path.exists(b_stack):
            with open(b_stack, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) < 3: continue
                    # Key: (pid, term)
                    stack_scores[(parts[0], parts[1])] = parts[2]
        
        # Stream Base Bucket and Write
        if os.path.exists(b_base):
            with open(b_base, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) < 3: continue
                    pid, term = parts[0], parts[1]
                    
                    if (pid, term) in stack_scores:
                        # Override
                        score = stack_scores[(pid, term)]
                        f_out.write(f"{pid}\t{term}\t{score}\n")
                        # Mark used? No need, we inject later
                        del stack_scores[(pid, term)]
                    else:
                        # Keep Base
                        f_out.write(line)
                        
                    total_preds += 1
        
        # Inject Remaining Stacking
        for (pid, term), score in stack_scores.items():
            f_out.write(f"{pid}\t{term}\t{score}\n")
            total_preds += 1
            
        # Clean bucket files immediately to save space
        if os.path.exists(b_base): os.remove(b_base)
        if os.path.exists(b_stack): os.remove(b_stack)
        
        print(f"   ☑️ Bucket {i+1}/{NUM_BUCKETS} done.")
        
    f_out.close()
    print(f"✅ Rescue Complete. Total: {total_preds:,}")

if __name__ == "__main__":
    cleanup()
    split_files()
    process_buckets()
    # cleanup() # Optional
