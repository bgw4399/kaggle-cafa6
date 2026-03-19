import sys
import os
from tqdm import tqdm

FILE_BASE = "./results/temp_378_sorted.tsv"
FILE_INJECT = "./results/temp_foldseek_sorted.tsv"
OUTPUT_FILE = "./results/submission_Foldseek_Injection.tsv"

def read_line(f):
    line = f.readline()
    if not line: return None, None, None
    parts = line.strip().split('\t')
    if len(parts) < 3: return read_line(f)
    return (parts[0], parts[1]), float(parts[2]), line

def inject():
    print(f"?? Injecting Foldseek into Best_378 (Low RAM)...")
    print(f"   Base: {FILE_BASE}")
    print(f"   Inject: {FILE_INJECT}")
    
    if not os.path.exists(FILE_BASE) or not os.path.exists(FILE_INJECT):
        print("??Sorted temp files missing!")
        return

    with open(FILE_BASE, 'r') as f_base, open(FILE_INJECT, 'r') as f_inj, open(OUTPUT_FILE, 'w') as f_out:
        
        # Initial read
        k_base, s_base, l_base = read_line(f_base)
        k_inj, s_inj, l_inj = read_line(f_inj)
        
        count = 0
        injected = 0
        
        pbar = tqdm(total=200_000_000, desc="Merging", unit="lines") # Estimate
        
        while k_base is not None or k_inj is not None:
            # Case 1: Base finished, flush Inject
            if k_base is None:
                if s_inj > 0.001:
                    f_out.write(f"{k_inj[0]}\t{k_inj[1]}\t{s_inj:.5f}\n")
                    injected += 1
                k_inj, s_inj, l_inj = read_line(f_inj)
                
            # Case 2: Inject finished, flush Base
            elif k_inj is None:
                f_out.write(l_base)
                k_base, s_base, l_base = read_line(f_base)
                
            # Case 3: Compare Keys
            elif k_base == k_inj:
                # MATCH! Take MAX
                final_score = max(s_base, s_inj)
                if final_score > 1.0: final_score = 1.0
                f_out.write(f"{k_base[0]}\t{k_base[1]}\t{final_score:.5f}\n")
                if s_inj > s_base: injected += 1
                
                k_base, s_base, l_base = read_line(f_base)
                k_inj, s_inj, l_inj = read_line(f_inj)
                
            elif k_base < k_inj:
                # Base is behind, write Base (Check score)
                if s_base > 1.0: s_base = 1.0
                if s_base > 0.001:
                    f_out.write(f"{k_base[0]}\t{k_base[1]}\t{s_base:.5f}\n")
                k_base, s_base, l_base = read_line(f_base)
                
            else: # k_base > k_inj
                # Inject is behind (new term!), write Inject
                if s_inj > 1.0: s_inj = 1.0
                if s_inj > 0.001:
                    f_out.write(f"{k_inj[0]}\t{k_inj[1]}\t{s_inj:.5f}\n")
                    injected += 1
                k_inj, s_inj, l_inj = read_line(f_inj)
            
            count += 1
            if count % 10000 == 0: pbar.update(10000)

        pbar.close()
        
    print(f"?럦 Injection Complete!")
    print(f"   Injected/Boosted: {injected:,} terms")
    print(f"   Saved: {OUTPUT_FILE}")

if __name__ == "__main__":
    inject()

