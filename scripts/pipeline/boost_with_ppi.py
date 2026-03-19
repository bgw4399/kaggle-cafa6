import sys
import os
from tqdm import tqdm

FILE_BASE = "./results/temp_378_sorted.tsv"
FILE_PPI = "./results/temp_ppi_sorted.tsv"
OUTPUT_FILE = "./results/submission_PPI_Boost.tsv"

# Boost Config
BOOST_FACTOR = 0.2  # Increase by 20% of the PPI confidence

def read_line(f):
    line = f.readline()
    if not line: return None, None, None
    parts = line.strip().split('\t')
    if len(parts) < 3: return read_line(f)
    try:
        score = float(parts[2])
    except:
        return read_line(f)
        
    return (parts[0], parts[1]), score, line

def boost():
    print(f"?? Boosting Best_378 with PPI (Safe Mode)...")
    print(f"   Base: {FILE_BASE}")
    print(f"   PPI:  {FILE_PPI}")
    
    if not os.path.exists(FILE_BASE) or not os.path.exists(FILE_PPI):
        print("??Sorted temp files missing!")
        return

    with open(FILE_BASE, 'r') as f_base, open(FILE_PPI, 'r') as f_ppi, open(OUTPUT_FILE, 'w') as f_out:
        
        k_base, s_base, l_base = read_line(f_base)
        k_ppi, s_ppi, l_ppi = read_line(f_ppi)
        
        count = 0
        boosted = 0
        
        pbar = tqdm(total=150_000_000, desc="Boosting", unit="lines")
        
        while k_base is not None:
             # Case 1: PPI finished, just write rest of Base
            if k_ppi is None:
                f_out.write(l_base)
                k_base, s_base, l_base = read_line(f_base)
                
            # Case 2: Compare Keys
            elif k_base == k_ppi:
                # MATCH! Apply Boost
                # Final = Base * (1 + PPI * Factor)
                # If PPI=1.0, Base becomes Base * 1.2
                # If PPI=0.5, Base becomes Base * 1.1
                factor = 1.0 + (s_ppi * BOOST_FACTOR)
                new_score = s_base * factor
                if new_score > 1.0: new_score = 1.0
                
                f_out.write(f"{k_base[0]}\t{k_base[1]}\t{new_score:.5f}\n")
                boosted += 1
                
                k_base, s_base, l_base = read_line(f_base)
                k_ppi, s_ppi, l_ppi = read_line(f_ppi)
                
            elif k_base < k_ppi:
                # Base is behind, PPI doesn't have this term.
                # Just write Base as is.
                f_out.write(l_base)
                k_base, s_base, l_base = read_line(f_base)
                
            else: # k_base > k_ppi
                # PPI is behind (Term exists in PPI but not Base)
                # IGNORE PPI (Safe Mode)
                # We don't want to add new predictions that base model rejected.
                k_ppi, s_ppi, l_ppi = read_line(f_ppi)
            
            count += 1
            if count % 10000 == 0: pbar.update(10000)

        pbar.close()
        
    print(f"?럦 Boost Complete!")
    print(f"   Boosted Terms: {boosted:,}")
    print(f"   Saved: {OUTPUT_FILE}")

if __name__ == "__main__":
    boost()

