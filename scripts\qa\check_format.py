import sys

FILE = "./results/submission_Foldseek_Injection.tsv"

def check():
    print(f"?빑截?Checking format of {FILE}...")
    try:
        with open(FILE, 'r') as f:
            # Check first line
            head = f.readline().strip()
            parts = head.split('\t')
            if len(parts) != 3:
                print(f"??Error: Line 1 has {len(parts)} columns (Expected 3). Content: {head}")
                return
            
            # Check header
            try:
                float(parts[2])
                print("??Top line looks like data (No header found).")
            except ValueError:
                print(f"??Error: Top line looks like a header! Col 3 is not float: {parts[2]}")
                return
                
            # Scan a few more
            for i in range(100):
                line = f.readline()
                if not line: break
                p = line.strip().split('\t')
                if len(p) != 3:
                     print(f"??Error: Line {i+2} malformed.")
                     return
                s = float(p[2])
                if s < 0 or s > 1.0:
                    print(f"??Error: Score out of range in line {i+2}: {s}")
                    return

        print("??Format Check Passed (Sample).")
        
    except FileNotFoundError:
        print("??File not found.")

if __name__ == "__main__":
    check()

