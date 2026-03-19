import sys

def main():
    prev_key = None
    max_score = -1.0
    
    # Read from stdin
    for line in sys.stdin:
        parts = line.strip().split('\t')
        if len(parts) < 3:
            continue
            
        pid, term = parts[0], parts[1]
        try:
            score = float(parts[2])
        except ValueError:
            continue
            
        key = (pid, term)
        
        if key != prev_key:
            # New key, print previous max if it exists
            if prev_key is not None:
                # Use format matching the input precision or standard
                print(f"{prev_key[0]}\t{prev_key[1]}\t{max_score:.5f}")
            
            # Reset
            prev_key = key
            max_score = score
        else:
            # Same key, update max
            if score > max_score:
                max_score = score

    # Print last one
    if prev_key is not None:
        print(f"{prev_key[0]}\t{prev_key[1]}\t{max_score:.5f}")

if __name__ == "__main__":
    main()
