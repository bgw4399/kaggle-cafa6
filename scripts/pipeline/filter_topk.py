import sys
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', help='Input sorted TSV file')
    parser.add_argument('output_file', help='Output TSV file')
    parser.add_argument('--k', type=int, default=50, help='Top K to keep')
    args = parser.parse_args()

    print(f"✂️ Filtering Top-{args.k} from {args.input_file} -> {args.output_file} (Streaming)")

    current_pid = None
    buffer = []

    try:
        with open(args.input_file, 'r') as fin, open(args.output_file, 'w') as fout:
            for line in fin:
                parts = line.strip().split('\t')
                if len(parts) < 3: continue
                
                pid = parts[0]
                try:
                    score = float(parts[2])
                except ValueError: continue

                if pid != current_pid:
                    # Flush previous buffer
                    if buffer:
                        # Sort by score descending and keep top K
                        # Since we want stability and speed, we sort locally
                        buffer.sort(key=lambda x: x[1], reverse=True)
                        top_k = buffer[:args.k]
                        for item in top_k:
                            # reconstruct line or just write formatted
                            fout.write(f"{current_pid}\t{item[0]}\t{item[1]:.5f}\n")
                    
                    current_pid = pid
                    buffer = []

                # Add to buffer
                buffer.append((parts[1], score))

            # Flush last buffer
            if buffer:
                buffer.sort(key=lambda x: x[1], reverse=True)
                top_k = buffer[:args.k]
                for item in top_k:
                    fout.write(f"{current_pid}\t{item[0]}\t{item[1]:.5f}\n")

        print("✅ Done.")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
