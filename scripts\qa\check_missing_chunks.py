import os
import glob

db_dir = "partial_dbs"
missing = []

for i in range(113):
    dbtype_file = os.path.join(db_dir, f"part_{i}.dbtype")
    if not os.path.exists(dbtype_file):
        missing.append(i)

if missing:
    print(f"❌ Missing main DB files for chunks: {missing}")
    print(f"   Total: {len(missing)} chunks need repair")
else:
    print("✅ All chunks have main DB files!")
