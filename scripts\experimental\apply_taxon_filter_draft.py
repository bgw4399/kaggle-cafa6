import pandas as pd
import numpy as np
import collections
import os
from tqdm import tqdm

# Config
PRED_FILE = "./results/final_submission/submission_Final_Repaired.tsv" # Input (0.31 Model)
OUTPUT_FILE = "./results/final_submission/submission_Final_Taxon_Filtered_Fast.tsv"
TRAIN_TAXON = "./data/raw/train/train_taxonomy.tsv"
TRAIN_TERMS = "./data/raw/train/train_terms.tsv"
TEST_TAXON = "./data/raw/test/testsuperset-taxon-list.tsv"

PENALTY_FACTOR = 0.1 # Severe penalty for taxon violation
RESTORE_THRESHOLD = 0.9 # If Deep Learning is 90% sure, we assume it might know better (Annotation Error?)

def main():
    print("?뙼 Taxonomy-Aware Filtering (SOTA Trend)...")
    
    # 1. Load Train Taxonomy (ID -> Species)
    print("   ?뱿 Loading Train Taxonomy...")
    train_id_to_species = {}
    try:
        # Check format
        df_tax = pd.read_csv(TRAIN_TAXON, sep='\t')
        # Columns: EntryID, TaxonomyID
        for pid, tax in zip(df_tax['EntryID'], df_tax['TaxonomyID']):
            train_id_to_species[str(pid)] = int(tax)
    except:
        # Maybe headerless?
        df_tax = pd.read_csv(TRAIN_TAXON, sep='\t', header=None, names=['EntryID', 'TaxonomyID'])
        for pid, tax in zip(df_tax['EntryID'], df_tax['TaxonomyID']):
            train_id_to_species[str(pid)] = int(tax)
            
    print(f"      Mapped {len(train_id_to_species):,} training proteins.")

    # 2. Build Allowed Terms per Species
    print("   ?뱴 Building Knowledge Base (Species -> Terms)...")
    # Species -> Set(Terms)
    species_knowledge = collections.defaultdict(set)
    
    df_terms = pd.read_csv(TRAIN_TERMS, sep='\t') 
    # Columns: EntryID, term
    for pid, term in zip(df_terms['EntryID'], df_terms['term']):
        pid = str(pid)
        if pid in train_id_to_species:
            sp = train_id_to_species[pid]
            species_knowledge[sp].add(term)
            
    print(f"      Knowledge captured for {len(species_knowledge)} species.")
    
    # 3. Load Test Taxonomy
    print("   ?뱿 Loading Test Taxonomy...")
    test_id_to_species = {}
    df_test_tax = pd.read_csv(TEST_TAXON, sep='\t')
    # Columns: ID, Species (ID might be int or string)
    for pid, sp in zip(df_test_tax['ID'], df_test_tax['Species']):
        # Clean ID just in case
        pid = str(pid).strip()
        # Clean Species (might be "9606" or "Homo Sapiens")
        # Actually file usually has ID [tab] Species Name or ID
        # Wait, I viewed the file earlier. Column 2 is "Species Name" or ID?
        # Let's check format again or robustly handle it. 
        # Actually I saw: "9606\tHomo sapiens". Wait, ID is column 1?
        # File view showed:
        # ID	Species
        # 9606	Homo sapiens 
        # Wait, NO. 9606 IS THE SPECIES ID. Column 1 is ID?
        # Let's re-read the view_file output from Step 3889.
        # "ID	Species"
        # "9606	Homo sapiens"
        # It seems Column 1 is Taxon ID?? 
        # NO. "ID" usually means Protein ID. But 9606 is Human Taxon ID. 
        # Maybe the file lists TARGET ORGANISMS?
        # "10116 Rattus norvegicus"
        # This file might be a LIST OF SPECIES, not a mapping of Protein->Species.
        # IF so, where is the per-protein mapping?
        # "./data/raw/test/testsuperset-taxon-list.tsv" name implies "Taxon List" of the superset?
        # Ah. I need to map Test PROTEIN -> SPECIES.
        # If this file is just a list of species, it's useless for per-protein filtering.
        # I need to find the mapping.
        # "testsuperset-taxon-list.tsv" might be "Taxon ID -> Species Name".
        pass 
        
    # Correcting Logic based on file inspection suspicion
    # Actually, usually CAFA provides a mapping file.
    # If I don't have Protein -> Taxon map for Test, I cannot filter.
    # Let's look for `data/embeddings/testsuperset_ids.npy` / `test_species_idx.npy` in previous script.
    # User doesn't have `data/embeddings` folder (Step 3643 failed).
    # IS THERE A FILE MAPPING TEST PROTEINS TO TAXA?
    # I will search for it.
    
    return # Abort for now until I confirm mapping file.

if __name__ == "__main__":
    main()


