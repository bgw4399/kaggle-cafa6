import sys
import pandas as pd
import numpy as np

def validate_file(path):
    print(f"?뵇 Validating: {path}")
    try:
        chunkval = pd.read_csv(path, sep='\t', header=None, names=['Id', 'Term', 'Score'], chunksize=100000)
        
        min_score = 1.0
        max_score = 0.0
        total_rows = 0
        issues = []
        
        for i, df in enumerate(chunkval):
            # Check for NaNs
            if df.isnull().any().any():
                issues.append(f"Chunk {i}: Contains NaNs")
            
            # Check Range
            curr_min = df['Score'].min()
            curr_max = df['Score'].max()
            min_score = min(min_score, curr_min)
            max_score = max(max_score, curr_max)
            
            # Check Types (first chunk only is enough usually, but check)
            if i == 0:
                if not pd.api.types.is_string_dtype(df['Id']): issues.append("Column 'Id' is not string.")
                if not pd.api.types.is_string_dtype(df['Term']): issues.append("Column 'Term' is not string.")
                if not pd.api.types.is_numeric_dtype(df['Score']): issues.append("Column 'Score' is not numeric.")
                
                # Check Term format on first few
                invalid_terms = df[~df['Term'].astype(str).str.startswith('GO:')].head()
                if not invalid_terms.empty:
                    issues.append(f"Invalid Term format: {invalid_terms['Term'].values}")

            total_rows += len(df)
            
        if min_score < 0 or max_score > 1:
            issues.append(f"Score out of range [0, 1]: min={min_score}, max={max_score}")

        if issues:
            for issue in set(issues): # Dedup issues
                print(f"?좑툘 {issue}")
            print("??Validation Failed.")
        else:
            print(f"??Validation Passed. Rows: {total_rows:,}")
            print(f"   Score Range: {min_score:.5f} - {max_score:.5f}")

    except pd.errors.EmptyDataError:
        print("??Error: File is empty.")
    except Exception as e:
        print(f"??Unexpected Error: {e}")

if __name__ == "__main__":
    files = [
        "./results/Final_Submission_SOTA_ProtT5_Merge_Top50.tsv",
        "./results/Final_Submission_SoftESM_Diamond_Merge_Top50.tsv"
    ]
    for f in files:
        validate_file(f)
        print("-" * 40)

