import pandas as pd
import os
import shutil

# Paths
base_dir = "e:/vscode/data analytics project"
large_csv_path = os.path.join(base_dir, "archivexx", "postings.csv")
ibm_csv_src = os.path.join(base_dir, "archivex", "WA_Fn-UseC_-HR-Employee-Attrition.csv")

dest_dir = os.path.join(base_dir, "filess", "hr-attrition-analytics", "hr-attrition-analytics", "data", "raw")
os.makedirs(dest_dir, exist_ok=True)

ibm_csv_dest = os.path.join(dest_dir, "ibm_hr_attrition.csv")
linkedin_csv_dest = os.path.join(dest_dir, "linkedin_jobs.csv")
linkedin_xlsx_dest = os.path.join(base_dir, "small_linkedin_jobs.xlsx")

# Copy IBM HR data
shutil.copy2(ibm_csv_src, ibm_csv_dest)
print(f"Copied IBM HR data to {ibm_csv_dest}")

# Columns to keep
cols_to_keep = [
    'title', 'max_salary', 'med_salary', 'min_salary', 'applies',
    'remote_allowed', 'work_type', 'formatted_experience_level', 'formatted_work_type'
]

# Read large CSV in chunks
chunk_size = 100000
chunks = []

for chunk in pd.read_csv(large_csv_path, usecols=cols_to_keep, chunksize=chunk_size):
    # Filter rows where med_salary is not null
    chunk_filtered = chunk[chunk['med_salary'].notnull()]
    chunks.append(chunk_filtered)

# Concatenate all filtered chunks
final_df = pd.concat(chunks, ignore_index=True)

# Save to CSV and XLSX
final_df.to_csv(linkedin_csv_dest, index=False)
final_df.to_excel(linkedin_xlsx_dest, index=False)

print(f"Processed LinkedIn data. Total rows: {len(final_df)}")
print(f"Saved to {linkedin_csv_dest} and {linkedin_xlsx_dest}")
