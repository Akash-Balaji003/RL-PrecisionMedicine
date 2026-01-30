import pandas as pd
from datetime import datetime

# ================================
# 1. Load raw MIMIC-III tables
# ================================

BASE_PATH = "model/dataset/mimic-iii-demo/"

patients = pd.read_csv(BASE_PATH + "PATIENTS.csv")
diagnoses = pd.read_csv(BASE_PATH + "DIAGNOSES_ICD.csv")
icd_map = pd.read_csv(BASE_PATH + "D_ICD_DIAGNOSES.csv")
prescriptions = pd.read_csv(BASE_PATH + "PRESCRIPTIONS.csv")

print("Loaded raw tables")

# ================================
# 2. Basic patient demographics
# ================================

# Convert DOB and DOD to datetime
patients["dob"] = pd.to_datetime(patients["dob"])
patients["dod"] = pd.to_datetime(patients["dod"], errors="coerce")

# Use first admission proxy age (simplified)
REFERENCE_YEAR = 2100
patients["age"] = REFERENCE_YEAR - patients["dob"].dt.year

patients_demo = patients[[
    "subject_id",
    "gender",
    "age"
]]

print("Processed demographics")

# ================================
# 3. Map ICD codes → diagnosis text
# ================================

diagnoses = diagnoses.merge(
    icd_map[["icd9_code", "long_title"]],
    on="icd9_code",
    how="left"
)

# Sort to get primary diagnosis (seq_num = 1)
primary_dx = (
    diagnoses
    .sort_values("seq_num")
    .groupby(["subject_id", "hadm_id"])
    .first()
    .reset_index()
)

primary_dx = primary_dx[[
    "subject_id",
    "hadm_id",
    "icd9_code",
    "long_title"
]].rename(columns={
    "long_title": "primary_diagnosis"
})

print("Mapped ICD diagnoses")

# ================================
# 4. Prescription aggregation
# ================================

# Keep only main medications
prescriptions_clean = prescriptions[
    prescriptions["drug_type"] == "MAIN"
]

# Choose most frequent drug per admission
top_drug = (
    prescriptions_clean
    .groupby(["subject_id", "hadm_id"])["drug"]
    .agg(lambda x: x.value_counts().idxmax())
    .reset_index()
    .rename(columns={"drug": "primary_treatment"})
)

print("Aggregated prescriptions")

# ================================
# 5. Combine everything
# ================================

combined = (
    primary_dx
    .merge(patients_demo, on="subject_id", how="left")
    .merge(top_drug, on=["subject_id", "hadm_id"], how="left")
)

# Clean missing values
combined["primary_treatment"] = combined["primary_treatment"].fillna("Unknown")

# Reorder columns
combined = combined[[
    "subject_id",
    "hadm_id",
    "age",
    "gender",
    "primary_diagnosis",
    "primary_treatment"
]]

print("Combined dataset created")

# ================================
# 6. Save for RL training
# ================================

OUTPUT_PATH = "model/dataset/mimic_combined_initial.csv"
combined.to_csv(OUTPUT_PATH, index=False)

print(f"Saved combined dataset to {OUTPUT_PATH}")
print("Rows:", len(combined))
