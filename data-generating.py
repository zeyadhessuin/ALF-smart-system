# 0. Install once if needed
# pip install pandas numpy scikit-learn faker tqdm

import pandas as pd
import numpy as np
from faker import Faker
from tqdm import tqdm
from datetime import timedelta
import random

fake = Faker()
Faker.seed(42)
np.random.seed(42)

# 1. Parameters ─ tweak if you want more rows / facilities
N_RESIDENTS   = 500
N_FACILITIES  = 5
START_DATE    = pd.to_datetime("2024-01-01")
END_DATE      = pd.to_datetime("2024-04-30")   # ~4 months = 120 days
DIAGNOSES     = ["Diabetes", "Dementia", "CHF", "COPD", "Hypertension",
                 "Osteoarthritis", "Depression", "Parkinson"]

# 2. Helper: realistic vitals by age & diagnosis
def sample_vitals(age, diag):
    hr  = int(np.random.normal(80, 10))
    if diag == "CHF": hr += 10
    if diag == "COPD": hr += 5

    bps = int(np.random.normal(120 + max(0, age-70)/2, 15))
    bpd = int(np.random.normal(75 + max(0, age-70)/3, 10))
    temp = np.round(np.random.normal(36.7, 0.3), 1)

    return hr, bps, bpd, temp

# 3. Build residents
residents = []
for pid in range(1, N_RESIDENTS+1):
    residents.append({
        "patient_id": f"P{pid:04d}",
        "facility_id": f"F{np.random.randint(1,N_FACILITIES+1)}",
        "age": np.random.randint(65, 100),
        "gender": np.random.choice(["Male", "Female"], p=[0.4, 0.6]),
        "diagnosis": np.random.choice(DIAGNOSES, p=[0.25,0.15,0.12,0.08,0.15,0.10,0.08,0.07])
    })
res_df = pd.DataFrame(residents)

# 4. Generate daily logs
records = []
for _, row in tqdm(res_df.iterrows(), total=len(res_df)):
    pid, fid, age, gender, diag = row
    cur_date = START_DATE
    while cur_date <= END_DATE:
        hr, bps, bpd, temp = sample_vitals(age, diag)
        med_adherence = np.clip(np.random.beta(8,2), 0, 1)   # high adherence skewed
        # Incident next day (target) ─ low base rate
        incident = np.random.choice([0,1], p=[0.95, 0.05])
        # Make risk higher for dementia/CHF and older ages
        if diag in ["Dementia","CHF"] and age > 85: incident = np.random.choice([0,1], p=[0.85,0.15])

        records.append({
            "patient_id": pid,
            "facility_id": fid,
            "date": cur_date.strftime("%Y-%m-%d"),
            "age": age,
            "gender": gender,
            "diagnosis": diag,
            "heart_rate": hr,
            "blood_pressure_sys": bps,
            "blood_pressure_dia": bpd,
            "temperature": temp,
            "med_adherence": np.round(med_adherence,2),
            "incident_next_day": incident
        })
        cur_date += timedelta(days=1)

df = pd.DataFrame(records)

# 5. Inject missingness (≈3 %) at random
for col in ["heart_rate", "blood_pressure_sys", "blood_pressure_dia", "temperature"]:
    mask = np.random.rand(len(df)) < 0.03
    df.loc[mask, col] = np.nan

# 6. Save
df.to_csv("alf_synthetic.csv", index=False)
print("Shape:", df.shape)
print(df.head())