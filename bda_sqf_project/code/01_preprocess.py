import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json, os

os.makedirs("plots", exist_ok=True)
plt.style.use("seaborn-v0_8-whitegrid")

print("Loading data...")
USECOLS = [
    "MONTH2", "SUSPECT_ARRESTED_FLAG", "FRISKED_FLAG", "SEARCHED_FLAG",
    "WEAPON_FOUND_FLAG", "FIREARM_FLAG", "SUSPECTED_CRIME_DESCRIPTION",
    "SUSPECT_RACE_DESCRIPTION", "SUSPECT_SEX", "SUSPECT_REPORTED_AGE",
    "STOP_LOCATION_BORO_NAME", "STOP_LOCATION_PRECINCT",
    "STOP_WAS_INITIATED",
    "BACKROUND_CIRCUMSTANCES_VIOLENT_CRIME_FLAG",
    "BACKROUND_CIRCUMSTANCES_SUSPECT_KNOWN_TO_CARRY_WEAPON_FLAG",
]
df = pd.read_excel("sqf-2024.xlsx", engine="openpyxl", usecols=USECOLS)
print(f"Loaded: {df.shape[0]} rows")

# Clean nulls
NULL_VALS = {"(null)", "#N/A", ""}
for col in df.select_dtypes("object").columns:
    df[col] = df[col].where(~df[col].isin(NULL_VALS), other=np.nan)

# Outcome
df["y"] = (df["SUSPECT_ARRESTED_FLAG"] == "Y").astype(np.int8)

# Binary flags
FLAG_MAP = {
    "FRISKED_FLAG":  "frisked_bin",
    "SEARCHED_FLAG": "searched_bin",
    "WEAPON_FOUND_FLAG": "weapon_found_bin",
    "FIREARM_FLAG":  "firearm_bin",
    "BACKROUND_CIRCUMSTANCES_VIOLENT_CRIME_FLAG":                 "violent_bg_bin",
    "BACKROUND_CIRCUMSTANCES_SUSPECT_KNOWN_TO_CARRY_WEAPON_FLAG": "known_weapon_bin",
}
for src, dst in FLAG_MAP.items():
    df[dst] = (df[src] == "Y").astype(np.int8)
df["self_init"] = df["STOP_WAS_INITIATED"].str.contains("Self", na=False).astype(np.int8)
df["male_bin"]  = (df["SUSPECT_SEX"] == "MALE").astype(np.int8)

# Age
df["age_num"] = pd.to_numeric(df["SUSPECT_REPORTED_AGE"], errors="coerce")
df = df[(df["age_num"] >= 10) & (df["age_num"] <= 80)].copy()

# Race
df = df[df["SUSPECT_RACE_DESCRIPTION"].notna()].copy()
df["race_grp"] = df["SUSPECT_RACE_DESCRIPTION"].map({
    "BLACK": "Black", "WHITE HISPANIC": "White Hispanic",
    "BLACK HISPANIC": "Black Hispanic", "WHITE": "White",
}).fillna("Other")

# Crime
df = df[df["SUSPECTED_CRIME_DESCRIPTION"].notna()].copy()
def crime_group(c):
    if c in ("CPW", "POSSESSION OF WEAPON"):                 return "Weapon"
    if c in ("ROBBERY","BURGLARY","GRAND LARCENY AUTO",
             "GRAND LARCENY","PETIT LARCENY"):               return "Property/Theft"
    if c in ("ASSAULT","CRIMINAL MISCHIEF"):                 return "Violence"
    if c == "CRIMINAL TRESPASS":                             return "Trespass"
    return "Other"
df["crime_grp"] = df["SUSPECTED_CRIME_DESCRIPTION"].apply(crime_group)

# Month + Season
MONTHS = ["January","February","March","April","May","June",
          "July","August","September","October","November","December"]
df = df[df["MONTH2"].isin(MONTHS)].copy()
df["month_num"] = df["MONTH2"].apply(lambda m: MONTHS.index(m) + 1)
def season(m):
    if m in (12,1,2): return "Winter"
    if m in (3,4,5):  return "Spring"
    if m in (6,7,8):  return "Summer"
    return "Autumn"
df["season"] = df["month_num"].apply(season)

# Borough
VALID = {"BRONX","BROOKLYN","MANHATTAN","QUEENS","STATEN ISLAND"}
df = df[df["STOP_LOCATION_BORO_NAME"].isin(VALID)].copy()
df["borough"] = df["STOP_LOCATION_BORO_NAME"].str.title()

# Precinct index (1-based for JAGS)
df = df[df["STOP_LOCATION_PRECINCT"].notna()].copy()
df["precinct"] = df["STOP_LOCATION_PRECINCT"].astype(int)
prec_sorted = sorted(df["precinct"].unique())
df["precinct_idx"] = df["precinct"].map({p:i+1 for i,p in enumerate(prec_sorted)}).astype(np.int16)
J = int(df["precinct_idx"].max())

# Standardise age
age_mean, age_std = df["age_num"].mean(), df["age_num"].std()
df["age_z"] = ((df["age_num"] - age_mean) / age_std).astype(np.float32)

# Final sample
KEEP = [
    "y","month_num","season","race_grp","male_bin","age_num","age_z",
    "crime_grp","borough","precinct","precinct_idx",
    "frisked_bin","searched_bin","weapon_found_bin","firearm_bin",
    "violent_bg_bin","known_weapon_bin","self_init",
]
df_clean = df[KEEP].dropna().reset_index(drop=True)
N = len(df_clean)

print(f"\nFinal sample:  N = {N}")
print(f"Precincts:     J = {J}")
print(f"Arrest rate:   {df_clean['y'].mean():.3f}")

df_clean.to_csv("sqf_clean.csv", index=False)
with open("data_info.json","w") as f:
    json.dump({"N":N,"J":J,
               "arrest_rate": round(float(df_clean["y"].mean()),3),
               "age_mean": round(age_mean,2),
               "age_std":  round(age_std,2)}, f, indent=2)
print("Saved: sqf_clean.csv, data_info.json")


BLUE="#2c7bb6"; RED="#d73027"; GREEN="#1a9641"; ORANGE="#fc8d59"
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax in axes.flatten():
    ax.spines[["top","right"]].set_visible(False)

# Borough
boro = df_clean.groupby("borough")["y"].agg(["mean","count"]).sort_values("mean")
ax = axes[0,0]
ax.barh(boro.index, boro["mean"], color=BLUE, alpha=0.85)
for i,(r,n) in enumerate(zip(boro["mean"],boro["count"])):
    ax.text(r+0.005, i, f"n={n}", va="center", fontsize=8)
ax.set_xlabel("Arrest Rate"); ax.set_title("(a) Arrest Rate by Borough", fontweight="bold")
ax.set_xlim(0, boro["mean"].max()+0.10)

# Race
race = df_clean.groupby("race_grp")["y"].agg(["mean","count"]).sort_values("mean")
ax = axes[0,1]
ax.barh(race.index, race["mean"], color=RED, alpha=0.85)
for i,(r,n) in enumerate(zip(race["mean"],race["count"])):
    ax.text(r+0.005, i, f"n={n}", va="center", fontsize=8)
ax.set_xlabel("Arrest Rate"); ax.set_title("(b) Arrest Rate by Race", fontweight="bold")
ax.set_xlim(0, race["mean"].max()+0.10)

# Crime
crime = df_clean.groupby("crime_grp")["y"].agg(["mean","count"]).sort_values("mean")
ax = axes[1,0]
ax.barh(crime.index, crime["mean"], color=GREEN, alpha=0.85)
for i,(r,n) in enumerate(zip(crime["mean"],crime["count"])):
    ax.text(r+0.005, i, f"n={n}", va="center", fontsize=8)
ax.set_xlabel("Arrest Rate"); ax.set_title("(c) Arrest Rate by Crime Type", fontweight="bold")
ax.set_xlim(0, crime["mean"].max()+0.10)

# Monthly
month_stats = df_clean.groupby("month_num")["y"].mean()
ax = axes[1,1]
ax.plot(month_stats.index, month_stats.values, "o-", color=ORANGE, lw=2, ms=7)
ax.axhline(df_clean["y"].mean(), color="grey", ls="--", lw=1, label="Overall mean")
ax.set_xticks(range(1,13))
ax.set_xticklabels(["J","F","M","A","M","J","J","A","S","O","N","D"])
ax.set_xlabel("Month"); ax.set_ylabel("Arrest Rate")
ax.set_title("(d) Seasonal Arrest Rate Pattern", fontweight="bold")
ax.legend(fontsize=8)

plt.suptitle("EDA: NYPD Stop, Question & Frisk 2024", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("plots/eda_overview.png", dpi=130, bbox_inches="tight")
plt.close()
print("Saved plots/eda_overview.png")
