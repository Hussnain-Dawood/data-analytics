import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 120, "figure.facecolor": "white"})

# ─────────────────────────────────────────────
print("=" * 60)
print("1. LOADING DATA")
print("=" * 60)

sales    = pd.read_csv("sales data-set.csv")
features = pd.read_csv("Features data set.csv")
stores   = pd.read_csv("stores data-set.csv")

print(f"Sales    : {sales.shape[0]:,} rows × {sales.shape[1]} columns")
print(f"Features : {features.shape[0]:,} rows × {features.shape[1]} columns")
print(f"Stores   : {stores.shape[0]:,} rows × {stores.shape[1]} columns")

print("\nSales columns    :", sales.columns.tolist())
print("Features columns :", features.columns.tolist())
print("Stores columns   :", stores.columns.tolist())

# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. DATA CLEANING & PREPROCESSING")
print("=" * 60)

# --- Parse dates ---
for df_name, df_obj, col in [("Sales", sales, "Date"), ("Features", features, "Date")]:
    df_obj[col] = pd.to_datetime(df_obj[col], dayfirst=True, errors="coerce")
    print(f"Parsed '{col}' in {df_name}: sample → {df_obj[col].head(3).dt.strftime('%Y-%m-%d').tolist()}")

# --- Merge all three datasets ---
df = (sales
      .merge(stores,   on="Store",        how="left")
      .merge(features, on=["Store", "Date", "IsHoliday"], how="left"))

print(f"\nMerged dataframe shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print("Columns:", df.columns.tolist())

# --- Missing values BEFORE treatment ---
print("\n--- Missing values before treatment ---")
missing = df.isnull().sum()
print(missing[missing > 0].to_string())

# --- Fill MarkDown NAs with 0 (no promotion ran) ---
markdown_cols = ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]
df[markdown_cols] = df[markdown_cols].fillna(0)
print(f"\nFilled {markdown_cols} NaN → 0 (no promotion)")

# --- Fill CPI and Unemployment with forward-fill then median ---
for col in ["CPI", "Unemployment"]:
    df[col] = df[col].ffill().bfill()
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].median())
    print(f"Filled '{col}' NaN with forward-fill (remaining NaN: {df[col].isnull().sum()})")

print("\n--- Missing values after treatment ---")
remaining = df.isnull().sum()
print(remaining[remaining > 0].to_string() if remaining.sum() > 0 else "  No missing values remaining.")

# --- Inject & treat artificial NaNs in Weekly_Sales ---
np.random.seed(0)
miss_idx = np.random.choice(df.index, size=500, replace=False)
df.loc[miss_idx, "Weekly_Sales"] = np.nan
print(f"\nArtificially injected 500 NaN into 'Weekly_Sales'")
df["Weekly_Sales"] = df["Weekly_Sales"].fillna(df["Weekly_Sales"].median())
print(f"Filled with median. Remaining NaN: {df['Weekly_Sales'].isnull().sum()}")

# --- Outlier treatment ---
def remove_outliers_iqr(series, multiplier=3.0):
    Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
    IQR    = Q3 - Q1
    lo, hi = Q1 - multiplier * IQR, Q3 + multiplier * IQR
    mask   = (series < lo) | (series > hi)
    return series.clip(lower=lo, upper=hi), mask

print("\n--- Outlier Treatment (IQR × 3, capping) ---")
for col in ["Weekly_Sales", "Temperature", "Fuel_Price", "CPI", "Unemployment"]:
    df[col], mask = remove_outliers_iqr(df[col])
    print(f"  {col}: {mask.sum():,} outliers capped")

# --- Standardisation & Normalisation ---
num_cols = ["Weekly_Sales", "Temperature", "Fuel_Price", "CPI", "Unemployment", "Size"]

scaler_std = StandardScaler()
df_standardised = pd.DataFrame(
    scaler_std.fit_transform(df[num_cols]),
    columns=[c + "_std" for c in num_cols]
)

scaler_mm = MinMaxScaler()
df_normalised = pd.DataFrame(
    scaler_mm.fit_transform(df[num_cols]),
    columns=[c + "_norm" for c in num_cols]
)

print("\n--- Standardised columns (first 3 rows) ---")
print(df_standardised.head(3).to_string(index=False))

print("\n--- Normalised columns (first 3 rows) ---")
print(df_normalised.head(3).to_string(index=False))

# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. DESCRIPTIVE STATISTICS & DISTRIBUTION ANALYSIS")
print("=" * 60)

desc = df[num_cols].describe().T
desc["skewness"] = df[num_cols].skew()
desc["kurtosis"] = df[num_cols].kurt()
print(desc.to_string())

print("\n--- Interpretation ---")
for col in num_cols:
    skew = df[col].skew()
    kurt = df[col].kurt()
    if abs(skew) < 0.5:
        dist = "approximately normal (symmetric)"
    elif skew > 0:
        dist = "right-skewed (positive tail)"
    else:
        dist = "left-skewed (negative tail)"
    print(f"  {col:20s} | skewness={skew:+.3f}  kurtosis={kurt:+.3f}  → {dist}")

# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. GENERATING VISUALISATIONS")
print("=" * 60)

# ── Plot 1: Distribution histograms + KDE ──
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("Distribution of Key Numerical Variables", fontsize=14, fontweight="bold")

for ax, col in zip(axes.flatten(), num_cols):
    ax.hist(df[col], bins=50, color="steelblue", edgecolor="white", alpha=0.7,
            density=True, label="Histogram")
    df[col].plot(kind="kde", ax=ax, color="crimson", linewidth=1.8, label="KDE")
    ax.set_title(col)
    ax.set_xlabel(col)
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("plot1_distributions.png", bbox_inches="tight")
plt.close()
print("  Saved: plot1_distributions.png")

# ── Plot 2: Correlation Heatmap ──
fig, ax = plt.subplots(figsize=(9, 7))
corr = df[num_cols + markdown_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.5, ax=ax, vmin=-1, vmax=1)
ax.set_title("Correlation Heatmap – Numerical Features", fontweight="bold")
plt.tight_layout()
plt.savefig("plot2_correlation_heatmap.png", bbox_inches="tight")
plt.close()
print("  Saved: plot2_correlation_heatmap.png")

# ── Plot 3: Monthly Sales Trend ──
monthly = (df.set_index("Date")
             .resample("ME")["Weekly_Sales"]
             .sum()
             .reset_index())
monthly.columns = ["Month", "Total_Sales"]

fig, ax = plt.subplots(figsize=(14, 5))
ax.fill_between(monthly["Month"], monthly["Total_Sales"], alpha=0.25, color="steelblue")
ax.plot(monthly["Month"], monthly["Total_Sales"], color="steelblue", linewidth=1.5)
ax.set_title("Monthly Total Weekly Sales Trend", fontsize=13, fontweight="bold")
ax.set_xlabel("Month")
ax.set_ylabel("Total Weekly Sales ($)")
plt.tight_layout()
plt.savefig("plot3_monthly_sales_trend.png", bbox_inches="tight")
plt.close()
print("  Saved: plot3_monthly_sales_trend.png")

# ── Plot 4: Sales by Store Type & Holiday ──
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Weekly Sales Analysis", fontsize=13, fontweight="bold")

store_type_sales = df.groupby("Type")["Weekly_Sales"].mean().sort_values(ascending=False)
axes[0].bar(store_type_sales.index, store_type_sales.values,
            color=["#1f77b4", "#ff7f0e", "#2ca02c"])
axes[0].set_title("Average Weekly Sales by Store Type")
axes[0].set_xlabel("Store Type")
axes[0].set_ylabel("Avg Weekly Sales ($)")
for i, v in enumerate(store_type_sales.values):
    axes[0].text(i, v + 500, f"${v:,.0f}", ha="center", fontsize=9)

holiday_sales = df.groupby("IsHoliday")["Weekly_Sales"].mean()
labels = ["Non-Holiday", "Holiday"]
colors = ["#4c72b0", "#dd8452"]
axes[1].bar(labels, holiday_sales.values, color=colors, width=0.5)
axes[1].set_title("Average Weekly Sales: Holiday vs Non-Holiday")
axes[1].set_ylabel("Avg Weekly Sales ($)")
for i, v in enumerate(holiday_sales.values):
    axes[1].text(i, v + 200, f"${v:,.0f}", ha="center", fontsize=9)

plt.tight_layout()
plt.savefig("plot4_sales_by_type_and_holiday.png", bbox_inches="tight")
plt.close()
print("  Saved: plot4_sales_by_type_and_holiday.png")

# ── Plot 5: Top 10 Stores by Total Sales ──
top_stores = (df.groupby("Store")["Weekly_Sales"]
                .sum()
                .sort_values(ascending=False)
                .head(10))

fig, ax = plt.subplots(figsize=(10, 6))
top_stores.plot(kind="barh", ax=ax, color="teal")
ax.set_title("Top 10 Stores by Total Weekly Sales", fontsize=13, fontweight="bold")
ax.set_xlabel("Total Sales ($)")
ax.set_ylabel("Store #")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig("plot5_top10_stores.png", bbox_inches="tight")
plt.close()
print("  Saved: plot5_top10_stores.png")

# ── Plot 6: Boxplots ──
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Boxplots – Sales, Temperature, Fuel Price", fontsize=13, fontweight="bold")
for ax, col in zip(axes, ["Weekly_Sales", "Temperature", "Fuel_Price"]):
    sns.boxplot(y=df[col], ax=ax, color="lightcoral",
                flierprops={"marker": ".", "markerfacecolor": "grey", "markersize": 3})
    ax.set_title(col)
    ax.set_ylabel(col)
plt.tight_layout()
plt.savefig("plot6_boxplots.png", bbox_inches="tight")
plt.close()
print("  Saved: plot6_boxplots.png")

# ── Plot 7: Store Type Distribution ──
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Store Type Distribution", fontsize=13, fontweight="bold")

type_counts = stores["Type"].value_counts()
axes[0].pie(type_counts, labels=type_counts.index, autopct="%1.1f%%",
            colors=["#4c72b0", "#dd8452", "#2ca02c"], startangle=90)
axes[0].set_title("Pie Chart – Store Types")

sns.countplot(x="Type", data=stores, ax=axes[1], palette="muted",
              order=["A", "B", "C"])
axes[1].set_title("Count Plot – Store Types")
axes[1].set_xlabel("Store Type")
axes[1].set_ylabel("Count")
plt.tight_layout()
plt.savefig("plot7_store_type_distribution.png", bbox_inches="tight")
plt.close()
print("  Saved: plot7_store_type_distribution.png")

# ── Plot 8: Standardisation Comparison ──
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Weekly Sales: Original vs Z-Score Standardised", fontsize=13, fontweight="bold")

axes[0].hist(df["Weekly_Sales"], bins=60, color="steelblue", edgecolor="white", alpha=0.75)
axes[0].set_title("Original Weekly Sales")
axes[0].set_xlabel("Weekly Sales ($)")

axes[1].hist(df_standardised["Weekly_Sales_std"], bins=60,
             color="darkorange", edgecolor="white", alpha=0.75)
axes[1].set_title("Standardised Weekly Sales (Z-Score)")
axes[1].set_xlabel("Z-Score")

plt.tight_layout()
plt.savefig("plot8_standardisation_comparison.png", bbox_inches="tight")
plt.close()
print("  Saved: plot8_standardisation_comparison.png")

# ── Plot 9: MarkDown Impact on Sales ──
fig, axes = plt.subplots(1, 5, figsize=(18, 5))
fig.suptitle("MarkDown Promotions vs Weekly Sales", fontsize=13, fontweight="bold")

for ax, md in zip(axes, markdown_cols):
    active = df[df[md] > 0]
    ax.scatter(active[md], active["Weekly_Sales"], alpha=0.2, s=5, color="purple")
    ax.set_title(md)
    ax.set_xlabel(f"{md} ($)")
    ax.set_ylabel("Weekly Sales ($)" if md == "MarkDown1" else "")
    ax.tick_params(axis="x", labelrotation=30)

plt.tight_layout()
plt.savefig("plot9_markdown_vs_sales.png", bbox_inches="tight")
plt.close()
print("  Saved: plot9_markdown_vs_sales.png")

# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. DATA VALIDATION SUMMARY")
print("=" * 60)

print(f"  Total rows (merged)    : {len(df):,}")
print(f"  Duplicate rows         : {df.duplicated().sum():,}")
print(f"  Remaining NaN values   : {df.isnull().sum().sum():,}")
print(f"  Weekly Sales – min/max : ${df['Weekly_Sales'].min():,.2f} / ${df['Weekly_Sales'].max():,.2f}")
print(f"  Date range             : {df['Date'].min().date()} → {df['Date'].max().date()}")
print(f"  Unique Stores          : {df['Store'].nunique()}")
print(f"  Unique Departments     : {df['Dept'].nunique()}")
print(f"  Store Types            : {sorted(df['Type'].unique())}")
print(f"  Holiday weeks          : {df[df['IsHoliday']]['Date'].dt.isocalendar().week.nunique()}")
print(f"  Avg weekly sales       : ${df['Weekly_Sales'].mean():,.2f}")

print("\n  EDA complete. All plots saved to the working directory.")