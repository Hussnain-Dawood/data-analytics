import os

os.environ["JAVA_HOME"] = r"C:\Program Files\Eclipse Adoptium\jdk-21.0.8.9-hotspot"

os.environ["HADOOP_HOME"] = os.path.abspath("./hadoop")

os.environ["PATH"] = os.path.join(os.environ["JAVA_HOME"], "bin") + ";" + os.environ["PATH"]

os.environ["PYSPARK_SUBMIT_ARGS"] = "--driver-java-options '--add-modules=jdk.incubator.vector' pyspark-shell"

from pyspark.sql import SparkSession

try:
    spark = SparkSession.builder \
        .master("local[*]") \
        .getOrCreate()
    print("Congrats! Spark 4.1.1 is now running with Java 21.")
except Exception as e:
    print(f"Still error: {e}")


import os
import sys

project_dir = os.getcwd()
os.environ["JAVA_HOME"] = os.path.join(project_dir, "jdk17")
os.environ["HADOOP_HOME"] = os.path.join(project_dir, "hadoop")
os.environ["PATH"] = os.path.join(os.environ["JAVA_HOME"], "bin") + ";" + os.environ["PATH"]

if not os.path.exists(os.environ["JAVA_HOME"]):
    print(f"Error: {os.environ['JAVA_HOME']} folder nahi mila!")
# ------------------------------------------

import pyspark

import os
import sys

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType

import matplotlib
matplotlib.use("Agg")        
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import pandas as pd
import numpy as np

DATA_PATH   = "creditcard.csv"        
OUTPUT_DIR  = "task1_outputs"
APP_NAME    = "Task1_CreditCard_EDA"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# 1. SPARK SESSION
# =============================================================================
def create_spark_session(app_name: str) -> SparkSession:
    """
    Initialise a local SparkSession.
    In production, replace master('local[*]') with the cluster URL.
    """
    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")   # keep small for local
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


# =============================================================================
# 2. LOAD DATA
# =============================================================================
def load_data(spark: SparkSession, path: str):
    """
    Load creditcard.csv with inferred schema.
    The dataset contains:
      - Time    : seconds elapsed since first transaction
      - V1-V28  : PCA-transformed anonymised features
      - Amount  : transaction amount (EUR)
      - Class   : 0 = genuine, 1 = fraudulent
    """
    print(f"\n{'='*60}")
    print("  TASK 1 — Loading Dataset")
    print(f"{'='*60}")

    df = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(path)
    )
    print(f"  File loaded: {path}")
    return df


# =============================================================================
# 3. SCHEMA & BASIC STATISTICS
# =============================================================================
def schema_summary(df):
    """Print schema, shape, and basic statistics."""

    print(f"\n{'─'*60}")
    print("  3.1  Schema")
    print(f"{'─'*60}")
    df.printSchema()

    n_rows = df.count()
    n_cols = len(df.columns)
    print(f"  Shape  :  {n_rows:,} rows  ×  {n_cols} columns")

    print(f"\n{'─'*60}")
    print("  3.2  Descriptive Statistics (Amount & Time)")
    print(f"{'─'*60}")
    df.select("Time", "Amount").describe().show(truncate=False)

    print(f"\n{'─'*60}")
    print("  3.3  Null / Missing Values")
    print(f"{'─'*60}")
    null_counts = df.select(
        [F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df.columns]
    )
    null_counts.show(truncate=False)

    return n_rows, n_cols


# =============================================================================
# 4. CLASS DISTRIBUTION (fraud imbalance analysis)
# =============================================================================
def class_distribution(df, n_rows: int):
    """
    Analyse and visualise the severe class imbalance.
    Fraud accounts for only ~0.172% of all transactions.
    """
    print(f"\n{'─'*60}")
    print("  4.1  Class Distribution")
    print(f"{'─'*60}")

    dist = (
        df.groupBy("Class")
        .agg(
            F.count("*").alias("Count"),
            F.round(F.count("*") / n_rows * 100, 4).alias("Percentage_%")
        )
        .orderBy("Class")
    )
    dist.show()

    dist_pd = dist.toPandas()
    dist_pd["Label"] = dist_pd["Class"].map({0: "Genuine (0)", 1: "Fraud (1)"})

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Class Distribution — Credit Card Transactions", fontsize=14, fontweight="bold")

    colours = ["#185FA5", "#A32D2D"]

    axes[0].bar(dist_pd["Label"], dist_pd["Count"], color=colours, edgecolor="white", linewidth=0.5)
    axes[0].set_title("Transaction Count by Class")
    axes[0].set_ylabel("Number of Transactions")
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    for bar, val in zip(axes[0].patches, dist_pd["Count"]):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                     f"{val:,}", ha="center", va="bottom", fontsize=10)

    axes[1].pie(
        dist_pd["Count"],
        labels=dist_pd["Label"],
        autopct="%1.3f%%",
        colors=colours,
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1}
    )
    axes[1].set_title("Class Proportion")

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/fig1_class_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Saved] {path}")

    return dist_pd


# =============================================================================
# 5. FEATURE STATISTICS (V1–V28, Amount, Time)
# =============================================================================
def feature_statistics(df):
    """
    Compute per-feature mean, std, min, max for genuine vs fraud.
    Key analytical insight: PCA features with largest mean difference
    between classes are the most discriminative.
    """
    print(f"\n{'─'*60}")
    print("  5.1  Feature Statistics by Class")
    print(f"{'─'*60}")

    feature_cols = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]

    stats_fraud   = df.filter(F.col("Class") == 1).select(feature_cols).describe()
    stats_genuine = df.filter(F.col("Class") == 0).select(feature_cols).describe()

    print("  Fraud transactions — descriptive stats:")
    stats_fraud.show(truncate=False)
    print("  Genuine transactions — descriptive stats:")
    stats_genuine.show(truncate=False)

    # Mean comparison: fraud vs genuine for V-features
    agg_exprs_fraud = [
        F.mean(c).alias(c) for c in feature_cols
    ]
    mean_fraud   = df.filter(F.col("Class") == 1).agg(*agg_exprs_fraud).toPandas().T
    mean_genuine = df.filter(F.col("Class") == 0).agg(*agg_exprs_fraud).toPandas().T

    mean_fraud.columns   = ["Fraud"]
    mean_genuine.columns = ["Genuine"]
    mean_comparison = pd.concat([mean_fraud, mean_genuine], axis=1)
    mean_comparison["Abs_Diff"] = (mean_comparison["Fraud"] - mean_comparison["Genuine"]).abs()
    mean_comparison = mean_comparison.sort_values("Abs_Diff", ascending=False)

    print("\n  Top 10 most discriminative features (by mean difference):")
    print(mean_comparison.head(10).to_string())

    # Bar chart of mean difference
    top15 = mean_comparison.head(15)
    fig, ax = plt.subplots(figsize=(12, 5))
    x = range(len(top15))
    width = 0.35
    ax.bar([i - width/2 for i in x], top15["Genuine"], width, label="Genuine", color="#185FA5", alpha=0.85)
    ax.bar([i + width/2 for i in x], top15["Fraud"],   width, label="Fraud",   color="#A32D2D", alpha=0.85)
    ax.set_xticks(list(x))
    ax.set_xticklabels(top15.index, rotation=45, ha="right")
    ax.set_title("Feature Means: Genuine vs Fraud (Top 15 by Absolute Difference)", fontweight="bold")
    ax.set_ylabel("Mean Value")
    ax.legend()
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/fig2_feature_means_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Saved] {path}")

    return mean_comparison


# =============================================================================
# 6. AMOUNT DISTRIBUTION
# =============================================================================
def amount_distribution(df):
    """
    Analyse transaction amount distributions.
    Fraudulent transactions tend to cluster at lower amounts.
    """
    print(f"\n{'─'*60}")
    print("  6.1  Transaction Amount Distribution")
    print(f"{'─'*60}")

    amount_stats = df.groupBy("Class").agg(
        F.mean("Amount").alias("Mean_Amount"),
        F.stddev("Amount").alias("Std_Amount"),
        F.min("Amount").alias("Min_Amount"),
        F.max("Amount").alias("Max_Amount"),
        F.expr("percentile_approx(Amount, 0.5)").alias("Median_Amount")
    ).orderBy("Class")
    amount_stats.show(truncate=False)

    # Collect sample for plotting (limit for performance)
    sample_pd = df.select("Amount", "Class").sample(fraction=0.1, seed=42).toPandas()
    sample_pd["Label"] = sample_pd["Class"].map({0: "Genuine", 1: "Fraud"})

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Transaction Amount Distribution", fontsize=14, fontweight="bold")

    for i, (label, colour) in enumerate([("Genuine", "#185FA5"), ("Fraud", "#A32D2D")]):
        data = sample_pd[sample_pd["Label"] == label]["Amount"]
        axes[0].hist(data, bins=60, alpha=0.6, color=colour, label=label, edgecolor="none")

    axes[0].set_xlabel("Amount (EUR)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Amount Histogram (10% sample)")
    axes[0].legend()
    axes[0].set_yscale("log")

    # Box plot
    sample_pd.boxplot(column="Amount", by="Label", ax=axes[1],
                      boxprops=dict(color="#185FA5"),
                      medianprops=dict(color="#A32D2D", linewidth=2))
    axes[1].set_title("Amount Box Plot by Class")
    axes[1].set_xlabel("Class")
    axes[1].set_ylabel("Amount (EUR)")
    axes[1].set_yscale("log")
    plt.suptitle("")
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/fig3_amount_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Saved] {path}")


# =============================================================================
# 7. TIME DISTRIBUTION
# =============================================================================
def time_distribution(df):
    """
    Analyse transaction timing patterns.
    Time is in seconds from the first transaction over a 48-hour window.
    """
    print(f"\n{'─'*60}")
    print("  7.1  Transaction Time Distribution")
    print(f"{'─'*60}")

    df_time = df.withColumn("Hour", F.round(F.col("Time") / 3600, 1))

    hourly = df_time.groupBy("Hour", "Class").count().orderBy("Hour")
    print("  Hourly transaction counts (sample):")
    hourly.show(10, truncate=False)

    sample_pd = df_time.select("Hour", "Class").sample(fraction=0.2, seed=42).toPandas()
    sample_pd["Label"] = sample_pd["Class"].map({0: "Genuine", 1: "Fraud"})

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle("Transaction Time Distribution (over 48-hour window)", fontsize=14, fontweight="bold")

    axes[0].hist(sample_pd[sample_pd["Label"]=="Genuine"]["Hour"],
                 bins=48, color="#185FA5", alpha=0.8, edgecolor="none")
    axes[0].set_title("Genuine Transactions")
    axes[0].set_ylabel("Count")
    axes[0].set_xlabel("Hour")

    axes[1].hist(sample_pd[sample_pd["Label"]=="Fraud"]["Hour"],
                 bins=48, color="#A32D2D", alpha=0.8, edgecolor="none")
    axes[1].set_title("Fraudulent Transactions")
    axes[1].set_ylabel("Count")
    axes[1].set_xlabel("Hour")

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/fig4_time_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Saved] {path}")


# =============================================================================
# 8. CORRELATION HEATMAP (on a pandas sample)
# =============================================================================
def correlation_heatmap(df):
    """
    Compute Pearson correlation among all numeric features.
    PCA features V1-V28 are by design uncorrelated with each other;
    the interest lies in their correlation with Class.
    """
    print(f"\n{'─'*60}")
    print("  8.1  Correlation Analysis")
    print(f"{'─'*60}")

    feature_cols = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time", "Class"]

    sample_pd = df.select(feature_cols).sample(fraction=0.2, seed=42).toPandas()
    corr_matrix = sample_pd.corr()

    class_corr = corr_matrix["Class"].drop("Class").sort_values(key=abs, ascending=False)
    print("\n  Feature correlations with Class (|r| descending):")
    print(class_corr.head(15).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle("Correlation Analysis", fontsize=14, fontweight="bold")

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix, mask=mask, ax=axes[0],
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        linewidths=0.3, square=False,
        xticklabels=True, yticklabels=True,
        cbar_kws={"shrink": 0.6}
    )
    axes[0].set_title("Full Correlation Matrix (20% sample)")
    axes[0].tick_params(labelsize=7)

    colours = ["#A32D2D" if v < 0 else "#185FA5" for v in class_corr.values]
    axes[1].barh(class_corr.index, class_corr.values, color=colours, edgecolor="none")
    axes[1].axvline(0, color="gray", linewidth=0.8)
    axes[1].set_title("Feature Correlation with Class (fraud=1)")
    axes[1].set_xlabel("Pearson r")
    axes[1].invert_yaxis()

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/fig5_correlation_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Saved] {path}")

    return class_corr


# =============================================================================
# 9. SUMMARY REPORT
# =============================================================================
def print_summary(n_rows, n_cols, dist_pd, class_corr):
    """Print a clean EDA summary to console."""

    fraud_count   = int(dist_pd[dist_pd["Class"]==1]["Count"].values[0])
    genuine_count = int(dist_pd[dist_pd["Class"]==0]["Count"].values[0])
    fraud_pct     = float(dist_pd[dist_pd["Class"]==1]["Percentage_%"].values[0])

    print(f"\n{'='*60}")
    print("  TASK 1 — EDA SUMMARY")
    print(f"{'='*60}")
    print(f"  Total transactions    : {n_rows:,}")
    print(f"  Features              : {n_cols}")
    print(f"  Genuine transactions  : {genuine_count:,}  ({100-fraud_pct:.3f}%)")
    print(f"  Fraud transactions    : {fraud_count:,}  ({fraud_pct:.3f}%)")
    print(f"  Imbalance ratio       : 1 fraud per {genuine_count//fraud_count:,} genuine")
    print(f"\n  Top 5 features correlated with fraud:")
    for feat, val in class_corr.head(5).items():
        print(f"    {feat:8s}  r = {val:+.4f}")
    print(f"\n  Key finding: Severe class imbalance ({fraud_pct:.3f}% fraud).")
    print(f"  SMOTE oversampling required before training (Task 2).")
    print(f"  Outputs saved to: {OUTPUT_DIR}/")
    print(f"{'='*60}\n")


# =============================================================================
# MAIN
# =============================================================================
def main():
    spark = create_spark_session(APP_NAME)

    try:
        df = load_data(spark, DATA_PATH)
        n_rows, n_cols   = schema_summary(df)
        dist_pd          = class_distribution(df, n_rows)
        mean_comparison  = feature_statistics(df)
        amount_distribution(df)
        time_distribution(df)
        class_corr       = correlation_heatmap(df)
        print_summary(n_rows, n_cols, dist_pd, class_corr)

    finally:
        spark.stop()
        print("  Spark session stopped.\n")


if __name__ == "__main__":
    main()
