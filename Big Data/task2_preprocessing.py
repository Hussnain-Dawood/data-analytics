import os

os.environ["JAVA_HOME"]  = r"C:\Program Files\Eclipse Adoptium\jdk-21.0.8.9-hotspot"
os.environ["HADOOP_HOME"] = os.path.abspath("./hadoop")
os.environ["PATH"] = os.path.join(os.environ["JAVA_HOME"], "bin") + ";" + os.environ["PATH"]
os.environ["PYSPARK_SUBMIT_ARGS"] = (
    "--driver-java-options '--add-modules=jdk.incubator.vector' pyspark-shell"
)

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from functools import reduce

from pyspark.sql import SparkSession, DataFrame as SparkDF
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.functions import vector_to_array   # native JVM — no Python UDF

from imblearn.over_sampling import SMOTE

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_PATH   = "creditcard.csv"
OUTPUT_DIR  = "task2_outputs"
PARQUET_DIR = "preprocessed_data"
APP_NAME    = "Task2_Preprocessing"
LABEL_COL   = "Class"
VECTOR_COL  = "features"
RANDOM_SEED = 42

os.makedirs(OUTPUT_DIR,  exist_ok=True)
os.makedirs(PARQUET_DIR, exist_ok=True)


# =============================================================================
# 1. SPARK SESSION
# =============================================================================
def create_spark_session(app_name: str) -> SparkSession:
    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", "8g")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")
        .config("spark.sql.execution.pyspark.udf.faulthandler.enabled", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    print("Congrats! Session Starts")
    return spark


# =============================================================================
# 2. LOAD & VALIDATE
# =============================================================================
def load_and_validate(spark: SparkSession, path: str):
    print(f"\n{'='*60}")
    print("  TASK 2 — Preprocessing Pipeline")
    print(f"{'='*60}")

    df = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(path)
    )

    n_rows    = df.count()
    n_fraud   = df.filter(F.col(LABEL_COL) == 1).count()
    n_genuine = n_rows - n_fraud

    print(f"\n  Loaded  : {n_rows:,} rows")
    print(f"  Genuine : {n_genuine:,}  ({n_genuine/n_rows*100:.3f}%)")
    print(f"  Fraud   : {n_fraud:,}  ({n_fraud/n_rows*100:.3f}%)")

    null_check = df.select(
        [F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df.columns]
    ).collect()[0].asDict()
    print(f"  Null values: {sum(null_check.values())} (dataset is complete)")

    return df, n_rows, n_fraud, n_genuine


# =============================================================================
# 3. STANDARDISE Amount AND Time
# =============================================================================
def scale_amount_time(spark: SparkSession, df):
    """
    StandardScaler on Amount and Time.
    V1-V28 are already PCA-scaled; Amount and Time are not.

    FIX vs original: replaced Python UDF with vector_to_array() (native JVM).
    The UDF caused 'Python worker exited unexpectedly' (EOFException) in
    PySpark 4.x when processing 284K rows across the Python-JVM boundary.
    """
    print(f"\n{'─'*60}")
    print("  3.1  StandardScaler — Amount & Time")
    print(f"{'─'*60}")

    amount_assembler = VectorAssembler(inputCols=["Amount"], outputCol="Amount_vec")
    time_assembler   = VectorAssembler(inputCols=["Time"],   outputCol="Time_vec")
    amount_scaler    = StandardScaler(inputCol="Amount_vec", outputCol="Amount_scaled_vec",
                                      withMean=True, withStd=True)
    time_scaler      = StandardScaler(inputCol="Time_vec",   outputCol="Time_scaled_vec",
                                      withMean=True, withStd=True)

    pipeline     = Pipeline(stages=[amount_assembler, time_assembler,
                                    amount_scaler,    time_scaler])
    scaler_model = pipeline.fit(df)
    df_scaled    = scaler_model.transform(df)

    df_scaled = (
        df_scaled
        .withColumn("Amount_scaled", vector_to_array(F.col("Amount_scaled_vec"))[0])
        .withColumn("Time_scaled",   vector_to_array(F.col("Time_scaled_vec"))[0])
        .drop("Amount_vec", "Time_vec", "Amount_scaled_vec", "Time_scaled_vec")
    )

    print("\n  Before scaling:")
    df.select("Amount", "Time").describe().show(truncate=False)
    print("  After scaling:")
    df_scaled.select("Amount_scaled", "Time_scaled").describe().show(truncate=False)

    return df_scaled


# =============================================================================
# 4. VECTOR ASSEMBLER
# =============================================================================
def assemble_features(df):
    """
    Combine V1-V28 + Amount_scaled + Time_scaled into a single 30-d vector.
    Required input format for all PySpark MLlib estimators.
    """
    print(f"\n{'─'*60}")
    print("  4.1  VectorAssembler — building feature vector")
    print(f"{'─'*60}")

    all_feature_cols = [f"V{i}" for i in range(1, 29)] + ["Amount_scaled", "Time_scaled"]

    assembler    = VectorAssembler(inputCols=all_feature_cols, outputCol=VECTOR_COL,
                                   handleInvalid="skip")
    df_assembled = assembler.transform(df).select(VECTOR_COL, LABEL_COL)

    print(f"  Feature vector assembled: {len(all_feature_cols)} features")
    print(f"  Columns: {VECTOR_COL}, {LABEL_COL}")
    df_assembled.show(5, truncate=True)

    return df_assembled, all_feature_cols


# =============================================================================
# 5. TRAIN / TEST SPLIT (stratified)
# =============================================================================
def stratified_split(df):
    """
    Stratified 80/20 split: split each class independently, then union.
    Preserves the ~0.173% fraud ratio in both sets.
    """
    print(f"\n{'─'*60}")
    print("  5.1  Stratified Train/Test Split  (80/20)")
    print(f"{'─'*60}")

    df_genuine = df.filter(F.col(LABEL_COL) == 0)
    df_fraud   = df.filter(F.col(LABEL_COL) == 1)

    train_genuine, test_genuine = df_genuine.randomSplit([0.80, 0.20], seed=RANDOM_SEED)
    train_fraud,   test_fraud   = df_fraud.randomSplit([0.80, 0.20],   seed=RANDOM_SEED)

    train_df = train_genuine.union(train_fraud).orderBy(F.rand(seed=RANDOM_SEED))
    test_df  = test_genuine.union(test_fraud).orderBy(F.rand(seed=RANDOM_SEED))

    n_train       = train_df.count()
    n_test        = test_df.count()
    n_train_fraud = train_df.filter(F.col(LABEL_COL) == 1).count()
    n_test_fraud  = test_df.filter(F.col(LABEL_COL)  == 1).count()

    print(f"  Training set  : {n_train:,} rows  ({n_train_fraud} fraud  = {n_train_fraud/n_train*100:.3f}%)")
    print(f"  Test set      : {n_test:,}  rows  ({n_test_fraud}  fraud  = {n_test_fraud/n_test*100:.3f}%)")
    print(f"  Stratification verified: class ratio preserved in both splits")

    return train_df, test_df, n_train, n_test, n_train_fraud


# =============================================================================
# 6. SMOTE OVERSAMPLING (training set only)
# =============================================================================
def apply_smote(spark: SparkSession, train_df, n_train_fraud: int):
    """
    SMOTE applied ONLY to training set — prevents data leakage into test.
    sampling_strategy=0.5 → fraud becomes ~33% of training data.

    FIX: Arrow disabled + chunked createDataFrame (50K rows/chunk).
    PySpark 4.x + Arrow crashes when converting a 341K-row pandas DF in one
    shot (EOFException during shuffle). Chunking keeps each Spark task small.
    """
    print(f"\n{'─'*60}")
    print("  6.1  SMOTE Oversampling (training set only)")
    print(f"{'─'*60}")
    print(f"  Reason: {n_train_fraud} fraud samples insufficient for model learning.")

    train_pd = (
        train_df
        .withColumn("features_arr", vector_to_array(F.col(VECTOR_COL)))
        .select(
            *[F.col("features_arr")[i].alias(f"f{i}") for i in range(30)],
            F.col(LABEL_COL)
        )
        .toPandas()
    )

    X_train = train_pd.drop(columns=[LABEL_COL]).values
    y_train = train_pd[LABEL_COL].values

    print(f"\n  Before SMOTE: {sum(y_train==0):,} genuine  |  {sum(y_train==1):,} fraud")

    smote = SMOTE(sampling_strategy=0.5, random_state=RANDOM_SEED, k_neighbors=5)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    print(f"  After SMOTE:  {sum(y_resampled==0):,} genuine  |  {sum(y_resampled==1):,} fraud")
    print(f"  New fraud ratio: {sum(y_resampled==1)/len(y_resampled)*100:.1f}%")

    feature_names = [f"f{i}" for i in range(30)]
    resampled_pd  = pd.DataFrame(X_resampled, columns=feature_names)
    resampled_pd[LABEL_COL] = y_resampled.astype(int)

    CHUNK        = 50_000
    chunks       = [resampled_pd.iloc[i: i + CHUNK]
                    for i in range(0, len(resampled_pd), CHUNK)]
    print(f"  Converting {len(resampled_pd):,} rows to Spark via {len(chunks)} chunks...")

    assembler    = VectorAssembler(inputCols=feature_names, outputCol=VECTOR_COL)
    spark_chunks = [
        assembler.transform(spark.createDataFrame(chunk)).select(VECTOR_COL, LABEL_COL)
        for chunk in chunks
    ]
    train_smote = reduce(SparkDF.union, spark_chunks)

    return train_smote, X_resampled, y_resampled, resampled_pd


# =============================================================================
# 7. VISUALISE CLASS BALANCE
# =============================================================================
def visualise_smote(n_train, n_train_fraud, y_resampled):
    n_train_genuine = n_train - n_train_fraud
    after_fraud     = int(sum(y_resampled == 1))
    after_genuine   = int(sum(y_resampled == 0))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Class Balance: Before vs After SMOTE", fontsize=14, fontweight="bold")

    colours = ["#185FA5", "#A32D2D"]
    labels  = ["Genuine (0)", "Fraud (1)"]

    for ax, title, vals in [
        (axes[0], "Before SMOTE (Training Set)", [n_train_genuine, n_train_fraud]),
        (axes[1], "After SMOTE (Training Set)",  [after_genuine,   after_fraud]),
    ]:
        ax.bar(labels, vals, color=colours, edgecolor="white")
        ax.set_title(title)
        ax.set_ylabel("Count")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
        for bar, val in zip(ax.patches, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                    f"{val:,}", ha="center", fontsize=10)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/fig6_smote_class_balance.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Saved] {path}")


# =============================================================================
# 8. SAVE SPLITS — both via pyarrow (bypasses Hadoop Windows DLL entirely)
# =============================================================================
def save_splits(resampled_pd, test_df, LABEL_COL, PARQUET_DIR):
    """
    Write both splits using pyarrow directly.

    WHY pyarrow instead of spark.write.parquet():
      Spark's FileOutputCommitter calls Hadoop's FileUtil.canRead() which on
      Windows requires hadoop.dll (NativeIO$Windows.access0). Without the
      exact matching DLL version this throws UnsatisfiedLinkError and aborts
      the job. pyarrow writes Parquet natively with no Hadoop dependency.

    Train: resampled_pd (pandas) already in memory -> direct pyarrow write.
    Test:  Spark DF -> toPandas() -> pyarrow write. (57K rows, fits in RAM)
    The resulting .parquet files are fully compatible with spark.read.parquet()
    in downstream tasks (Tasks 3, 4, 5).
    """
    print(f"\n{'─'*60}")
    print("  8.1  Saving preprocessed data as Parquet  [via pyarrow]")
    print(f"{'─'*60}")

    import shutil

    train_path = f"{PARQUET_DIR}/train_smote.parquet"
    test_path  = f"{PARQUET_DIR}/test.parquet"
    for p in [train_path, test_path]:
        if os.path.isdir(p):
            shutil.rmtree(p)
            print(f"  [Cleaned stale directory] {p}")
        elif os.path.isfile(p):
            os.remove(p)
            print(f"  [Cleaned stale file] {p}")

    train_table = pa.Table.from_pandas(resampled_pd, preserve_index=False)
    pq.write_table(train_table, train_path)
    print(f"  Training set (post-SMOTE) -> {train_path}  [{len(resampled_pd):,} rows]")

    feature_names = [f"f{i}" for i in range(30)]
    test_pd = (
        test_df
        .withColumn("features_arr", vector_to_array(F.col(VECTOR_COL)))
        .select(
            *[F.col("features_arr")[i].alias(feature_names[i]) for i in range(30)],
            F.col(LABEL_COL)
        )
        .toPandas()
    )
    test_table = pa.Table.from_pandas(test_pd, preserve_index=False)
    pq.write_table(test_table, test_path)
    print(f"  Test set (original)       -> {test_path}  [{len(test_pd):,} rows]")
    print(f"\n  Both splits saved. Hadoop Windows DLL fully bypassed.")


# =============================================================================
# 9. SUMMARY
# =============================================================================
def print_summary(n_rows, n_train, n_test, n_train_fraud, y_resampled):
    after_fraud = int(sum(y_resampled == 1))
    print(f"\n{'='*60}")
    print("  TASK 2 — PREPROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"  Raw dataset          : {n_rows:,} rows")
    print(f"  Train split (80%)    : {n_train:,} rows  |  fraud: {n_train_fraud}")
    print(f"  Test split  (20%)    : {n_test:,}  rows  (untouched — no SMOTE)")
    print(f"  After SMOTE training : {len(y_resampled):,} rows  |  fraud: {after_fraud:,}")
    print(f"\n  Steps completed:")
    print(f"    [1] StandardScaler applied to Amount & Time")
    print(f"    [2] VectorAssembler -> 30-feature dense vector")
    print(f"    [3] Stratified 80/20 split (class ratio preserved)")
    print(f"    [4] SMOTE oversampling on training set only")
    print(f"    [5] Splits saved as Parquet -> {PARQUET_DIR}/")
    print(f"\n  Next step: Task 3 — Kafka streaming simulation")
    print(f"{'='*60}\n")


# =============================================================================
# MAIN
# =============================================================================
def main():
    spark = create_spark_session(APP_NAME)
    try:
        df, n_rows, n_fraud, n_genuine                    = load_and_validate(spark, DATA_PATH)
        df_scaled                                         = scale_amount_time(spark, df)
        df_assembled, feats                               = assemble_features(df_scaled)
        train_df, test_df, n_train, n_test, n_train_fraud = stratified_split(df_assembled)
        train_smote, X_res, y_res, resampled_pd           = apply_smote(spark, train_df, n_train_fraud)
        visualise_smote(n_train, n_train_fraud, y_res)
        save_splits(resampled_pd, test_df, LABEL_COL, PARQUET_DIR)
        print_summary(n_rows, n_train, n_test, n_train_fraud, y_res)
    finally:
        spark.stop()
        print("  Spark session stopped.\n")


if __name__ == "__main__":
    main()