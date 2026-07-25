import os
import time
import json
import pickle
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from pyspark.ml import Pipeline
from pyspark.ml.classification import (
    LogisticRegression,
    RandomForestClassifier,
    GBTClassifier,
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.linalg import Vectors

# =============================================================================
# CONFIGURATION (OPTIMIZED)
# =============================================================================
PARQUET_TRAIN  = "preprocessed_data/train_smote.parquet"
PARQUET_TEST   = "preprocessed_data/test.parquet"
OUTPUT_DIR     = "task4_outputs"
MODELS_DIR     = "task4_outputs/saved_models"
FEATURE_COL    = "features"
LABEL_COL      = "Class"
RANDOM_SEED    = 42
N_FOLDS        = 3         
APP_NAME       = "Task4_MLModels_Optimized"

os.makedirs(OUTPUT_DIR,  exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)


# =============================================================================
# HELPER: Safe Model Save (Fixes Hadoop Windows Error)
# =============================================================================
def safe_model_save(model, path, model_name):
    """
    Safely save model with error handling for Hadoop Windows issue.
    """
    try:
        model.write().overwrite().save(path)
        print(f"  Model saved: {path}")
    except Exception as e:
        print(f"  Warning: Could not save model to disk (Hadoop Windows issue)")
        print(f"  Error: {str(e)[:100]}...")
        print(f"  Model training successful - continuing with evaluation")
        try:
            with open(f"{path}.pkl", "wb") as f:
                pickle.dump(model, f)
            print(f"  Backup saved as pickle: {path}.pkl")
        except:
            print(f"  Pickle backup also failed - skipping save")


# =============================================================================
# 1. SPARK SESSION
# =============================================================================
def create_spark_session(app_name: str) -> SparkSession:
    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", "8g")
        .config("spark.executor.memory", "4g")
        .config("spark.sql.shuffle.partitions", "4")  
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


# =============================================================================
# 2. LOAD PREPROCESSED DATA
# =============================================================================
def load_data(spark: SparkSession):
    """
    Load the SMOTE-oversampled training set and the original test set.
    """
    from pyspark.ml.feature import VectorAssembler

    print(f"\n{'='*60}")
    print("  TASK 4 — PySpark ML Classification Models (OPTIMIZED)")
    print(f"{'='*60}")

    train_df = spark.read.parquet(PARQUET_TRAIN)
    test_df  = spark.read.parquet(PARQUET_TEST)

    def _ensure_features(df):
        """Reassemble features if needed."""
        if FEATURE_COL in df.columns:
            return df

        flat_cols = [c for c in df.columns if c.startswith("f") and c[1:].isdigit()]
        if flat_cols:
            flat_cols_sorted = sorted(flat_cols, key=lambda c: int(c[1:]))
            print(f"  [INFO] Reassembling {len(flat_cols_sorted)} flat columns "
                  f"({flat_cols_sorted[0]}..{flat_cols_sorted[-1]}) → '{FEATURE_COL}'")
            assembler = VectorAssembler(
                inputCols=flat_cols_sorted,
                outputCol=FEATURE_COL,
                handleInvalid="skip"
            )
            return assembler.transform(df).select(FEATURE_COL, LABEL_COL)

        raise ValueError(f"Cannot find '{FEATURE_COL}' column or flat columns in Parquet.")

    train_df = _ensure_features(train_df)
    test_df  = _ensure_features(test_df)

    n_train       = train_df.count()
    n_test        = test_df.count()
    n_train_fraud = train_df.filter(F.col(LABEL_COL) == 1).count()
    n_test_fraud  = test_df.filter(F.col(LABEL_COL)  == 1).count()

    print(f"\n  Training set : {n_train:,} rows  ({n_train_fraud:,} fraud = "
          f"{n_train_fraud/n_train*100:.1f}%)")
    print(f"  Test set     : {n_test:,}  rows  ({n_test_fraud}  fraud = "
          f"{n_test_fraud/n_test*100:.3f}%)")

    train_df = train_df.withColumn(LABEL_COL, F.col(LABEL_COL).cast("double")).cache()
    test_df  = test_df.withColumn(LABEL_COL,  F.col(LABEL_COL).cast("double")).cache()

    train_df.count()
    test_df.count()
    print("  [INFO] Both splits cached in memory for faster CV training.")

    return train_df, test_df


# =============================================================================
# 3. EVALUATION HELPER
# =============================================================================
def evaluate_model(predictions, model_name: str) -> dict:
    """Compute comprehensive metrics for binary fraud classification."""
    auc_roc_eval = BinaryClassificationEvaluator(
        labelCol=LABEL_COL, rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )
    auc_pr_eval = BinaryClassificationEvaluator(
        labelCol=LABEL_COL, rawPredictionCol="rawPrediction",
        metricName="areaUnderPR"
    )
    acc_eval = MulticlassClassificationEvaluator(
        labelCol=LABEL_COL, predictionCol="prediction",
        metricName="accuracy"
    )
    prec_eval = MulticlassClassificationEvaluator(
        labelCol=LABEL_COL, predictionCol="prediction",
        metricName="weightedPrecision"
    )
    rec_eval = MulticlassClassificationEvaluator(
        labelCol=LABEL_COL, predictionCol="prediction",
        metricName="weightedRecall"
    )
    f1_eval = MulticlassClassificationEvaluator(
        labelCol=LABEL_COL, predictionCol="prediction",
        metricName="f1"
    )

    auc_roc   = auc_roc_eval.evaluate(predictions)
    auc_pr    = auc_pr_eval.evaluate(predictions)
    accuracy  = acc_eval.evaluate(predictions)
    precision = prec_eval.evaluate(predictions)
    recall    = rec_eval.evaluate(predictions)
    f1        = f1_eval.evaluate(predictions)
    f2        = (5 * precision * recall) / (4 * precision + recall + 1e-10)

    tp = predictions.filter((F.col("prediction") == 1) & (F.col(LABEL_COL) == 1)).count()
    fp = predictions.filter((F.col("prediction") == 1) & (F.col(LABEL_COL) == 0)).count()
    tn = predictions.filter((F.col("prediction") == 0) & (F.col(LABEL_COL) == 0)).count()
    fn = predictions.filter((F.col("prediction") == 0) & (F.col(LABEL_COL) == 1)).count()

    fraud_precision = tp / (tp + fp + 1e-10)
    fraud_recall    = tp / (tp + fn + 1e-10)
    fraud_f1        = 2 * fraud_precision * fraud_recall / (fraud_precision + fraud_recall + 1e-10)

    results = {
        "model":           model_name,
        "auc_roc":         round(auc_roc,         4),
        "auc_pr":          round(auc_pr,           4),
        "accuracy":        round(accuracy,         4),
        "weighted_precision": round(precision,     4),
        "weighted_recall": round(recall,           4),
        "f1_weighted":     round(f1,               4),
        "f2_score":        round(f2,               4),
        "fraud_precision": round(fraud_precision,  4),
        "fraud_recall":    round(fraud_recall,     4),
        "fraud_f1":        round(fraud_f1,         4),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn
    }

    print(f"\n  ─── {model_name} Results ───")
    print(f"  AUC-ROC           : {auc_roc:.4f}")
    print(f"  AUC-PR            : {auc_pr:.4f}")
    print(f"  Accuracy          : {accuracy:.4f}")
    print(f"  Fraud Precision   : {fraud_precision:.4f}")
    print(f"  Fraud Recall      : {fraud_recall:.4f}  ← primary metric")
    print(f"  Fraud F1          : {fraud_f1:.4f}")
    print(f"  F2 Score          : {f2:.4f}  (recall-weighted)")
    print(f"  Confusion Matrix  : TP={tp}  FP={fp}  TN={tn:,}  FN={fn}")

    return results


# =============================================================================
# 4a. MODEL 1: LOGISTIC REGRESSION (OPTIMIZED)
# =============================================================================
def train_logistic_regression(train_df, test_df) -> dict:
    """
    Logistic Regression — Baseline Model (OPTIMIZED)
    
    OPTIMIZATIONS:
    - 3 folds instead of 5 (40% faster)
    - 6 hyperparameter combinations instead of 12 (50% reduction)
    - Parallelism reduced to 2 for stability
    """
    print(f"\n{'─'*60}")
    print("  4a.  Model 1: Logistic Regression (OPTIMIZED)")
    print(f"{'─'*60}")

    lr = LogisticRegression(
        featuresCol=FEATURE_COL,
        labelCol=LABEL_COL,
        family="binomial",
        standardization=False,
        tol=1e-6
    )

    param_grid = (
        ParamGridBuilder()
        .addGrid(lr.regParam,          [0.01, 0.1])       
        .addGrid(lr.elasticNetParam,   [0.0, 0.5, 1.0])  
        .build()
    )

    evaluator = BinaryClassificationEvaluator(
        labelCol=LABEL_COL,
        metricName="areaUnderROC"
    )

    cv = CrossValidator(
        estimator=lr,
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        numFolds=N_FOLDS,         
        seed=RANDOM_SEED,
        parallelism=2,            
        collectSubModels=False
    )

    print(f"  Training with {N_FOLDS}-fold CV, grid size={len(param_grid)}")
    print(f"  Total models: {len(param_grid) * N_FOLDS} = 18 (was 60)")
    print(f"  Estimated time: 4-6 minutes...")
    
    t0 = time.time()
    cv_model = cv.fit(train_df)
    elapsed = time.time() - t0

    best_lr     = cv_model.bestModel
    best_params = {
        "regParam":        best_lr.getRegParam(),
        "elasticNetParam": best_lr.getElasticNetParam(),
        "maxIter":         best_lr.getMaxIter(),
    }
    print(f"  Best params : {best_params}")
    print(f"  Training time : {elapsed:.1f}s ({elapsed/60:.1f} min)")

    safe_model_save(best_lr, f"{MODELS_DIR}/logistic_regression", "LogisticRegression")

    predictions = best_lr.transform(test_df)
    results     = evaluate_model(predictions, "Logistic Regression")
    results["training_time_s"] = round(elapsed, 1)
    results["best_params"]     = best_params

    coefs      = best_lr.coefficients.toArray()
    feat_names = [f"V{i}" for i in range(1, 29)] + ["Amount_scaled", "Time_scaled"]
    coef_df    = pd.DataFrame({
        "feature":    feat_names,
        "coefficient": coefs,
        "abs_coef":   np.abs(coefs)
    }).sort_values("abs_coef", ascending=False)

    print("\n  Top 10 features by |coefficient|:")
    print(coef_df.head(10).to_string(index=False))

    coef_df.to_csv(f"{OUTPUT_DIR}/lr_coefficients.csv", index=False)
    results["lr_coefficients"] = coef_df.head(15).to_dict()

    return results, predictions, coef_df


# =============================================================================
# 4b. MODEL 2: RANDOM FOREST (OPTIMIZED)
# =============================================================================
def train_random_forest(train_df, test_df) -> dict:
    """
    Random Forest — Primary Model (OPTIMIZED)
    
    OPTIMIZATIONS:
    - 3 folds instead of 5
    - 9 hyperparameter combinations instead of 18
    - Reduced tree counts for faster training
    """
    print(f"\n{'─'*60}")
    print("  4b.  Model 2: Random Forest Classifier (OPTIMIZED)")
    print(f"{'─'*60}")

    rf = RandomForestClassifier(
        featuresCol=FEATURE_COL,
        labelCol=LABEL_COL,
        seed=RANDOM_SEED,
        featureSubsetStrategy="sqrt"
    )

    param_grid = (
        ParamGridBuilder()
        .addGrid(rf.numTrees,               [50, 100])     
        .addGrid(rf.maxDepth,               [8, 12])       
        .addGrid(rf.minInstancesPerNode,    [1, 5])      
        .build()
    )

    evaluator = BinaryClassificationEvaluator(
        labelCol=LABEL_COL,
        metricName="areaUnderROC"
    )

    cv = CrossValidator(
        estimator=rf,
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        numFolds=N_FOLDS,
        seed=RANDOM_SEED,
        parallelism=2,
        collectSubModels=False
    )

    print(f"  Training with {N_FOLDS}-fold CV, grid size={len(param_grid)}")
    print(f"  Total models: {len(param_grid) * N_FOLDS} = {len(param_grid) * N_FOLDS}")
    print(f"  Estimated time: 5-8 minutes...")
    
    t0 = time.time()
    cv_model = cv.fit(train_df)
    elapsed  = time.time() - t0

    best_rf     = cv_model.bestModel
    best_params = {
        "numTrees":            best_rf.getNumTrees,
        "maxDepth":            best_rf.getMaxDepth(),
        "minInstancesPerNode": best_rf.getMinInstancesPerNode(),
    }
    print(f"  Best params  : {best_params}")
    print(f"  Training time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    safe_model_save(best_rf, f"{MODELS_DIR}/random_forest", "RandomForest")

    predictions = best_rf.transform(test_df)
    results     = evaluate_model(predictions, "Random Forest")
    results["training_time_s"] = round(elapsed, 1)
    results["best_params"]     = best_params

    feat_names  = [f"V{i}" for i in range(1, 29)] + ["Amount_scaled", "Time_scaled"]
    importances = best_rf.featureImportances.toArray()
    fi_df = pd.DataFrame({
        "feature":    feat_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    print("\n  Top 10 features by Gini importance:")
    print(fi_df.head(10).to_string(index=False))

    fi_df.to_csv(f"{OUTPUT_DIR}/rf_feature_importances.csv", index=False)
    results["feature_importances"] = fi_df.head(15).to_dict()

    return results, predictions, fi_df


# =============================================================================
# 4c. MODEL 3: GRADIENT BOOSTED TREES (OPTIMIZED)
# =============================================================================
def train_gbt(train_df, test_df) -> dict:
    """
    GBT Classifier (OPTIMIZED)
    
    OPTIMIZATIONS:
    - 3 folds instead of 5
    - 6 hyperparameter combinations instead of 18
    - Reduced maxIter for faster training
    """
    print(f"\n{'─'*60}")
    print("  4c.  Model 3: Gradient Boosted Trees (OPTIMIZED)")
    print(f"{'─'*60}")

    gbt = GBTClassifier(
        featuresCol=FEATURE_COL,
        labelCol=LABEL_COL,
        seed=RANDOM_SEED,
        lossType="logistic"
    )

    param_grid = (
        ParamGridBuilder()
        .addGrid(gbt.maxIter,   [50])         
        .addGrid(gbt.maxDepth,  [4, 6])        
        .addGrid(gbt.stepSize,  [0.1, 0.2, 0.3])
        .build()
    )

    evaluator = BinaryClassificationEvaluator(
        labelCol=LABEL_COL,
        metricName="areaUnderROC"
    )

    cv = CrossValidator(
        estimator=gbt,
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        numFolds=N_FOLDS,
        seed=RANDOM_SEED,
        parallelism=1,             
        collectSubModels=False
    )

    print(f"  Training with {N_FOLDS}-fold CV, grid size={len(param_grid)}")
    print(f"  Total models: {len(param_grid) * N_FOLDS} = {len(param_grid) * N_FOLDS}")
    print(f"  Estimated time: 8-12 minutes...")
    
    t0 = time.time()
    cv_model = cv.fit(train_df)
    elapsed  = time.time() - t0

    best_gbt    = cv_model.bestModel
    best_params = {
        "maxIter":  best_gbt.getMaxIter(),
        "maxDepth": best_gbt.getMaxDepth(),
        "stepSize": best_gbt.getStepSize(),
    }
    print(f"  Best params  : {best_params}")
    print(f"  Training time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    safe_model_save(best_gbt, f"{MODELS_DIR}/gbt", "GBT")

    predictions = best_gbt.transform(test_df)
    results     = evaluate_model(predictions, "GBT")
    results["training_time_s"] = round(elapsed, 1)
    results["best_params"]     = best_params


    feat_names  = [f"V{i}" for i in range(1, 29)] + ["Amount_scaled", "Time_scaled"]
    importances = best_gbt.featureImportances.toArray()
    fi_df = pd.DataFrame({
        "feature":    feat_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    print("\n  Top 10 features by GBT importance:")
    print(fi_df.head(10).to_string(index=False))

    fi_df.to_csv(f"{OUTPUT_DIR}/gbt_feature_importances.csv", index=False)
    results["feature_importances"] = fi_df.head(15).to_dict()

    return results, predictions, fi_df


# =============================================================================
# 5. COLLECT ROC/PR CURVE DATA
# =============================================================================
def extract_roc_pr_data(predictions, model_name: str, output_dir: str):
    """Extract probability scores for ROC and PR curve plotting."""
    from pyspark.ml.functions import vector_to_array

    prob_df = (
        predictions
        .withColumn("prob_fraud", vector_to_array(F.col("probability"))[1])
        .select("prob_fraud", LABEL_COL)
        .toPandas()
    )

    safe_name = model_name.replace(" ", "_").lower()
    prob_df.to_csv(f"{output_dir}/{safe_name}_probs.csv", index=False)
    print(f"  [Saved] {safe_name}_probs.csv  ({len(prob_df):,} rows)")

    return prob_df


# =============================================================================
# 6. SAVE ALL RESULTS
# =============================================================================
def save_results(all_results: list):
    """Save model comparison results as JSON for Task 5 reporting."""
    path = f"{OUTPUT_DIR}/model_results.json"
    serialisable = []
    for r in all_results:
        sr = {k: v for k, v in r.items()
              if isinstance(v, (int, float, str, dict, list))}
        serialisable.append(sr)

    with open(path, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"\n  [Saved] {path}")

    summary_cols = [
        "model", "auc_roc", "auc_pr", "accuracy",
        "fraud_precision", "fraud_recall", "fraud_f1",
        "f2_score", "tp", "fp", "tn", "fn", "training_time_s"
    ]
    summary_df = pd.DataFrame([
        {k: r.get(k, "") for k in summary_cols} for r in all_results
    ])
    csv_path = f"{OUTPUT_DIR}/model_comparison.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"  [Saved] {csv_path}")

    return summary_df


# =============================================================================
# 7. PRINT COMPARISON TABLE
# =============================================================================
def print_comparison_table(summary_df):
    print(f"\n{'='*80}")
    print("  TASK 4 — MODEL COMPARISON SUMMARY (OPTIMIZED)")
    print(f"{'='*80}")
    cols = ["model", "auc_roc", "auc_pr", "fraud_recall", "fraud_f1", "f2_score"]
    print(summary_df[cols].to_string(index=False))
    print(f"{'='*80}")

    best_roc    = summary_df.loc[summary_df["auc_roc"].idxmax(),    "model"]
    best_recall = summary_df.loc[summary_df["fraud_recall"].idxmax(), "model"]
    print(f"\n  Best AUC-ROC   : {best_roc}")
    print(f"  Best Recall    : {best_recall}  ← recommended for fraud detection")
    print(f"\n  Results saved to: {OUTPUT_DIR}/")
    print(f"  Probability CSVs → {OUTPUT_DIR}/  (used by Task 5)\n")


# =============================================================================
# MAIN
# =============================================================================
def main():
    spark = create_spark_session(APP_NAME)

    try:
        train_df, test_df = load_data(spark)

        print("\n" + "="*60)
        print("  OPTIMIZATION SUMMARY:")
        print("  - CV Folds: 5 → 3 (40% faster)")
        print("  - LR Grid: 12 → 6 combinations (50% reduction)")
        print("  - RF Grid: 18 → 8 combinations (56% reduction)")
        print("  - GBT Grid: 18 → 6 combinations (67% reduction)")
        print("  - Total estimated time: 15-25 minutes (all 3 models)")
        print("="*60 + "\n")

        lr_results,  lr_preds,  lr_coefs  = train_logistic_regression(train_df, test_df)
        rf_results,  rf_preds,  rf_fi     = train_random_forest(train_df, test_df)
        gbt_results, gbt_preds, gbt_fi    = train_gbt(train_df, test_df)

        extract_roc_pr_data(lr_preds,  "Logistic Regression", OUTPUT_DIR)
        extract_roc_pr_data(rf_preds,  "Random Forest",       OUTPUT_DIR)
        extract_roc_pr_data(gbt_preds, "GBT",                 OUTPUT_DIR)

        all_results = [lr_results, rf_results, gbt_results]
        summary_df  = save_results(all_results)
        print_comparison_table(summary_df)

    finally:
        spark.stop()
        print("  Spark session stopped.\n")


if __name__ == "__main__":
    main()