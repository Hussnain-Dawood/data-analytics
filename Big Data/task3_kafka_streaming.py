import os
import sys
import json
import time
import logging
import threading
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

os.environ["JAVA_HOME"]  = r"C:\Program Files\Eclipse Adoptium\jdk-21.0.8.9-hotspot"
os.environ["HADOOP_HOME"] = r"D:\Big Data \hadoop"
os.environ["PATH"] += ";" + os.path.join(os.environ["HADOOP_HOME"], "bin")
os.environ["PYSPARK_SUBMIT_ARGS"] = (
    "--driver-java-options '--add-modules=jdk.incubator.vector' pyspark-shell"
)

print("Winutils Exists:", os.path.exists(
    os.path.join(os.environ["HADOOP_HOME"], "bin", "winutils.exe")
))

DATA_PATH      = "creditcard.csv"
KAFKA_BROKER   = "localhost:9092"
TOPIC_NAME     = "credit-transactions"
CHECKPOINT_DIR = "task3_checkpoint"
OUTPUT_DIR     = "task3_outputs"

os.makedirs(OUTPUT_DIR,     exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("Task3")

def _pyspark_version() -> str:
    import pyspark
    return pyspark.__version__       


def _kafka_maven_pkg(spark_ver: str) -> str:
    """
    Return the correct Maven coordinate for spark-sql-kafka.
    PySpark 4.x  →  Scala 2.13  (_2.13 artifact)
    PySpark 3.x  →  Scala 2.12  (_2.12 artifact)
    Using Maven packages avoids the manual JAR version-mismatch problem.
    """
    major = int(spark_ver.split(".")[0])
    scala = "2.13" if major >= 4 else "2.12"
    return f"org.apache.spark:spark-sql-kafka-0-10_{scala}:{spark_ver}"


def _check_kafka(host: str = "localhost", port: int = 9092) -> bool:
    """Return True if a Kafka broker is reachable, False otherwise."""
    import socket
    try:
        with socket.create_connection((host, port), timeout=3):
            return True
    except (ConnectionRefusedError, OSError):
        return False

def build_spark(app_name: str = "FraudDetection", need_kafka: bool = False):
    """
    Build and return a SparkSession.

    Parameters
    ----------
    app_name   : Displayed in the Spark UI.
    need_kafka : If True, adds the correct Kafka Maven package for the
                 detected PySpark version.  Requires internet on first run.
    """
    from pyspark.sql import SparkSession

    ver = _pyspark_version()
    print(f"   PySpark version : {ver}")

    builder = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .config("spark.hadoop.io.native.lib.available", "false")
        .config("spark.sql.debug.maxToStringFields", "100")
    )

    if need_kafka:
        pkg = _kafka_maven_pkg(ver)
        print(f"   Kafka package   : {pkg}")
        builder = builder.config("spark.jars.packages", pkg)

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    try:
        log4j = spark.sparkContext._jvm.org.apache.log4j
        log4j.Logger.getLogger("org.apache.spark.SparkEnv")\
                    .setLevel(log4j.Level.OFF)
        log4j.Logger.getLogger("org.apache.spark.network.util.JavaUtils")\
                    .setLevel(log4j.Level.OFF)
    except Exception:
        pass

    print(" Spark Session started successfully.")
    return spark

def apply_rule_score(df):
    """
    Rule-based fraud scoring applied to both batch and streaming DataFrames.

    Rules (based on EDA findings from Task 1):
      HIGH_V14     : |V14| > 10  — strongest negative correlate with fraud
      HIGH_V12     : |V12| > 10  — second strongest negative correlate
      MICRO_AMOUNT : Amount < 1  — very small amounts common in fraud probing
      NORMAL       : everything else
    """
    from pyspark.sql import functions as F
    return (
        df.withColumn(
            "RuleFlag",
            F.when(F.abs(F.col("V14")) > 10, "HIGH_V14")
             .when(F.abs(F.col("V12")) > 10, "HIGH_V12")
             .when(F.col("Amount") < 1,       "MICRO_AMOUNT")
             .otherwise("NORMAL"),
        ).withColumn(
            "AlertLevel",
            F.when(F.col("RuleFlag").isin("HIGH_V14", "HIGH_V12"), "HIGH")
             .when(F.col("RuleFlag") == "MICRO_AMOUNT",             "MEDIUM")
             .otherwise("NORMAL"),
        )
    )

class CreditCardProducer:
    """
    Reads creditcard.csv row-by-row and sends each transaction as a
    JSON message to the Kafka topic 'credit-transactions'.

    Prerequisite: Kafka broker running at localhost:9092
    Run with   : python task3_kafka_streaming.py --mode producer
    """

    def __init__(self, broker: str, topic: str):
        if not _check_kafka():
            print("\nKAFKA NOT RUNNING — cannot start producer.")
            print("    Start ZooKeeper + Kafka first, then retry.\n")
            print("    Quick-start (WSL / Linux):")
            print("      bin/zookeeper-server-start.sh config/zookeeper.properties &")
            print("      bin/kafka-server-start.sh config/server.properties &\n")
            sys.exit(1)

        self.broker = broker
        self.topic  = topic
        self._create_topic()

        from kafka import KafkaProducer as _KP
        self.producer = _KP(
            bootstrap_servers=[broker],
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )

    def _create_topic(self):
        from kafka.admin import KafkaAdminClient, NewTopic
        from kafka.errors import TopicAlreadyExistsError
        try:
            admin = KafkaAdminClient(bootstrap_servers=[self.broker])
            admin.create_topics([
                NewTopic(name=self.topic, num_partitions=3, replication_factor=1)
            ])
            log.info(f"Topic '{self.topic}' created.")
        except TopicAlreadyExistsError:
            log.info(f"Topic '{self.topic}' already exists — reusing.")

    def produce(self):
        import csv
        log.info(f"Streaming {DATA_PATH} → Kafka topic '{self.topic}' …")
        with open(DATA_PATH, "r") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                record = {
                    "transaction_id": i,
                    "timestamp": datetime.utcnow().isoformat(),
                    "Time":   float(row["Time"]),
                    "Amount": float(row["Amount"]),
                    "Class":  int(row["Class"]),
                    **{f"V{j}": float(row[f"V{j}"]) for j in range(1, 29)},
                }
                self.producer.send(self.topic, value=record)
                if i % 1000 == 0:
                    log.info(f"  Sent {i:,} records …")
                time.sleep(0.01)
        self.producer.flush()
        log.info("Producer finished.")

class FraudConsumer:
    """
    Spark Structured Streaming consumer.
    Reads JSON messages from Kafka, scores them with apply_rule_score(),
    and writes:
      • Suspicious alerts → console (every 5 s)
      • Full audit log    → CSV in task3_outputs/stream_csv/ (every 10 s)

    Prerequisite: Kafka running + topic receiving data from producer
    Run with   : python task3_kafka_streaming.py --mode consumer
    """

    def __init__(self):
        if not _check_kafka():
            print("\n  KAFKA NOT RUNNING — cannot start consumer.")
            sys.exit(1)
        self.spark = build_spark("FraudDetection-Consumer", need_kafka=True)

    @staticmethod
    def _schema():
        from pyspark.sql.types import (
            StructType, StructField, LongType, StringType,
            DoubleType, IntegerType,
        )
        return StructType(
            [StructField("transaction_id", LongType()),
             StructField("timestamp",      StringType()),
             StructField("Time",           DoubleType()),
             StructField("Amount",         DoubleType()),
             StructField("Class",          IntegerType())]
            + [StructField(f"V{i}", DoubleType()) for i in range(1, 29)]
        )

    def run(self):
        from pyspark.sql import functions as F

        print("\n" + "=" * 60)
        print("  Spark Structured Streaming — Fraud Detection Consumer")
        print(f"  Kafka topic : {TOPIC_NAME}")
        print("=" * 60 + "\n")

        raw = (
            self.spark.readStream
            .format("kafka")
            .option("kafka.bootstrap.servers", KAFKA_BROKER)
            .option("subscribe",       TOPIC_NAME)
            .option("startingOffsets", "latest")
            .load()
        )

        parsed = raw.select(
            F.from_json(F.col("value").cast("string"), self._schema()).alias("d")
        ).select("d.*")

        scored = apply_rule_score(parsed)

        q_alerts = (
            scored.filter(F.col("RuleFlag") != "NORMAL")
            .select("transaction_id", "timestamp", "Amount",
                    "RuleFlag", "AlertLevel", "Class")
            .writeStream
            .format("console")
            .option("truncate", False)
            .option("numRows",  20)
            .trigger(processingTime="5 seconds")
            .start()
        )

        q_audit = (
            scored.writeStream
            .format("csv")
            .option("path",               OUTPUT_DIR + "/stream_csv")
            .option("checkpointLocation", CHECKPOINT_DIR + "/stream_csv")
            .option("header", True)
            .trigger(processingTime="10 seconds")
            .start()
        )

        print("Streaming active. Press Ctrl+C to stop.\n")
        try:
            q_alerts.awaitTermination()
            q_audit.awaitTermination()
        except KeyboardInterrupt:
            print("\n⏹  Stopped by user.")
            q_alerts.stop()
            q_audit.stop()
        finally:
            self.spark.stop()

class BatchDemo:
    """
    Simulates the Kafka → Spark pipeline using a static batch CSV read.
    No Kafka installation or running broker is needed.

    Output:
      1. Rule-Flag × Class distribution table
      2. Amount statistics per flag
      3. Precision summary (how many flagged = actual fraud)
      4. Parquet file written via PyArrow (avoids Windows file-lock issue)

    Run with:  python task3_kafka_streaming.py --mode demo  [DEFAULT]
    """

    def __init__(self):
        self.spark = build_spark("FraudDetection-BatchDemo", need_kafka=False)

    def run(self):
        from pyspark.sql import functions as F

        print("\n" + "=" * 60)
        print("  BATCH DEMO — Simulated Kafka Streaming Pipeline")
        print("  (No Kafka broker required)")
        print("=" * 60 + "\n")

        df = (
            self.spark.read
            .option("header",      "true")
            .option("inferSchema", "true")
            .csv(DATA_PATH)
            .limit(2000)
        )
        total = df.count()
        print(f"Loaded {total:,} transactions from: {DATA_PATH}\n")

        scored = apply_rule_score(df)

        print("── Rule-Flag Distribution (Class 0=Genuine, 1=Fraud) ──")
        scored.groupBy("RuleFlag", "AlertLevel", "Class") \
              .count() \
              .orderBy("RuleFlag", "Class") \
              .show(truncate=False)

        print("── Transaction Amount Statistics per Flag ──")
        scored.groupBy("RuleFlag").agg(
            F.count("Amount").alias("Count"),
            F.round(F.mean("Amount"), 2).alias("Mean_EUR"),
            F.round(F.min("Amount"),  2).alias("Min_EUR"),
            F.round(F.max("Amount"),  2).alias("Max_EUR"),
        ).orderBy("RuleFlag").show(truncate=False)

        sus        = scored.filter(F.col("RuleFlag") != "NORMAL")
        sus_count  = sus.count()
        fraud_sus  = sus.filter(F.col("Class") == 1).count()
        sus_pct    = sus_count / total * 100 if total else 0
        precision  = fraud_sus / sus_count * 100 if sus_count else 0

        print(f"Suspicious flagged          : {sus_count} ({sus_pct:.1f}%)")
        print(f"   Actual fraud in flagged     : {fraud_sus}")
        print(f"   Rule precision              : {precision:.1f}%\n")

        out_dir = os.path.join(OUTPUT_DIR, "batch")
        os.makedirs(out_dir, exist_ok=True)

        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
            pandas_df = scored.toPandas()
            out_file  = os.path.join(out_dir, "batch_scored.parquet")
            pq.write_table(pa.Table.from_pandas(pandas_df), out_file)
            print(f"Parquet saved → {out_file}")
        except Exception as e:
            print(f"PyArrow failed ({e}). Falling back to CSV …")
            csv_out = os.path.join(out_dir, "batch_scored_csv")
            scored.write.mode("overwrite").option("header", True).csv(csv_out)
            print(f"CSV saved → {csv_out}")

        print("\nBatch demo complete. Stopping Spark …")
        self.spark.stop()
        print("Done.\n")

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Task 3 — Kafka + Spark Fraud Detection Pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--mode", default="demo",
        choices=["producer", "consumer", "demo", "both"],
        help=(
            "demo     : Batch simulation — NO Kafka needed [DEFAULT]\n"
            "producer : Stream CSV rows → Kafka  (Kafka must be running)\n"
            "consumer : Spark Structured Streaming consumer from Kafka\n"
            "both     : Start producer + consumer simultaneously"
        ),
    )
    args = parser.parse_args()

    if   args.mode == "producer":
        CreditCardProducer(KAFKA_BROKER, TOPIC_NAME).produce()
    elif args.mode == "consumer":
        FraudConsumer().run()
    elif args.mode == "demo":
        BatchDemo().run()
    elif args.mode == "both":
        consumer = FraudConsumer()
        def _produce():
            time.sleep(4)
            CreditCardProducer(KAFKA_BROKER, TOPIC_NAME).produce()
        threading.Thread(target=_produce, daemon=True).start()
        consumer.run()


if __name__ == "__main__":
    main()