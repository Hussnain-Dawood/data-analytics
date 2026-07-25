"""
ECS765P - Big Data Processing
Task 4: Structured Streaming with Smoke Detection IoT Data
Author: [Your Name]

Run on cluster ONLY:
    spark-submit task4.py

The streaming server sends IoT sensor data line by line over a socket.
Host: stream.comp-teach.qmul.ac.uk
Port: 9999
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType,
    FloatType, IntegerType, TimestampType
)

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────
STREAM_HOST  = "stream.comp-teach.qmul.ac.uk"
STREAM_PORT  = 9999
OUTPUT_DIR   = "s3a://YOUR-BUCKET-NAME/task4"   # replace with your bucket
CHECKPOINT   = "s3a://YOUR-BUCKET-NAME/task4/checkpoints"

# ─────────────────────────────────────────────────────────────────
# SPARK SESSION
# ─────────────────────────────────────────────────────────────────
spark = SparkSession.builder \
    .appName("ECS765P_Task4_SmokeDetection") \
    .config("spark.sql.shuffle.partitions", "4") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

print("\n" + "=" * 60)
print("  ECS765P Task 4: Smoke Detection IoT Streaming")
print("=" * 60 + "\n")

# ─────────────────────────────────────────────────────────────────
# Q1. READ STREAMING DATA FROM SOCKET  (3 points)
# ─────────────────────────────────────────────────────────────────
print("Q1. Reading stream from socket...")

# Schema of incoming CSV data
schema = StructType([
    StructField("timestamp",     StringType(),  True),
    StructField("temperature",   FloatType(),   True),
    StructField("humidity",      FloatType(),   True),
    StructField("TVOC",          IntegerType(), True),
    StructField("eCO2",          IntegerType(), True),
    StructField("raw_h2",        IntegerType(), True),
    StructField("raw_ethanol",   IntegerType(), True),
    StructField("pressure",      FloatType(),   True),
    StructField("pm1_0",         FloatType(),   True),
    StructField("pm2_5",         FloatType(),   True),
    StructField("nc0_5",         FloatType(),   True),
    StructField("nc1_0",         FloatType(),   True),
    StructField("nc2_5",         FloatType(),   True),
    StructField("cnt",           IntegerType(), True),
    StructField("fire_alarm",    IntegerType(), True),
])

# Read raw lines from socket
raw_stream = spark.readStream \
    .format("socket") \
    .option("host", STREAM_HOST) \
    .option("port", STREAM_PORT) \
    .load()

# Parse CSV lines into structured columns
parsed_stream = raw_stream.select(
    F.from_csv(F.col("value"), schema).alias("data")
).select("data.*")

# Convert timestamp string to TimestampType
parsed_stream = parsed_stream.withColumn(
    "timestamp", F.to_timestamp(F.col("timestamp"))
)

# ─────────────────────────────────────────────────────────────────
# Q2. REAL-TIME SUMMARY STATISTICS  (4 points)
# ─────────────────────────────────────────────────────────────────
print("Q2. Setting up real-time summary statistics...")

# Add watermark for late data (5 minute tolerance)
watermarked = parsed_stream.withWatermark("timestamp", "5 minutes")

# Sliding window: 10 min window, 5 min slide
summary_stats = watermarked.groupBy(
    F.window("timestamp", "10 minutes", "5 minutes")
).agg(
    F.count("*").alias("record_count"),
    F.round(F.avg("temperature"),   2).alias("avg_temperature"),
    F.round(F.avg("humidity"),      2).alias("avg_humidity"),
    F.round(F.avg("pm2_5"),         2).alias("avg_pm2_5"),
    F.round(F.avg("eCO2"),          2).alias("avg_eCO2"),
    F.round(F.max("temperature"),   2).alias("max_temperature"),
    F.round(F.min("temperature"),   2).alias("min_temperature"),
    F.sum("fire_alarm").alias("fire_alarm_count")
)

# ─────────────────────────────────────────────────────────────────
# Q3. FIRE ALARM DETECTION  (4 points)
# ─────────────────────────────────────────────────────────────────
print("Q3. Setting up fire alarm detection...")

# Filter only records where fire_alarm = 1
fire_events = parsed_stream.filter(F.col("fire_alarm") == 1) \
    .select(
        "timestamp", "temperature", "humidity",
        "pm2_5", "eCO2", "TVOC", "fire_alarm"
    )

# High-risk detection: high temp AND high CO2 AND high PM2.5
high_risk = parsed_stream.filter(
    (F.col("temperature") > 50) &
    (F.col("eCO2") > 1000) &
    (F.col("pm2_5") > 50)
).select(
    "timestamp", "temperature", "eCO2", "pm2_5",
    "TVOC", "humidity", "fire_alarm"
).withColumn("risk_level", F.lit("HIGH"))

# ─────────────────────────────────────────────────────────────────
# Q4. TUMBLING WINDOW — FIRE ALARM RATE  (4 points)
# ─────────────────────────────────────────────────────────────────
print("Q4. Setting up tumbling window fire alarm rate...")

# 1-minute tumbling window
alarm_rate = watermarked.groupBy(
    F.window("timestamp", "1 minute")
).agg(
    F.count("*").alias("total_records"),
    F.sum("fire_alarm").alias("fire_alarms"),
    F.round(
        F.sum("fire_alarm").cast("double") / F.count("*"), 4
    ).alias("alarm_rate"),
    F.round(F.avg("temperature"), 2).alias("avg_temp"),
    F.round(F.avg("pm2_5"),       2).alias("avg_pm2_5")
)

# ─────────────────────────────────────────────────────────────────
# START STREAMING QUERIES
# ─────────────────────────────────────────────────────────────────
print("\nStarting streaming queries...\n")

# Q2 — Summary stats to CSV
q2_query = summary_stats.writeStream \
    .outputMode("append") \
    .format("csv") \
    .option("path",           f"{OUTPUT_DIR}/q2_summary") \
    .option("checkpointLocation", f"{CHECKPOINT}/q2") \
    .option("header", "true") \
    .trigger(processingTime="30 seconds") \
    .start()

print("Q2 stream started — writing summary stats to bucket.")

# Q3a — Fire events to CSV
q3a_query = fire_events.writeStream \
    .outputMode("append") \
    .format("csv") \
    .option("path",           f"{OUTPUT_DIR}/q3_fire_events") \
    .option("checkpointLocation", f"{CHECKPOINT}/q3a") \
    .option("header", "true") \
    .trigger(processingTime="30 seconds") \
    .start()

print("Q3a stream started — writing fire events to bucket.")

# Q3b — High risk events to CSV
q3b_query = high_risk.writeStream \
    .outputMode("append") \
    .format("csv") \
    .option("path",           f"{OUTPUT_DIR}/q3_high_risk") \
    .option("checkpointLocation", f"{CHECKPOINT}/q3b") \
    .option("header", "true") \
    .trigger(processingTime="30 seconds") \
    .start()

print("Q3b stream started — writing high-risk events to bucket.")

# Q4 — Alarm rate (console for monitoring)
q4_query = alarm_rate.writeStream \
    .outputMode("append") \
    .format("csv") \
    .option("path",           f"{OUTPUT_DIR}/q4_alarm_rate") \
    .option("checkpointLocation", f"{CHECKPOINT}/q4") \
    .option("header", "true") \
    .trigger(processingTime="60 seconds") \
    .start()

print("Q4 stream started — writing alarm rate to bucket.")

# Also print Q2 to console for monitoring
console_query = summary_stats.writeStream \
    .outputMode("complete") \
    .format("console") \
    .option("truncate", "false") \
    .trigger(processingTime="30 seconds") \
    .start()

print("\nAll streams running. Waiting for data...\n")
print("Press Ctrl+C to stop.\n")

# Wait for all streams to finish
spark.streams.awaitAnyTermination()
