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
