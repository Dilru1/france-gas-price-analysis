from pyspark.sql import SparkSession
import os
import sys

def get_spark_session(app_name="GasPriceProject"):
    spark = SparkSession.builder \
        .master("local[*]") \
        .appName(app_name) \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
    return spark