import pytest
from pyspark.sql import SparkSession

@pytest.fixture(scope="session")
def spark():
    """
    Creates a single SparkSession for the entire test suite.
    """
    spark = SparkSession.builder \
        .master("local[1]") \
        .appName("GasPriceTest") \
        .getOrCreate()
    
    yield spark
    
    # Teardown (Optional, Spark handles this mostly)
    spark.stop()