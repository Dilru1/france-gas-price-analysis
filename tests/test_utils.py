from src.utils import get_spark_session
from pyspark.sql import SparkSession

def test_get_spark_session():
    """
    Verifies that the utility function returns a valid SparkSession.
    """
    spark = get_spark_session("TestApp")
    
    assert isinstance(spark, SparkSession)
    assert spark.sparkContext.appName == "TestApp"
    # Note: We don't stop it here to avoid killing the session for other tests if running in parallel