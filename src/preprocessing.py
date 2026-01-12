from pyspark.sql.functions import col, to_date, to_timestamp, year, month, weekofyear, avg, lit

def load_and_clean_data(spark, data_dir, config):
    """
    Full Phase 2 & 3 pipeline: Ingestion -> Cleaning -> Filtering -> Indices.
    """
    
    # 1. Define Schema
    column_names = ["id", "cp", "pop", "lat", "long", "date", "type", "name", "prix"]

    # 2. Read Data
    df_raw = spark.read.format("csv") \
        .option("header", "false") \
        .option("sep", ";") \
        .option("inferSchema", "true") \
        .load(f"{data_dir}/Prix*.csv.gz") \
        .toDF(*column_names)

    # 3. Clean & Feature Engineering
    df_prepared = df_raw \
        .withColumn("date_parsed", to_timestamp(col("date"), "yyyy-MM-dd HH:mm:ss")) \
        .withColumn("day_date", to_date(col("date_parsed"))) \
        .withColumn("year", year(col("date_parsed"))) \
        .withColumn("month", month(col("date_parsed"))) \
        .withColumn("week", weekofyear(col("date_parsed"))) \
        .withColumn("lat", col("lat") / 100000) \
        .withColumn("long", col("long") / 100000) \
        .withColumn("prix", col("prix").cast("double")) \
        .filter(col("date_parsed").isNotNull())
    
    return df_prepared
















'''
    # 4. Filter Low Interest Gas Types
    gas_counts = df_prepared.groupBy("name").count().orderBy("count")
    least_frequent_rows = gas_counts.take(2) 
    low_interest_types = [row['name'] for row in least_frequent_rows]
    
    print(f"Filtering out: {low_interest_types}")
    
    df_filtered = df_prepared.filter(~col("name").isin(low_interest_types))

    # 5. Calculate Indices
    
    # A. Price Index
    avg_france = df_filtered.groupBy("day_date", "name") \
        .agg(avg("prix").alias("avg_day_price_france"))
    
    df_indexed = df_filtered.join(avg_france, on=["day_date", "name"], how="left")
    
    df_indexed = df_indexed.withColumn("price_index", 
        100 * ((col("prix") - col("avg_day_price_france")) / col("avg_day_price_france") + 1))

    # B. Week Index
    start_year = config['data_settings']['first_year']
    df_final = df_indexed.withColumn("week_index", 
                                     (col("year") - start_year) * 52 + col("week"))

    return df_final
'''