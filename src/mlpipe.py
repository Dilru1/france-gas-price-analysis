# List of variables to clean up
vars_to_clean = ['train_data', 'test_data', 'df_features', 'df_model_ready', 'lr_model', 'rf_model', 'gbt_model']

print("Cleaning up previous variables...")

for var in vars_to_clean:
    if var in globals():
        # 1. If it's a Spark DataFrame, unpersist it to free up RAM
        try:
            if hasattr(globals()[var], "unpersist"):
                globals()[var].unpersist()
        except:
            pass
        
        # 2. Delete the Python variable
        del globals()[var]

print("Environment clean. Ready to rebuild Section 5.")


# --- 1. Re-Create Features ---
# Define Window
window_spec = Window.partitionBy("id", "name").orderBy("day_date")

# Create Lag & Rolling Features
df_features = df_ml \
    .withColumn("price_lag_1", lag("prix", 1).over(window_spec)) \
    .withColumn("price_lag_7", lag("prix", 7).over(window_spec)) \
    .withColumn("rolling_avg_7", avg("prix").over(window_spec.rowsBetween(-7, -1))) \
    .withColumn("day_of_week", dayofweek("day_date")) \
    .withColumn("month", month("day_date")) \
    .dropna() 

# Assemble Vectors
feature_cols = ["price_lag_1", "price_lag_7", "rolling_avg_7", "day_of_week", "month", "lat", "long"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df_model_ready = assembler.transform(df_features)

# --- 2. Re-Split Data (Chronological) ---
# Dynamic 80/20 split
split_timestamp = df_model_ready.withColumn("u", col("day_date").cast("timestamp").cast("long")) \
                                .stat.approxQuantile("u", [0.8], 0.001)[0]
split_date = pd.to_datetime(split_timestamp, unit='s').strftime('%Y-%m-%d')

print(f"New Split Date: {split_date}")

train_data = df_model_ready.filter(col("day_date") < split_date)
test_data = df_model_ready.filter(col("day_date") >= split_date)

print(f"Fresh Training Samples: {train_data.count():,}")
print(f"Fresh Testing Samples:  {test_data.count():,}")


from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler

def run_ml_pipeline(model_algorithm, model_name, train_df, test_df):
    """
    Creates a pipeline, trains it, and evaluates the results.
    Standardizes the feature assembly process to ensure no data leakage.
    """
    print(f"\nRunning Pipeline for: {model_name}...")
    
    # --- STAGE 1: Feature Assembly ---
    # Define input columns (Lag features + Date info + Location)
    feature_cols = ["price_lag_1", "price_lag_7", "rolling_avg_7", "day_of_week", "month", "lat", "long"]
    
    # Transform individual columns into a single 'features' vector
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    
    # --- STAGE 2: Build Pipeline ---
    # Combine the assembler (preprocessing) and the model (algorithm)
    pipeline = Pipeline(stages=[assembler, model_algorithm])
    
    # --- STAGE 3: Train & Predict ---
    model_fit = pipeline.fit(train_df)
    predictions = model_fit.transform(test_df)
    
    # --- STAGE 4: Evaluate ---
    evaluator_rmse = RegressionEvaluator(labelCol="prix", predictionCol="prediction", metricName="rmse")
    evaluator_mae = RegressionEvaluator(labelCol="prix", predictionCol="prediction", metricName="mae")
    evaluator_r2 = RegressionEvaluator(labelCol="prix", predictionCol="prediction", metricName="r2")
    
    rmse = evaluator_rmse.evaluate(predictions)
    mae = evaluator_mae.evaluate(predictions)
    r2 = evaluator_r2.evaluate(predictions)
    
    print(f"--- {model_name} Results ---")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f} €")
    print(f"R2:   {r2:.4f}")
    
    # Return model and predictions for visualization
    return model_fit, predictions

# ==========================================
# 3. Execution (Run All Models)
# ==========================================

# A. Linear Regression (Baseline)
lr = LinearRegression(labelCol="prix")
lr_pipeline_model, lr_preds = run_ml_pipeline(lr, "Linear Regression", train_data, test_data)

# B. Random Forest (Non-linear)
rf = RandomForestRegressor(labelCol="prix", numTrees=30, maxDepth=10)
rf_pipeline_model, rf_preds = run_ml_pipeline(rf, "Random Forest", train_data, test_data)

# C. Gradient Boosted Tree (Paper Recommendation)
gbt = GBTRegressor(labelCol="prix", maxIter=20)
gbt_pipeline_model, gbt_preds = run_ml_pipeline(gbt, "GBT Regressor", train_data, test_data)