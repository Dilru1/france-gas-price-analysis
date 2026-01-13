#!/usr/bin/env python
# coding: utf-8

# # Spark Mini Project: Analysis of Gas Prices in France
# 

# ## Environment Setup
# To run this pipeline, we need to install the project dependencies listed in `requirements.txt`.
# This ensures that **PySpark**, **Folium**, **Streamlit**, and other necessary libraries are available in your environment.
# 
# *Note: If running on Google Colab, ensure you have cloned the repository or uploaded the `requirements.txt` file first.*

# In[6]:


get_ipython().system('pip install -r ../requirements.txt')


# ## Setup
# Here we import the necessary **PySpark** modules for distributed data processing and machine learning.
# We also configure **Seaborn** and **Matplotlib** for high-quality visualizations.

# In[ ]:


# --- Standard Data & System Libraries ---
import os
import sys
import requests
import json
import pandas as pd
import numpy as np

# --- Visualization Libraries ---
import matplotlib.pyplot as plt
import seaborn as sns
import folium



# --- PySpark Core & SQL ---
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
import pyspark.sql.functions as F
from pyspark.sql.functions import (
    col, lit, when, to_date, to_timestamp, 
    year, month, weekofyear, lag, avg, 
    round, substring, lpad
)

# --- PySpark Machine Learning ---
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor, LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

# Add local 'src' module to path (for accessing your custom scripts)
project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

print("Libraries imported successfully!")


from src.data_loader import download_data_pipeline, load_config
config = load_config()


# ## Initialize Spark Session
# We start the **Spark Session**, which is the entry point for all PySpark functionality. We configure it with **4GB of memory** to handle the large dataset efficiently on a local machine.

# In[8]:


from pyspark.sql import SparkSession

def get_spark_session(app_name="GasPriceProject", memory="4g"):
    """
    Create and return a SparkSession.

    Parameters:
    -----------
    app_name : str, optional
        Name of the Spark application. Default is "GasPriceProject".
    memory : str, optional
        Amount of memory allocated to the Spark driver (e.g., "4g", "8g"). Default is "4g".

    Returns:
    --------
    SparkSession
        Configured SparkSession object ready for use.

    Notes:
    ------
    - Uses all available local cores (`local[*]`).
    - To reduce verbose logging, you can set the log level separately:
      spark.sparkContext.setLogLevel("ERROR")
    """
    spark = SparkSession.builder \
        .master("local[*]") \
        .appName(app_name) \
        .config("spark.driver.memory", memory) \
        .getOrCreate()
    return spark


spark = get_spark_session(memory="8g")
print("Driver memory:", spark.sparkContext.getConf().get("spark.driver.memory"))


# ## Data Collection
# 
# 

# ### Phase 1: Data Ingestion
# We automatically download the required gas price datasets (2022–2024) using our custom `data_loader` module.
# This script reads the source URLs from `config/config.yaml` and handles the file transfers, ensuring we have the latest raw data in the `data/raw/` folder.

# In[9]:


# ==========================================
# Environment Setup (Colab vs Local)
# ==========================================
try:
    from google.colab import drive
    drive.mount("/content/drive") # Mount Google Drive
    project_path = '/content/drive/MyDrive/Colab Notebooks/Spark Project'  #change for desired Project Path

    if not os.path.exists(project_path):
        os.makedirs(project_path)
    get_ipython().run_line_magic('cd', '"{project_path}"')

except ImportError:
    print("Local Environment")

    if os.path.basename(os.getcwd()) == "notebooks":
        os.chdir("..")
        print(f"Moved up to Project Root: {os.getcwd()}")
    else:
        print(f"Current Directory: {os.getcwd()}")

# ==========================================
# Run Data Pipeline
# ==========================================
try:
    from src.data_loader import download_data_pipeline
    download_data_pipeline()

except ImportError as e:
    print(f"Current Path: {os.getcwd()}")
    raise e


# ## Data Preparation – step 1

# ### Read and merge all gas files
# 
# We load all downloaded CSV files (`Prix2022`, `Prix2023`, `Prix2024`) into a single Spark DataFrame. Since the files share the same structure, Spark can merge them automatically using a wildcard path (`Prix*.csv.gz`).
# 
# * **Source:** `data/raw/Prix*.csv.gz`
# * **Format:** CSV (Semi-colon separated)
# * **Schema:** We manually define column names to ensure consistency across files.

# In[10]:


# 1. Define the path to the raw data
# We use a wildcard (*) to read all matching files (2022, 2023, 2024) at once
# Note: We ensure the path is absolute or relative to the notebook root
data_path = os.path.join(project_root, "data", "raw", "Prix*.csv.gz")

# 2. Define the schema (Column Names)
# The raw files do not have headers, so we must provide them
schema_cols = ["id", "cp", "pop", "lat", "long", "date", "type", "name", "prix"]


df_raw = spark.read \
    .format("csv") \
    .option("header", "false") \
    .option("sep", ";") \
    .option("inferSchema", "true") \
    .load(data_path) \
    .toDF(*schema_cols)


df_raw.show(5)
df_raw.printSchema()


# ### New: why this save (because spark is lazy)

# In[11]:


processed_path = os.path.join(project_root, "data", "processed", "gas_data_raw.parquet")

try:
    # Try to load preprocessed data (fast path)
    df_raw = spark.read.parquet(processed_path)
    print("Loaded cached processed data.")

except Exception as e:
    # If it does not exist or is corrupted → recompute

    df_raw.write.mode("overwrite").parquet(processed_path)
    print("Processed data saved to disk.")

# Trigger computation (Spark is lazy)
df_raw.show(5)


# ## Split date in year, month, week of the year
# Split date in year, month, week of the year
# • Prepare latitude & longitude for mapping (divide by the right power of
# 10)

# In[12]:


df_prepared = df_raw \
    .withColumn("date_parsed", to_timestamp(col("date"), "yyyy-MM-dd HH:mm:ss")) \
    .withColumn("day_date", to_date(col("date_parsed"))) \
    .withColumn("year", year(col("date_parsed"))) \
    .withColumn("month", month(col("date_parsed"))) \
    .withColumn("week", weekofyear(col("date_parsed"))) \
    .withColumn("lat", col("lat") / 100000) \
    .withColumn("long", col("long") / 100000) \
    .withColumn("prix", col("prix").cast("double")) \
    .filter(col("date_parsed").isNotNull())  # Remove rows with invalid dates


# In[13]:


processed_path = os.path.join(project_root, "data", "processed", "gas_data_prepared.parquet")

try:
    # Try to load preprocessed data (fast path)
    df_prepared = spark.read.parquet(processed_path)
    print("Loaded cached processed data.")

except Exception as e:
    # If it does not exist or is corrupted → recompute

    df_prepared.write.mode("overwrite").parquet(processed_path)
    print("Processed data saved to disk.")

# Trigger computation (Spark is lazy)
df_prepared.show(5)


# ### Make data available as a table in order to be able to use Spark SQL
# 

# In[14]:


df_prepared.createOrReplaceTempView("gas_prices")

print("Table 'gas_prices' registered successfully.")

# Verify by running a simple SQL query
print("Sample SQL Query Result:")
spark.sql("""
    SELECT year, name, AVG(prix) as avg_price 
    FROM gas_prices 
    GROUP BY year, name 
    ORDER BY year, name
""").show(5)


# ### Basic statistics 
# consider which gas types have some interest for the rest of the project. Two of them are of little interest and can be filtered
# out for the rest of the project.

# In[15]:


# 1. Compute Basic Statistics (Count per Gas Type)
print("Distribution of Records by Gas Type:")
gas_counts = df_prepared.groupBy("name").count().orderBy("count")
gas_counts.show()


# In[16]:


# 2. Identify the 2 Least Frequent Types
# We take the first 2 rows (ascending order means the smallest counts come first)
least_frequent_rows = gas_counts.take(2)
low_interest_types = [row['name'] for row in least_frequent_rows]
low_interest_types


# In[17]:


df_filtered = df_prepared.filter(~col("name").isin(low_interest_types))

gas_counts = df_filtered.groupBy("name").count().orderBy("count", ascending=True)
gas_counts.show()


# ## Data Preparation – step 2

# ### A. Price Index
# We compare each station's price to the **National Daily Average**.
# $$\text{Price Index} = 100 \times \left( \frac{\text{Station Price} - \text{National Avg}}{\text{National Avg}} + 1 \right)$$
# * **Interpretation:**
#     * 100 = Exact average price.
#     * 110 = 10% more expensive than average.
#     * 90 = 10% cheaper than average.
# 
# ### B. Week Index
# We create a continuous time variable (`week_index`) to handle the transition between years (e.g., Week 1 of 2023 should follow Week 52 of 2022).
# * **Formula:** $(\text{Year} - \text{Start Year}) \times 52 + \text{Week Number}$

# In[18]:


from pyspark.sql.functions import col, avg, round

# 1. Calculate Daily National Average (Reference Price)
# We group by 'day_date' and 'name' (gas type) to get the standard price for France that day.
avg_france = df_filtered.groupBy("day_date", "name") \
    .agg(avg("prix").alias("avg_day_price_france"))

# 2. Join the Average back to the main data
# This adds the column 'avg_day_price_france' to every row
df_with_avg = df_filtered.join(avg_france, on=["day_date", "name"], how="left")

# 3. Compute "Price Index"
# Formula: 100 * ((Price - Avg) / Avg + 1)
df_indices = df_with_avg.withColumn("price_index", 
    100 * ((col("prix") - col("avg_day_price_france")) / col("avg_day_price_france") + 1)
)

# 4. Compute "Week Index"
# Formula: (Year - Start_Year) * 52 + Week_Number
# This creates a continuous timeline: Week 1 of 2022 is 1, Week 1 of 2023 is 53.
start_year = config['data_settings']['first_year']  # Should be 2022

df_final = df_indices.withColumn("week_index", 
    (col("year") - start_year) * 52 + col("week")
)

# 5. Cleanup & Verification
# Round metrics for cleaner display
df_final = df_final.withColumn("price_index", round(col("price_index"), 2))

# Show the results as requested
print("Sample of Price Indices:")
df_final.select("day_date", "name", "prix", "avg_day_price_france", "price_index", "week_index").show(5)


# In[19]:


processed_path = os.path.join(project_root, "data", "processed", "gas_data_final.parquet")

try:
    # Try to load preprocessed data (fast path)
    df_prepared = spark.read.parquet(processed_path)
    print("Loaded cached processed data.")

except Exception as e:
    # If it does not exist or is corrupted → recompute

    df_prepared.write.mode("overwrite").parquet(processed_path)
    print("Processed data saved to disk.")

# Trigger computation (Spark is lazy)
df_prepared.show(5)



df_final.printSchema()
# You should see: |-- week_index: integer (nullable = true)


# ## Data Visualization
# 

# ## Using matplotlib/seaborn or plotly

# In[20]:


import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

# 1. Prepare Data for Plotting (Aggregate in Spark -> Convert to Pandas)
# We aggregate by week and gas type to get the mean price
df_viz = df_final.groupBy("week_index", "name") \
    .agg(avg("prix").alias("avg_weekly_price")) \
    .orderBy("week_index")

# Convert to Pandas for Seaborn/Matplotlib
pdf_viz = df_viz.toPandas()

# ==========================================
# Publication-Quality Visualization
# ==========================================

# Set global font scale and style for professional look
sns.set_context("talk")  # larger fonts for readability
sns.set_style("ticks")   # clear ticks on axes

plt.figure(figsize=(14, 7))

# Define a high-contrast palette
palette = sns.color_palette("deep")

# Main Line Plot
ax = sns.lineplot(
    data=pdf_viz,
    x="week_index",
    y="avg_weekly_price",
    hue="name",
    linewidth=3,
    palette=palette,
    alpha=0.9
)

# --- 1. Add Smart Context (Real Events) ---
# Approximate week indices for key French Gas Events (assuming Week 1 = Jan 2022)
events = [
    (10, "Ukraine War Impact"),
    (35, "Gov Rebate (-30c)"),
    (53, "End of Rebate")
]

for week, label in events:
    plt.axvline(x=week, color='#333333', linestyle='--', linewidth=1.5, alpha=0.7)
    plt.text(week + 1, plt.ylim()[1]*0.85, label, rotation=90, 
             fontsize=11, color='#333333', verticalalignment='center')

# --- 2. Highlight Global Peak ---
# Find the absolute max price across all data
max_row = pdf_viz.loc[pdf_viz['avg_weekly_price'].idxmax()]
plt.scatter(max_row['week_index'], max_row['avg_weekly_price'], 
            s=150, color='red', edgecolor='white', zorder=10, label='All-time High')
plt.annotate(
    f"{max_row['avg_weekly_price']:.2f}€",
    xy=(max_row['week_index'], max_row['avg_weekly_price']),
    xytext=(max_row['week_index']-10, max_row['avg_weekly_price']+0.15),
    arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
    fontsize=13, fontweight='bold', color='#B22222'
)

# --- 3. Professional Formatting ---
plt.title("Weekly Evolution of French Gas Prices (2022–2024)", 
          fontsize=20, fontweight='bold', pad=20, loc='left')
plt.xlabel("Week Index (Continuous)", fontsize=14, labelpad=10)
plt.ylabel("Average Price", fontsize=14, labelpad=10)

# Format Y-Axis to Euro currency
ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f€'))

# Despine (Remove top and right borders)
sns.despine(trim=True)

# Grid setup (Horizontal only for cleaner look)
ax.yaxis.grid(True, linestyle=':', alpha=0.6)
ax.xaxis.grid(False)

# Legend adjustment
plt.legend(title="Gas Type", bbox_to_anchor=(1, 1), loc='upper left', frameon=False)

plt.tight_layout()

# Save high-res version
plt.savefig("gas_price_trends.png", dpi=300, bbox_inches='tight')
plt.show()


# ## Data Visualization – Bonus question
# 
# 

# In[21]:


import folium
import requests
import json
from pyspark.sql.functions import col, substring, avg

# ==========================================
# 🗺️ Multi-Layer Map (Fixed)
# ==========================================

gas_types = ["Gazole", "SP95", "SP98", "E10"]

# Load Geometry Once
dept_geo_url = "https://france-geojson.gregoiredavid.fr/repo/departements.geojson"
geo_json_data = requests.get(dept_geo_url).json()

# Create Base Map
m_dept = folium.Map(location=[46.5, 2.3], zoom_start=6, tiles="cartodb positron")

for gas in gas_types:
    print(f"Processing layer: {gas}...")

    # 1. Prepare Data
    pdf_gas = df_final.filter(col("name") == gas) \
        .withColumn("dept", substring(col("cp"), 1, 2)) \
        .groupBy("dept") \
        .agg(avg("price_index").alias("avg_index")) \
        .toPandas()

    # 2. Create FeatureGroup (The Toggle Layer)
    # Only 'Gazole' is checked by default
    fg = folium.FeatureGroup(name=gas, show=(gas == "Gazole"))

    # 3. Create Choropleth (But don't add it yet)
    choropleth = folium.Choropleth(
        geo_data=geo_json_data,
        data=pdf_gas,
        columns=["dept", "avg_index"],
        key_on="feature.properties.code",
        fill_color="YlOrRd" if gas == "Gazole" else "BuPu",
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name=f"{gas} Price Index",
        nan_fill_color="white",
        highlight=True
    )

    # --- THE FIX ---
    # Instead of adding the whole choropleth to 'fg', we split it:

    # A. Add the Polygons (GeoJson) to the FeatureGroup
    # This ensures the map colors toggle on/off with the menu
    choropleth.geojson.add_to(fg)

    # B. Add the Legend (ColorScale) to the Map
    # Note: Folium legends do not toggle automatically. 
    # To avoid 4 overlapping legends, we only show the legend for the default layer (Gazole).
    if gas == "Gazole":
        m_dept.add_child(choropleth.color_scale)

    # Add the FeatureGroup to the map
    fg.add_to(m_dept)

# 4. Add Layer Control
folium.LayerControl(collapsed=False).add_to(m_dept)

# Add Title
title_html = '''
     <h3 align="center" style="font-size:16px"><b>Price Index by Department (Select Layer)</b></h3>
     '''
m_dept.get_root().html.add_child(folium.Element(title_html))

# Display
m_dept


# In[22]:


from folium.plugins import HeatMap
import folium
from pyspark.sql.functions import col, avg

# ==========================================
# 🗺️ Map 2: Interactive HeatMap (Commune Level)
# ==========================================

# 1. Setup Base Map
m_local = folium.Map(location=[46.5, 2.3], zoom_start=6, tiles="cartodb dark_matter")

# 2. Define Gas Types
gas_types = ["Gazole", "SP95", "SP98", "E10"]

# 3. Loop through Gas Types to create layers
for gas in gas_types:
    print(f"Processing HeatMap layer: {gas}...")

    # A. Prepare Data (Spark -> Pandas)
    # We filter specifically for the current gas type in the loop
    df_local_viz = df_final.filter(col("name") == gas) \
        .groupBy("lat", "long") \
        .agg(avg("price_index").alias("avg_index")) \
        .toPandas()

    # B. Prepare Heat Data
    # logic: We want to highlight EXPENSIVE areas.
    # If Index is 100 (average), weight is 10. If 110 (expensive), weight is 20.
    # We filter out data < 90 to keep the map clean (only showing above-average-ish areas)
    heat_data = [
        [row['lat'], row['long'], (row['avg_index'] - 90)] 
        for index, row in df_local_viz.iterrows() 
        if row['avg_index'] > 90  # Changed to > 90 to avoid negative weights
    ]

    # C. Create Feature Group (The Layer)
    # Only show 'Gazole' by default to avoid clutter
    fg = folium.FeatureGroup(name=gas, show=(gas == "Gazole"))

    # D. Add HeatMap to the Feature Group
    HeatMap(
        heat_data,
        radius=15,
        blur=20,
        max_zoom=10,
        gradient={0.4: 'cyan', 0.65: 'lime', 1: 'red'}
    ).add_to(fg)

    # E. Add Feature Group to Map
    fg.add_to(m_local)

# 4. Add Layer Control (The Dropdown)
folium.LayerControl(collapsed=False).add_to(m_local)

# Add Title
title_html = '''
     <h3 align="center" style="color: white; font-size:16px"><b>Gas Price Intensity (Select Layer)</b></h3>
     '''
m_local.get_root().html.add_child(folium.Element(title_html))

# Display
m_local


# In[ ]:





# In[ ]:





# ## Modeling – Forecast next day price

# In[23]:


from pyspark.sql.window import Window
from pyspark.sql.functions import col, lag, avg, dayofweek, month
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor, LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import GBTRegressor

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# In[ ]:


target_gas = "Gazole" 
df_ml = df_final.filter(col("name") == target_gas)
df_ml.select("id", "name", "day_date").show(5)

#df_ml = df_final.filter(col("name") == "Gazole") \
#                .filter(col("prix") < 3.0)  


# In[25]:


window_spec = Window.partitionBy("id", "name").orderBy("day_date")

# Create Features:
# - price_lag_1: Price 1 day ago (Autoregression)
# - price_lag_7: Price 1 week ago (Seasonality)
# - rolling_avg_7: Trend over the last week
df_features = df_ml \
    .withColumn("price_lag_1", lag("prix", 1).over(window_spec)) \
    .withColumn("price_lag_7", lag("prix", 7).over(window_spec)) \
    .withColumn("rolling_avg_7", avg("prix").over(window_spec.rowsBetween(-7, -1))) \
    .withColumn("day_of_week", dayofweek("day_date")) \
    .withColumn("month", month("day_date")) \
    .dropna()  # Remove first few days that have no history

# Prepare Vector for Spark ML
feature_cols = ["price_lag_1", "price_lag_7", "rolling_avg_7", "day_of_week", "month", "lat", "long"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df_model_ready = assembler.transform(df_features)


# In[26]:


df_model_ready.select("id", "prix", "price_lag_1", "price_lag_7", "rolling_avg_7", "day_of_week", "month", "lat", "long").show(5)


# In[27]:


split_date = "2023-10-01"

train_data = df_model_ready.filter(col("day_date") < split_date)
test_data = df_model_ready.filter(col("day_date") >= split_date)

print(f"Training Samples: {train_data.count():,}")
print(f"Testing Samples:  {test_data.count():,}")


# In[28]:


# ==========================================
# 3. Model Training (Linear Regression & Random Forest)
# ==========================================

# --- Model A: Linear Regression ---
print("\nTraining Linear Regression...")
lr = LinearRegression(featuresCol="features", labelCol="prix")
lr_model = lr.fit(train_data)
pred_lr = lr_model.transform(test_data)

# --- Model B: Random Forest (Recommended) ---
print("Training Random Forest...")
rf = RandomForestRegressor(featuresCol="features", labelCol="prix", numTrees=30, maxDepth=10)
rf_model = rf.fit(train_data)
pred_rf = rf_model.transform(test_data)

# --- Model C: Random Forest (Recommended) ---

gbt = GBTRegressor(featuresCol="features", labelCol="prix", maxIter=20)
gbt_model = rf.fit(train_data)
pred_gbt = rf_model.transform(test_data)


# In[29]:


# ==========================================
# 4. Evaluation (Accuracy Measures)
# ==========================================
evaluator_rmse = RegressionEvaluator(labelCol="prix", predictionCol="prediction", metricName="rmse")
evaluator_mae = RegressionEvaluator(labelCol="prix", predictionCol="prediction", metricName="mae")
evaluator_r2 = RegressionEvaluator(labelCol="prix", predictionCol="prediction", metricName="r2")


# In[30]:


def print_metrics(name, predictions):
    rmse = evaluator_rmse.evaluate(predictions)
    mae = evaluator_mae.evaluate(predictions)
    r2 = evaluator_r2.evaluate(predictions)
    print(f"--- {name} Results ---")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f} € (Average Error)")
    print(f"R2:   {r2:.4f}")

print_metrics("Linear Regression", pred_lr)
print_metrics("Random Forest", pred_rf)
print_metrics("GBT", pred_gbt)


# In[31]:


import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 5. Comparative Dispersion Plot (Side-by-Side)
# ==========================================
print("Generating Comparative Dispersion Plots...")

# 1. Sample Data from BOTH predictions
# We use the same seed to ensure we are looking at roughly the same subset of days
sample_lr = pred_lr.select("prix", "prediction").sample(fraction=0.05, seed=42).toPandas()
sample_rf = pred_rf.select("prix", "prediction").sample(fraction=0.05, seed=42).toPandas()

# 2. Determine Global Min/Max for Axis Scaling
# (Crucial: Both plots must use the exact same X and Y limits to be comparable)
global_min = min(sample_lr["prix"].min(), sample_lr["prediction"].min())
global_max = max(sample_lr["prix"].max(), sample_lr["prediction"].max())

# 3. Setup the Figure (1 Row, 2 Columns)
fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharey=True)

# --- Plot 1: Linear Regression ---
sns.scatterplot(
    x=sample_lr["prix"], 
    y=sample_lr["prediction"], 
    alpha=1, 
    color='tab:blue', 
    ax=axes[0]
)
axes[0].plot([global_min, global_max], [global_min, global_max], 'r--', lw=2, label="Perfect Prediction")
axes[0].set_title("Linear Regression", fontsize=16, fontweight='bold')
axes[0].set_xlabel("Actual Price (€)", fontsize=14)
axes[0].set_ylabel("Predicted Price (€)", fontsize=14)
axes[0].grid(True, linestyle='--', alpha=0.6)
axes[0].legend()

# --- Plot 2: Random Forest ---
sns.scatterplot(
    x=sample_rf["prix"], 
    y=sample_rf["prediction"], 
    alpha=1, 
    color='tab:green', 
    ax=axes[1]
)
axes[1].plot([global_min, global_max], [global_min, global_max], 'r--', lw=2, label="Perfect Prediction")
axes[1].set_title("Random Forest", fontsize=16, fontweight='bold')
axes[1].set_xlabel("Actual Price (€)", fontsize=14)
# Y-label is hidden on the second plot since they share the axis
axes[1].grid(True, linestyle='--', alpha=0.6)
axes[1].legend()

# Final Layout Adjustments
plt.suptitle(f"Model Comparison for {target_gas}: Which predicts better?", fontsize=20, y=1.02)
plt.tight_layout()
plt.show()


# In[34]:


from pyspark.sql.functions import col, lit

# 1. List of Suspicious Station IDs (from your output)
suspicious_ids = [17350003, 42680001, 83490003, 12230004, 86000021]

print("🕵️‍♂️ Inspecting Context for Suspicious Stations...")

# 2. Loop through each suspicious station to see the "Before" and "After"
for station_id in suspicious_ids:
    print(f"\n--- Station ID: {station_id} ---")

    # Get data for this station, sorted by date
    # We filter for the specific timeframe (Late 2023) to see the context
    context_df = df_final.filter(col("id") == station_id) \
                         .filter(col("day_date").between("2023-10-01", "2024-01-15")) \
                         .select("id", "day_date", "prix") \
                         .orderBy("day_date")

    # Show the rows around the outlier
    # (Spark .show() might cut it off, so we convert to pandas for a clean table)
    print(context_df.toPandas().to_string(index=False))


# In[32]:


# ==========================================
# 6. Time Series Inspection (Actual vs Forecast)
# ==========================================
# Pick a specific station to visualize (e.g., the one with the most data)
# OR just pick a random ID from the test set
station_id = test_data.select("id").first()[0] 

print(f"Visualizing Forecast for Station ID: {station_id}")

# 1. Filter data for this station & sort by time
# We convert to Pandas because it's small (only ~300 rows for one station)
pdf_station = pred_rf.filter(col("id") == station_id) \
                     .select("day_date", "name", "prix", "prediction") \
                     .orderBy("day_date") \
                     .toPandas()

# 2. Plot
plt.figure(figsize=(14, 6))

# Plot Actual Price
sns.lineplot(data=pdf_station, x="day_date", y="prix", label="Actual Price", color="black", linewidth=2)

# Plot Predicted Price (Dashed)
sns.lineplot(data=pdf_station, x="day_date", y="prediction", label="Model Forecast", color="red", linestyle="--", linewidth=2)

plt.title(f"Forecast Validation for Station {station_id}", fontsize=16)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Price (€)", fontsize=12)
plt.legend()
plt.grid(True, alpha=0.5)

# Format Date axis nicely
import matplotlib.dates as mdates
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())

plt.show()


# In[ ]:




