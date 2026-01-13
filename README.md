# French Gas Price Analysis with Apache Spark

This project is aimed at understanding and forecasting fuel price trends across France from 2022 to 2024 using Apache Spark. The **PySpark** ecosystem is utilized to efficiently handle large-scale datasets. The primary objective is to develop an end-to-end data pipeline through which the collection, cleaning, and transformation of raw gas price data is automated, producing actionable insights and predictive models. By leveraging distributed computing, the challenges of processing millions of records related to daily fuel prices, station locations, and services are addressed.


The analysis begins with a robust **Environment Setup** that automatically configures dependencies for both Google Colab and local Linux environments. The **Data Collection** phase involves automated scripts to fetch raw CSV datasets from external repositories. A key strength of this project is its efficient **Data Preparation** utilizing Spark's lazy evaluation and **Parquet** storage formats to optimize performance. The pipeline merges data from multiple years, cleanses inconsistencies, and performs extensive **Feature Engineering**. Critical features created include the "Price Index" (to compare local prices against national averages), continuous "Week Indices" for temporal continuity, and lagged variables (e.g., `price_lag_1`, `rolling_avg_7`) to transform the time-series data into a supervised learning problem.

![Fuel Price Trends](docs/images/docs/images/average_gas_trends.png)



## Visualization and Analysis
To derive qualitative insights, the project employs advanced visualization techniques. **Seaborn** and **Matplotlib** are used to plot temporal price trends, highlighting volatility and seasonal patterns. Furthermore, the project integrates **Folium** to generate interactive geospatial maps. These include heatmaps representing gas price intensity and choropleth maps visualizing average price indices by department, providing a clear spatial distribution of fuel costs across the country.

## Machine Learning and Conclusion
The core modeling phase focuses on forecasting next-day gas prices. The data was split chronologically (80% training, 20% testing) to prevent data leakage—a crucial step for time-series analysis. Three distinct machine learning models were trained and evaluated: **Linear Regression**, **Random Forest Regressor**, and **Gradient-Boosted Trees (GBT)**.

**Conclusion & Results:**
The evaluation produced the following performance metrics on the test set:
* **Linear Regression**: RMSE: ~0.022, $R^2$: 0.94
* **Random Forest**: RMSE: ~0.041, $R^2$: 0.77
* **GBT Regressor**: RMSE: ~0.036, $R^2$: 0.83

Surprisingly, the **Linear Regression** model outperformed the more complex tree-based ensemble methods (Random Forest and GBT) for this specific forecasting task. It achieved the lowest Root Mean Squared Error (RMSE) and the highest $R^2$ score of **94%**. This indicates that the short-term movement of gas prices in this dataset is highly linear and strongly correlated with the immediate past prices (lag features) and weekly trends (rolling averages). While the tree-based models provided respectable results ($R^2$ of 77-83%), the simplicity and interpretability of the linear model proved superior for capturing the immediate day-to-day variance in this specific feature space.

## Links and Resources
* **GitHub Repository**: [france-gas-price-analysis](https://github.com/Dilru1/france-gas-price-analysis)
* **Notebook**: [France Gas Price Notebook](https://colab.research.google.com/github/Dilru1/france-gas-price-analysis/blob/main/france-gas-price.ipynb)


3. 🔗 [Page](https://dilru1.github.io/france-gas-price-analysis/)


