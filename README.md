# Marginal CPC Analysis

This repository contains tools to analyze and forecast Cost Per Click (CPC) metrics for digital advertising campaigns. It includes two main components: a robust data processing and modeling notebook and an interactive Streamlit app for real-time CPC predictions based on future spend.

---

## Overview

### 1. [CPC Modeling Notebook](https://github.com/jra333/marginal_dimreturns_cpc/tree/main/marginal_cpc_notebooks)

- **Data Extraction & Cleaning**
  - Connects to Snowflake to retrieve raw cost, clicks, impressions, and search impression share data.
  - Cleans and preprocesses the data (e.g., date conversion, type corrections, sorting, and re-indexing).
  
- **Feature Engineering & Aggregation**
  - Computes additional variables like CPC and Eligible Impressions.
  - Applies resampling (daily) and aggregation.
  - Creates lag features and rolling averages for deeper time-series insights.
  
- **Exploratory Analysis**
  - Splits the dataset by year (2022, 2023, 2024) and provides descriptive statistics.
  - Exports the cleaned dataset for further analysis.

- **Modeling**
  - Leverages the cleaned and engineered dataset to train regression-based models aimed at forecasting CPC.
  - Performs model fitting, validation, and selection using various techniques (e.g., Linear Regression, Ridge, Lasso, etc.).
  - Provides diagnostic metrics (e.g., R², MSE, MAE) to assess model performance.
  - Saves the best performing model (with scalers and feature names) that is later used within the interactive app for real-time predictions.


### 2. [Interactive CPC Prediction App](https://github.com/jra333/marginal_dimreturns_cpc/tree/main/marginal_cpc_app_testing)

- **Model Loading & Prediction**
  - Loads a pre-trained model along with relevant scalers and feature names from a pickle file.
  - Provides a prediction function that uses scaled inputs to forecast the CPC.
  
- **Data Upload & Model Update**
  - Users can upload a CSV file of historical data that is parsed and cleaned inside the app.
  - If new data is present, lag features are generated and the model is updated (re-trained) accordingly.
  
- **User Interface**
  - Offers sliders for entering daily spend values for a week.
  - Computes and displays:
    - Predicted CPC per day.
    - Estimated number of clicks (Spend/CPC).
    - Estimated impressions (using user-defined CTR).
  - Features interactive visualizations (via Plotly) showing spend versus predicted CPC along with a regression trendline.

