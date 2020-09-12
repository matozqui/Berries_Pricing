# Berries market prices predictor
![berries frame](/img/berries_frame.jpg)

* [1_Introduction](#1_Introduction)
* [2_Directory_structure](#2_Directory_structure)
* [3_Raw_data](#3_Raw_data)
* [4_Methodology](#4_Methodology)
  * [4_1_Ingestion](#4_1_Ingestion)
  * [4_2_Cleaning](#4_2_Cleaning)
  * [4_3_Exploration](#4_3_Exploration)
  * [4_4_Modelling](#4_4_Modelling)
  * [4_5_Visualize](#4_5_Visualize)  
* [5_Architecture](#5_Architecture)
* [6_Summary](#6_Summary)
  * [6_1_Strawberries_Spain](#6_1_Strawberries_Spain)
  * [6_2_Strawberries_USA](#6_2_Strawberries_USA)
  * [6_3_Blueberries_Spain](#6_3_Blueberries_Spain)
  * [6_4_Blueberries_USA](#6_4_Blueberries_USA)
  * [6_5_Raspberries_Spain](#6_5_Raspberries_Spain)
  * [6_6_Raspberries_USA](#6_6_Raspberries_USA)
  * [6_7_Model_Results](#6_7_Model_Results)      
* [7_Conclusions](#7_Conclusions)
* [8_Next_steps](#8_Next_steps)

# 1. Introduction
The aim of this project is to predict market prices of strawberry, blueberry and raspberry fruits paid to farmers in the US (mexican imports) and in Spain (internal production).
<br>To anticipate prices in these markets is quite relevant because of various reasons:

-	The global market for fresh berries was valued at a volume of 18 million metric ton in 2018, and it is estimated to register a CAGR of 1.8%
-	Spain is the EU leading producer in berries and Mexico also leads US berries imports
-	Seasonality of the harvest is very important for the availability of these products during the year so the challenge for many companies is to create/produce differential varieties in precocity and productivity

<br>Also I work for a global operator in the agri-food sector, specialized in the berries nursery business and I have had the chance to grasp some knowledge about this market.

<br>You can check the prediction results [here](https://app.powerbi.com/view?r=eyJrIjoiODM1YTdhYmUtYWFkNi00YmZkLTllNmMtNzIwN2ViYzg4Y2M4IiwidCI6IjFkOGU3NzE5LWI2ZjctNGI3ZS1hN2IxLTliOTk3NTI5NTEyMiIsImMiOjh9)

# 2_Directory_structure
```
├── README.md          <- The top-level README for developers.
├── conf               <- Space for credentials
│
├── data
│   ├── 01_raw              <- Immutable input data
│   ├── 02_intermediate     <- Cleaned version of raw
│   ├── 03_models           <- trained models
│   └── 04_visualizations   <- Reports and input to frontend
│
├── docs               <- Space for documentation
│
├── img                <- Pictures for documentation
│
├── notebooks          <- Jupyter notebooks
│   ├── exploration    <- Exploration of data (mostly table dataframes and plots)
│   ├── processing     <- Calls to processing functions (import data, train and inference models)  
│   └── scripts        <- Jupyter notebooks saved in src directory as python scripts 
│
├── references         <- Data sources, dictionaries, manuals, inspitation material, etc.
│
├── requirements.txt   <- The requirements file for reproducing the 
|                         analysis environment.
│
├── .gitignore         <- Avoids uploading data, credentials, 
|                         outputs, system files etc
│
└── src                <- Source code for use in this project.
    ├── d00_utils           <- Scripts for general use
    ├── d01_data            <- Scripts for importing and extracting data
    ├── d02_processing      <- Scripts for transformations and data preparation
    ├── d03_modelling       <- Scripts for training and inferencing models
    └── d04_visualisation   <- Visualization .pbix files
```
Structure based on [Data Science for Social Good](https://github.com/dssg/hitchhikers-guide/blob/master/sources/curriculum/0_before_you_start/pipelines-and-project-workflow/README.md) with a slightly-modified schema.

Models' pickle files are available in [this public Google Drive directory](https://drive.google.com/drive/folders/1uSP2m3BMx9ofepc-X59iSIAm0VuQxhlj?usp=sharing)

# 3_Raw_data

| Data Source | Data Kind | Geography | Format | Periodicity | History | Rows |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| USDA | Market prices | US (84 regions) | URL | Daily | 10 years | 40000 |
| USDA | Market volumes | US (24 regions) | URL | Daily | 20 years | 150000 |
| Junta de Andalucia | Market prices | Spain | .xls | Weekly | 4 years | 500 |
| Int. Blueberry Org. | Market volumes | Spain | .xls | Weekly | 5 years | 3000 |
| Ministerio de Agricultura | Labour Cost | Spain | .xls | Yearly | 35 years | 35 |

# 4_Methodology

During the different steps of the project different technologies, techniques and analytical methods have been applied:

### 4_1_Ingestion

Data is collected using web scrapping techniques (US data) and pandas excel import feature (Spanish data). Then is stored in an SQL Server database which consolidates all data in one repository.

### 4_2_Cleaning

Data cleaning and transformation includes descriptive tags homogenizing, format conversions to kilogrames, geolocation, measures grouping, resampling, merging, interpolating, type transformations, normalization and denormalization of data to fit models.
One of the main issues is handling non-campaign periods in which there are not prices (target variable to predict.) To handle it data is interpolated to train models but omitted when validating and predicting.

### 4_3_Exploration

To better understand the nature of data a series of analysis are carried out:
* Series evolution line charts grouped by year to understand trend and seasonality
* Yearly-wise and weekly-wise boxplots to understand trend and seasonality respectively
* Autocorrelation function (ACF) to find correlations of present time points with lags
* Partially autocorrelation function (PACF) to find correlation of the residuals

### 4_4_Modelling

Models used in the project are the following:
* **ARIMA**: autoregression model based on the idea that time points of time series data can impact current and future time points
* **SARIMA**: improved ARIMA with a seasonal component that usually fits well on seasonal data
* **SARIMAX**: it is a model based on SARIMA and introduces exogenous variables in the model which needs to be fitted in the same training, testing and forecasting date index and shape as the endogenous variables

Measures used to valuate accuracy of the predicting results are the following:
* **AIC (Akaike Information Criterion)**: a good model is the one that has minimum AIC among all the other models. The AIC can be used to select between the additive and multiplicative Holt-Winters models
* **BIC (Bayesian Information Criterion)**: another criteria for model selection that measures the trade-off between model fit and complexity of the model. A lower AIC or BIC value indicates a better fit ([Quoc Nam Tran and Hamid Arabnia](https://www.sciencedirect.com/book/9780128025086/emerging-trends-in-computational-biology-bioinformatics-and-systems-biology))
* **MAE (Mean Absolute Error)**: measures the average of all absolute errors and it is an unambiguous accuracy measure, ideal to gain interpretability in a predictive model
* **MAPE (Mean Absolute Percentage Error)**: is the absolute error normalized over the data, which allows the error to be compared across data with different scales
* **MSE (Mean Square Error)**: shows the average over the test sample of the absolute differences between prediction and actual observation
* **RMSE (Root Mean Square Error)**: quadratic scoring rule that also measures the average magnitude of the error, giving a relatively high weight to large errors


### 4_5_Visualize

A [Microsoft® SQL Analysis Services](https://docs.microsoft.com/es-es/analysis-services/ssas-overview?view=asallproducts-allversions) model includes all cleaned data from external sources and predicted data resulting from the inference of the models.
<br>A star schema is being used, with four central fact tables (prices, volumes, prices plus volumes and prices predicted) surrounded by dimension tables which are denormalize (country, trade country, regions, time, campaign, product, format):

<p align="center">
  <img src="/img/SSAS_diagram.JPG" width="600" />
</p>

Then a series of Power BI dashboards connected to this model make possible to look through different type of visualizations that explain prices and volumes in different ways. <br>This visualizations are uploaded in [Power BI website](https://powerbi.microsoft.com/es-es/) and you can access them with the user _biprices@planasa.com_ and password _Planasa2020_. Also this [video](https://youtu.be/ROswL4pIjcY) shows how to access the reports step by step.
<br>The difference between these and the rest of visualizations included in this readme is the connection. While this one is an online connection to the SSAS model, making possible to query more than 200K rows in a matter of tenth of seconds, the public ones are made with local connections which means there are showing static data. That is why online connections dashboards shows more detailed price and volume analysis.   

# 5_Architecture

![arquitecture](/img/arquitecture.jpg)

Design made with [Cloud Skew app](https://www.cloudskew.com/)

The Jupyter Notebook [step_by_step](notebooks/processing/step_by_step.ipynb) include the whole process from ingesting data to predict. Also this repository includes the [SQL Server instance](data/01_raw/SQL_DB_Prices_CREATES.sql) with all tables structure and the [SSAS model](data/04_visualization/SSAS_Prices_CREATE.xmla). 

# 6_Summary

### 6_1_Strawberries_Spain

Prices tend to be very high at the beginning of campaigns (end of the year) and rapidly decrease, tending to form a Gamma distribution shape as most prices are concentrated in the low range and a few in a high range. Also there are strong positive correlations in same periods of previous years, a clear seasonality and a downtrend.
<p float="centre">
  <img src="/data/02_intermediate/exloratory_analysis/STRAWBERRIES_ES_ES_std_distribution.png" width=30% />
  <img src="/data/02_intermediate/exloratory_analysis/STRAWBERRIES_ES_ES_std_acf.png" width=30% /> 
  <img src="/data/02_intermediate/exloratory_analysis/STRAWBERRIES_ES_ES_std_pacf.png" width=30% />
</p>
<p float="centre">
  <img src="/data/02_intermediate/exloratory_analysis/STRAWBERRIES_ES_ES_std_boxplot_years.png" width=30% />
 <img src="/data/02_intermediate/exloratory_analysis/STRAWBERRIES_ES_ES_std_boxplot_weeks.png" width=60% />
</p>

### 6_2_Strawberries_USA

A bit more noise is being detected compared with Spanish same product. Also prices tend to be very high at the beginning of campaigns and then falls sharply but with a seeming rise before the end of campaigns. Resulting distribution is not very clear across campaigns but is mostly gaussian with a left peak. Not significant correlations appreciated from the ACF and PACF analysis.
<p float="centre">
  <img src="/data/02_intermediate/exloratory_analysis/STRAWBERRIES_US_MX_med_distribution.png" width=30% />
  <img src="/data/02_intermediate/exloratory_analysis/STRAWBERRIES_US_MX_med_acf.png" width=30% /> 
  <img src="/data/02_intermediate/exloratory_analysis/STRAWBERRIES_US_MX_med_pacf.png" width=30% />
</p>
<p float="centre">
  <img src="/data/02_intermediate/exloratory_analysis/STRAWBERRIES_US_MX_med_boxplot_years.png" width=30% />
 <img src="/data/02_intermediate/exloratory_analysis/STRAWBERRIES_US_MX_med_boxplot_weeks.png" width=60% />
</p>

### 6_3_Blueberries_Spain

At the beginning of campaign prices tend to be high, falling sharply and increasing at the end of the campaign, forming a bimodal distribution. Also there is a strong negative correlation with previous half year periods lags and positive for same periods of past years. Not a very clear trend identified but there is a clear normalization of prices as boxplot analysis suggests. 
<br>Blueberry harvest in Spain is usually concentrated between march (week 10) and September (week 36), so it seems to be a strong negative correlation between volume production and prices during the periods analyzed.
<p float="centre">
  <img src="/data/02_intermediate/exloratory_analysis/BLUEBERRIES_ES_ES_std_distribution.png" width=30% />
  <img src="/data/02_intermediate/exloratory_analysis/BLUEBERRIES_ES_ES_std_acf.png" width=30% /> 
  <img src="/data/02_intermediate/exloratory_analysis/BLUEBERRIES_ES_ES_std_pacf.png" width=30% />
</p>
<p float="centre">
  <img src="/data/02_intermediate/exloratory_analysis/BLUEBERRIES_ES_ES_std_boxplot_years.png" width=30% />
 <img src="/data/02_intermediate/exloratory_analysis/BLUEBERRIES_ES_ES_std_boxplot_weeks.png" width=60% />
</p>

### 6_4_Blueberries_USA

As it happens with same crop in Spanish geography campaigns tend to start in a high price range, then drop and finally increase. This forms a binomial distribution with two clear peaks. What it is remarkable is a clear downtrend of prices from years 2014 to 2020.
<p float="centre">
  <img src="/data/02_intermediate/exloratory_analysis/BLUEBERRIES_US_MX_std_distribution.png" width=30% />
  <img src="/data/02_intermediate/exloratory_analysis/BLUEBERRIES_US_MX_std_acf.png" width=30% /> 
  <img src="/data/02_intermediate/exloratory_analysis/BLUEBERRIES_US_MX_std_pacf.png" width=30% />
</p>
<p float="centre">
  <img src="/data/02_intermediate/exloratory_analysis/BLUEBERRIES_US_MX_std_boxplot_years.png" width=30% />
 <img src="/data/02_intermediate/exloratory_analysis/BLUEBERRIES_US_MX_std_boxplot_weeks.png" width=60% />
</p>

### 6_5_Raspberries_Spain

Quite different crop in terms of campaign dates as it starts around week 36, finishing around week 26 of the following year. Campaigns start in a medium-high range of prices, decreasing greatly, then increasing to reach campaign peak prices and finally decreasing again. These two valleys could be partially explained by the impact of day-neutral varieties (two annual harvest).
<p float="centre">
  <img src="/data/02_intermediate/exloratory_analysis/RASPBERRIES_ES_ES_std_distribution.png" width=30% />
  <img src="/data/02_intermediate/exloratory_analysis/RASPBERRIES_ES_ES_std_acf.png" width=30% /> 
  <img src="/data/02_intermediate/exloratory_analysis/RASPBERRIES_ES_ES_std_pacf.png" width=30% />
</p>
<p float="centre">
  <img src="/data/02_intermediate/exloratory_analysis/RASPBERRIES_ES_ES_std_boxplot_years.png" width=30% />
 <img src="/data/02_intermediate/exloratory_analysis/RASPBERRIES_ES_ES_std_boxplot_weeks.png" width=60% />
</p>

### 6_6_Raspberries_USA

Very similar behavior as Spanish raspberry in terms of seasonality and price range variations during years. Also during las few years there is a light downtrend.
<p float="centre">
  <img src="/data/02_intermediate/exloratory_analysis/RASPBERRIES_US_MX_std_distribution.png" width=30% />
  <img src="/data/02_intermediate/exloratory_analysis/RASPBERRIES_US_MX_std_acf.png" width=30% /> 
  <img src="/data/02_intermediate/exloratory_analysis/RASPBERRIES_US_MX_std_pacf.png" width=30% />
</p>
<p float="centre">
  <img src="/data/02_intermediate/exloratory_analysis/RASPBERRIES_US_MX_std_boxplot_years.png" width=30% />
 <img src="/data/02_intermediate/exloratory_analysis/RASPBERRIES_US_MX_std_boxplot_weeks.png" width=60% />
</p>

### 6_7_Model_Results

After doing the exploration a series of models have been generated, following a common criteria:
-	Data period ranges are weekly
-	Weeks with no target value (no campaign) have been interpolated for fitting the model and removed for measuring and predicting
-	Training data include all prices up to the end of the last year (currently all prices up to 31st December 2019)
-	Test data include all prices available related to the current year
-	The inference of predictions are generated for the current and next year
-	MAE is the measure selected to pick the best ARIMA order combination (p, d, q)
-	A seasonal order of (1, 1, 52) for SARIMA models is applied as a clear annual seasonality was identified during exploration phase

SARIMAX models have been generated for the three products in the US region including volumes imported from Mexico as exogenous variables. Volume data have been summarized weekly, fitted in an extra ARIMA model and predicted the convenient number of periods (current and next year) to finally fit the SARIMAX price model.

<br>The benchmark around the models applied can be checked in a radar diagram [here](https://app.powerbi.com/view?r=eyJrIjoiNDBmMjRlZTEtYzgzZi00MWNiLWI1YjQtMGNmNzU1Yjc5MGRjIiwidCI6IjFkOGU3NzE5LWI2ZjctNGI3ZS1hN2IxLTliOTk3NTI5NTEyMiIsImMiOjh9&pageName=ReportSection), but in summary these are the main results with the best models highlighted:
<p align="center">
<img width="30%" src="/img/Model_results.JPG">
</p>
Generally speaking more complex models have obtained the best predicting results (SARIMA in Spain and SARIMAX in the USA). The exceptions are Spanish raspberries and US strawberries, which casually showed unclear seasonality and trend during the exploration phase.<br>In terms of deviation models-to-beat work reasonably well with average absolute deviations between 13% and 30%.

# 7_Conclusions

-	Great model complexity does not guarantee better results. For example the best model for US strawberries prices prediction was ARIMA with a MAE of 0.53 VS 0.99 of SARIMAX.
-	It takes a great amount of time to get cleaned data. In this project specially for homogenizing different sales formats, geographies and units of measure. Anyway is worth it as it allow to create great visualizations to understand reality and there are better chances to get a good prediction model.
-	Structure and order is important for coding. It allows to scale problems faster and reduce programming errors.
-	To succeed in a Data Science project a constant self-improvement mentality is required. There is never an ending and models could always be improved, but it is also important to make quick wins and congratulate oneself to keep pushing.

# 8_Next_steps

-	Find other exogenous variables which potentially improve model predictions (weather conditions, auxiliary material prices, consumer demand evolution)
-	Include internal prices of my company in the Database to compare internal prices with market prices
-	Test new models such as Prophet and random forests
