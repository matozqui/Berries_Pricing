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
* [7_Conclusions](#7_Conclusions)

# 1. Introduction
The aim of this project is to predict market prices of strawberry, blueberry and raspberry fruits paid to farmers in the US (mexican imports) and in Spain (internal production).
<br>To anticipate prices in these markets is quite relevant because of various reasons:
-	The global market for fresh berries was valued at a volume of 18 million metric ton in 2018, and it is estimated to register a CAGR of 1.8%.
-	Spain is the EU leading producer in berries and Mexico also leads US berries imports
-	Seasonality of the harvest is very important for the availability of these products during the year so the challenge for many companies is to create/produce differential varieties in precocity and productivity

# 2_Directory_structure
```
├── README.md          <- The top-level README for developers.
├── conf               <- Space for credentials
│
├── data
│   ├── 01_raw              <- Immutable input data
│   ├── 02_intermediate     <- Cleaned version of raw
│   ├── 03_processed        <- Data used to develop models
│   ├── 04_models           <- trained models
│   └── 05_visualizations   <- Reports and input to frontend
│
├── docs               <- Space for documentation
│
├── img                <- Pictures for documentation
│
├── notebooks          <- Jupyter notebooks
│
├── references         <- Data sources, dictionaries, manuals, inspitation material, etc.
│
├── results            <- Final analysis docs.
│
├── requirements.txt   <- The requirements file for reproducing the 
|                         analysis environment.
│
├── .gitignore         <- Avoids uploading data, credentials, 
|                         outputs, system files etc
│
└── src                <- Source code for use in this project.
```

Structure based on [Data Science for Social Good](https://github.com/dssg/hitchhikers-guide/blob/master/sources/curriculum/0_before_you_start/pipelines-and-project-workflow/README.md) with a slightly-modified schema.

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
* **ARIMA**: autoregression model based on the idea that time points of time series data can impact current and future time points.
* **SARIMA**: improved ARIMA with a seasonal component that usually fits well on seasonal data.
* **SARIMAX**: it is a model based on SARIMA and introduces exogenous variables in the model which needs to be fitted in the same training, testing and forecasting date index and shape as the endogenous variables.

Measures used to valuate accuracy of the predicting results are the following:
* **AIC (Akaike Information Criterion)**: a good model is the one that has minimum AIC among all the other models. The AIC can be used to select between the additive and multiplicative Holt-Winters models.
* **BIC (Bayesian Information Criterion)**: another criteria for model selection that measures the trade-off between model fit and complexity of the model. A lower AIC or BIC value indicates a better fit ([Quoc Nam Tran and Hamid Arabnia](https://www.sciencedirect.com/book/9780128025086/emerging-trends-in-computational-biology-bioinformatics-and-systems-biology)).
* **MAE (Mean Absolute Error)**: measures the average of all absolute errors and it is an unambiguous accuracy measure, ideal to gain interpretability in a predictive model.


### 4_5_Visualize

A [Microsoft® SQL Analysis Services](https://docs.microsoft.com/es-es/analysis-services/ssas-overview?view=asallproducts-allversions) model includes all cleaned data from external sources and predicted data resulting from the inference of the models.
<br>A star schema is being used, with four central fact tables (prices, volumes, prices plus volumes and prices predicted) surrounded by dimension tables which are denormalize (country, trade country, regions, time, campaign, product, format):

<p float="centre">
  <img src="/img/SSAS_diagram.JPG" width="600" />
</p>

Then a series of Power BI dashboards connected to this model make possible to look through different type of visualizations that explain prices and volumes in different ways. <br>This visualizations are uploaded in [Power BI website](https://powerbi.microsoft.com/es-es/) and you can access them with the user _biprices@planasa.com_ and password _Planasa2020_. Also this [video](https://youtu.be/ROswL4pIjcY) shows how to access the reports step by step.

# 5_Architecture

![arquitecture](/img/arquitecture.jpg)

Design made with [Cloud Skew app](https://www.cloudskew.com/)

# 6_Summary

### 6_1_Strawberries_Spain

Prices tend to be very high at the beginning of campaigns (end of the year) and rapidly decrease, tending to form a Gamma distribution shape as most prices are concentrated in the low range and a few in a high range. Also there are strong positive correlations in same periods of previous years, a clear seasonality and a downtrend.
<p float="centre">
  <img src="/data/02_intermediate/exloratory_analysis/STRAWBERRIES_ES_ES_std_distribution.png" width="280px" />
  <img src="/data/02_intermediate/exloratory_analysis/STRAWBERRIES_ES_ES_std_acf.png" width="280px" /> 
  <img src="/data/02_intermediate/exloratory_analysis/STRAWBERRIES_ES_ES_std_pacf.png" width="280px" />
</p>
<p float="centre">
  <img src="/data/02_intermediate/exloratory_analysis/STRAWBERRIES_ES_ES_std_boxplot_years.png" width="280px" />
 <img src="/data/02_intermediate/exloratory_analysis/STRAWBERRIES_ES_ES_std_boxplot_weeks.png" width="560px" />
</p>

### 6_2_Strawberries_USA

A bit more noise is being detected compared with Spanish same product. Also prices tend to be very high at the beginning of campaigns and then falls sharply but with a seeming rise before the end of campaigns. Resulting distribution is not very clear across campaigns but is mostly gaussian with a left peak. Not significant correlations appreciated from the ACF and PACF analysis.
<p float="centre">
  <img src="/data/02_intermediate/exloratory_analysis/STRAWBERRIES_US_MX_med_distribution.png" width="280px" />
  <img src="/data/02_intermediate/exloratory_analysis/STRAWBERRIES_US_MX_med_acf.png" width="280px" /> 
  <img src="/data/02_intermediate/exloratory_analysis/STRAWBERRIES_US_MX_med_pacf.png" width="280px" />
</p>
<p float="centre">
  <img src="/data/02_intermediate/exloratory_analysis/STRAWBERRIES_US_MX_med_boxplot_years.png" width="280px" />
 <img src="/data/02_intermediate/exloratory_analysis/STRAWBERRIES_US_MX_med_boxplot_weeks.png" width="560x" />
</p>

### 6_3_Blueberries_Spain

At the beginning of campaign prices tend to be high, falling sharply and increasing at the end of the campaign, forming a bimodal distribution. Also there is a strong negative correlation with previous half year periods lags and positive for same periods of past years. Not a very clear trend identified but there is a clear normalization of prices as boxplot analysis suggests. 
<br>Blueberry harvest in Spain is usually concentrated between march (week 10) and September (week 36), so it seems to be a strong negative correlation between volume production and prices during the periods analyzed.
<p float="centre">
  <img src="/data/02_intermediate/exloratory_analysis/BLUEBERRIES_ES_ES_std_distribution.png" width="280px" />
  <img src="/data/02_intermediate/exloratory_analysis/BLUEBERRIES_ES_ES_std_acf.png" width="280px" /> 
  <img src="/data/02_intermediate/exloratory_analysis/BLUEBERRIES_ES_ES_std_pacf.png" width="280px" />
</p>
<p float="centre">
  <img src="/data/02_intermediate/exloratory_analysis/BLUEBERRIES_ES_ES_std_boxplot_years.png" width="280px" />
 <img src="/data/02_intermediate/exloratory_analysis/BLUEBERRIES_ES_ES_std_boxplot_weeks.png" width="560px" />
</p>

### 6_4_Blueberries_USA

As it happens with same crop in Spanish geography campaigns tend to start in a high price range, then drop and finally increase. This forms a binomial distribution with two clear peaks. What it is remarkable is a clear downtrend of prices from years 2014 to 2020.
<p float="centre">
  <img src="/data/02_intermediate/exloratory_analysis/BLUEBERRIES_US_MX_std_distribution.png" width="280px" />
  <img src="/data/02_intermediate/exloratory_analysis/BLUEBERRIES_US_MX_std_acf.png" width="280px" /> 
  <img src="/data/02_intermediate/exloratory_analysis/BLUEBERRIES_US_MX_std_pacf.png" width="280px" />
</p>
<p float="centre">
  <img src="/data/02_intermediate/exloratory_analysis/BLUEBERRIES_US_MX_std_boxplot_years.png" width="280px" />
 <img src="/data/02_intermediate/exloratory_analysis/BLUEBERRIES_US_MX_std_boxplot_weeks.png" width="560px" />
</p>

### 6_5_Raspberries_Spain

Quite different crop in terms of campaign dates as it starts around week 36, finishing around week 26 of the following year. Campaigns start in a medium-high range of prices, decreasing greatly, then increasing to reach campaign peak prices and finally decreasing again. These two valleys could be partially explained by the impact of day-neutral varieties (two annual harvest).
<p float="centre">
  <img src="/data/02_intermediate/exloratory_analysis/RASPBERRIES_ES_ES_std_distribution.png" width="280px" />
  <img src="/data/02_intermediate/exloratory_analysis/RASPBERRIES_ES_ES_std_acf.png" width="280px" /> 
  <img src="/data/02_intermediate/exloratory_analysis/RASPBERRIES_ES_ES_std_pacf.png" width="280px" />
</p>
<p float="centre">
  <img src="/data/02_intermediate/exloratory_analysis/RASPBERRIES_ES_ES_std_boxplot_years.png" width="280px" />
 <img src="/data/02_intermediate/exloratory_analysis/RASPBERRIES_ES_ES_std_boxplot_weeks.png" width="560px" />
</p>

### 6_6_Raspberries_USA

Very similar behavior as Spanish raspberry in terms of seasonality and price range variations during years. Also during las few years there is a light downtrend.
<p float="centre">
  <img src="/data/02_intermediate/exloratory_analysis/RASPBERRIES_US_MX_std_distribution.png" width="280px" />
  <img src="/data/02_intermediate/exloratory_analysis/RASPBERRIES_US_MX_std_acf.png" width="280px" /> 
  <img src="/data/02_intermediate/exloratory_analysis/RASPBERRIES_US_MX_std_pacf.png" width="280px" />
</p>
<p float="centre">
  <img src="/data/02_intermediate/exloratory_analysis/RASPBERRIES_US_MX_std_boxplot_years.png" width="280px" />
 <img src="/data/02_intermediate/exloratory_analysis/RASPBERRIES_US_MX_std_boxplot_weeks.png" width="560px" />
</p>

# 7_Conclusions


