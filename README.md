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

During the project different technologies, techniques and analytical methods have been used trying to fit-to-purpose of each step.

### 4_1_Ingestion

Data is collected using web scrapping techniques (US data) and pandas excel import feature (Spanish data). Then is stored in an SQL Server database which consolidates all data in one repository.

### 4_2_Cleaning

Data cleaning and transformation includes descriptive tags homogenizing, format conversions to kilogrames, geolocation, measures grouping, resampling, merging, interpolating, type transformations.
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
* **AIC (Akaike Information Criterion)**: a good model is the one that has minimum AIC among all the other models. The AIC can be used to select between the additive and multiplicative Holt-Winters models ([Quoc Nam Tran and Hamid Arabnia](https://www.sciencedirect.com/book/9780128025086/emerging-trends-in-computational-biology-bioinformatics-and-systems-biology)).
* **BIC (Bayesian Information Criterion)**: another criteria for model selection that measures the trade-off between model fit and complexity of the model. A lower AIC or BIC value indicates a better fit [Quoc Nam Tran and Hamid Arabnia](https://www.sciencedirect.com/book/9780128025086/emerging-trends-in-computational-biology-bioinformatics-and-systems-biology)).
* **MAE (Mean Absolute Error)**: measures the average of all absolute errors and it is an unambiguous accuracy measure, ideal to gain interpretability in a predictive model.


### 4_5_Visualize

A [Microsoft® SQL Analysis Services](https://docs.microsoft.com/es-es/analysis-services/ssas-overview?view=asallproducts-allversions) model includes all cleaned data from external sources and predicted data resulting from the inference of the models.
A star schema is being used, with four central fact tables (prices, volumes, prices plus volumes and prices predicted) surrounded by dimension tables which are denormalize (country, trade country, regions, time, campaign, product, format):

![SSAS_diagram](/img/SSAS_diagram.JPG)

Then a series of [Power BI]([https://powerbi.microsoft.com/es-es/]) dashboards connected to this model make possible to look through different type of visualizations that explain prices and volumes in different ways.

# 5_Architecture

![arquitecture](/img/arquitecture.jpg)

Design made with [Cloud Skew app](https://www.cloudskew.com/)

# 6_Summary

![Exploratory results](https://github.com/matozqui/Berries_Pricing/tree/master/data/02_intermediate/exloratory_analysis)

<object data="https://github.com/matozqui/Berries_Pricing/blob/master/data/02_intermediate/exloratory_analysis/BLUEBERRIES_ES_ES_std.pdf" type="application/pdf" width="700px" height="700px">
    <embed src="https://github.com/matozqui/Berries_Pricing/blob/master/data/02_intermediate/exloratory_analysis/BLUEBERRIES_ES_ES_std.pdf">
        <p>This browser does not support PDFs. Please download the PDF to view it: <a href="https://github.com/matozqui/Berries_Pricing/blob/master/data/02_intermediate/exloratory_analysis/BLUEBERRIES_ES_ES_std.pdf">Download PDF</a>.</p>
    </embed>
</object>

<embed src="https://github.com/matozqui/Berries_Pricing/blob/master/data/02_intermediate/exloratory_analysis/BLUEBERRIES_ES_ES_std.pdf" width="500" height="375" 
 type="application/pdf">
 
 <embed src="https://drive.google.com/viewerng/
viewer?embedded=true&url=https://github.com/matozqui/Berries_Pricing/blob/master/data/02_intermediate/exloratory_analysis/BLUEBERRIES_ES_ES_std.pdf" width="500" height="375">

<iframe src="http://docs.google.com/gview?url=https://github.com/matozqui/Berries_Pricing/blob/master/data/02_intermediate/exloratory_analysis/BLUEBERRIES_ES_ES_std.pdf&embedded=true" style="width:718px; height:700px;" frameborder="0"></iframe>

# 7_Conclusions


