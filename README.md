# Berries market prices predictor
![berries frame](/docs/berries_frame.jpg)

* [1_Introduction](#1_Introduction)
* [2_Directory_structure](#2_Directory_structure)
* [3_Raw_data](#3_Raw_data)
* [4_Methodology](#4_Methodology)
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

# 5_Architecture

![arquitecture](/docs/arquitecture.jpg)

Design made with [Cloud Skew app](https://www.cloudskew.com/)

# 6_Summary

# 7_Conclusions


