# STRADA — Traffic Risk Assessment & Reliability Framework
Research Project | Machine Learning | Traffic Risk Modeling
Implements the STRADA research framework for accident severity prediction and reliability-aware safety interpretation.

## Overview
**STRADA** (System for Traffic Risk Assessment & Dynamic Analysis) is a machine learning–based research framework for accident severity prediction using large-scale traffic, environmental, and temporal data.

The project focuses on building a reliability-aware prediction pipeline that estimates accident severity and provides interpretable safety insights using contextual features such as weather, time, and regional patterns.

This repository contains the implementation supporting the STRADA research framework and accompanying experiments.

## Key Objectives
* **Predict** accident severity using structured environmental and temporal data.
* **Study** how ML models behave in safety-critical, imbalanced datasets.
* **Provide** interpretable reliability-aware outputs instead of raw predictions.
* **Build** a modular research pipeline for further experimentation.

## Core Features
* **Random Forest–based** severity prediction model.
* **Large-scale data handling** (~7.7M records, sampled training subset).
* **Environmental + temporal** feature engineering.
* **Regional clustering** for spatial risk modeling.
* **Reliability Index** for confidence interpretation.
* **Streamlit interface** for interactive safety analysis.
* **Retrieval-based** safety recommendation module.

## Dataset
This project uses the **US Accidents Dataset**, a large public dataset containing accident records across the United States.

* **Total records:** ~7.7 million
* **Training subset used:** ~500,000 samples (post-pandemic period 2021–2023)
* **Severity scale:**
    1.  Minor
    2.  Moderate
    3.  Severe
    4.  Critical

**Note:** The dataset is not included in this repository due to size. It can be downloaded from Kaggle:
* [US Accidents Dataset (Moosavi et al.)](https://www.kaggle.com/sobhanmoosavi/us-accidents)

### Features Used
* **Weather conditions** (encoded)
* **Temperature**
* **Visibility**
* **Hour of day**
* **Month**
* **Weekday**
* **Regional clusters**

## Tech Stack
* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-learn
* **Database:** PostgreSQL
* **Interface:** Streamlit
* **Integration:** PyTorch (data pipeline integration)

## Project Structure
```text
data/                 # Dataset (excluded from repo)
models/               # Trained model files
src/                  # Training and evaluation scripts
notebooks/            # Exploratory analysis
research_files/       # Generated plots & outputs
app.py                # Streamlit interface