## 🌍 Synthetix – Forest & Flood Intelligence

**An Integrated Climate Intelligence Platform for Forest Monitoring, Flood Prediction, and District-Level Risk Assessment.**

## 📌 Project Overview

Synthetix is a climate intelligence web platform that combines machine learning, forest segmentation, flood prediction, and geospatial data visualization.

The platform integrates two previously trained machine learning models to analyze forest cover and predict flood probabilities. It brings these results together to provide district-level environmental risk insights through an interactive web application.

The project was developed as part of the **Resilience Nexus – Three-Round Challenge**.

## 🎯 Project Objectives

- Predict potential flood events using machine learning.
- Monitor forest cover through satellite imagery analysis.
- Classify forest and non-forest regions.
- Analyze environmental conditions at district level.
- Combine flood exposure and forest deficit into a risk score.
- Present environmental information through an interactive dashboard.

## 🧠 Machine Learning Models

### Round 1 – Flood Prediction

**Model:** XGBoost Classifier

The flood prediction component uses a trained machine learning model to estimate flood probabilities from time-series and environmental data.

Features include:
- Flood probability prediction.
- District-level flood analysis.
- Station-level monitoring.
- Date-based filtering.
- Flood prediction exports.
- Model performance reporting.

### Round 2 – Forest Segmentation

**Model:** XGBoost Classifier

The forest intelligence component performs pixel-wise classification of satellite imagery into forest and non-forest classes.

Features include:
- Forest and non-forest classification.
- Segmentation mask generation.
- Forest coverage estimation.
- NDVI visualization.
- Probability mapping.
- NoData pixel preservation.
- District-level forest statistics.

## 🖥️ Round 3 – Web Application

The third round integrates both machine learning models into a web-based climate intelligence platform.

### Main Features

- Interactive forest monitoring.
- Flood prediction dashboard.
- District-level risk ranking.
- Forest coverage statistics.
- Flood probability visualization.
- Satellite image overlays.
- Model performance analysis.
- Dataset upload functionality.
- CSV data exports.
- HTML summary report generation.

## ⚖️ Combined Environmental Risk Assessment

Synthetix combines flood exposure and forest deficit into a district-level risk score.

The default formula is:

`Risk Score = 60% × Flood Exposure + 40% × Forest Deficit`

Risk categories:

- High Risk: Score ≥ 60
- Medium Risk: Score ≥ 35 and < 60
- Low Risk: Score < 35

The system also supports districts where only forest or flood information is available.

## 🛠️ Technologies Used

| Category | Technology |
|---|---|
| Programming Language | Python |
| Backend Framework | FastAPI |
| Frontend | HTML, JavaScript |
| Visualization | Plotly |
| Machine Learning | XGBoost |
| Data Processing | Pandas, NumPy |
| Model Loading | Joblib |
| Image Processing | Pillow |
| Web Server | Uvicorn |

## 🏗️ Application Architecture

```text
                 SYNT HETIX
                     |
          Forest & Flood Intelligence
                     |
             FastAPI Backend
                     |
          +----------+----------+
          |                     |
    Forest Intelligence   Flood Intelligence
          |                     |
     XGBoost Model         XGBoost Model
          |                     |

