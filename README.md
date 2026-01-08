# Cambodia-SBA-ML-Analysis
Code and scripts for the analysis of determinants of skilled birth attendance in Cambodia using DHS 2021–22 data, machine learning models, and explainable AI (SHAP).
# Cambodia-SBA-ML-R

Code and scripts for the analysis of determinants of skilled birth attendance (SBA) in Cambodia using DHS 2021–22 data, machine learning models, and explainable AI (SHAP and LIME).

## Overview

This repository contains the R code used for the paper:  
"Determinants of Skilled Birth Attendance in Cambodia: Evidence from the 2021–22 Demographic and Health Survey using Interpretable Machine Learning and Spatial Analysis" (submitted to BMJ Digital Health & AI).

The repository includes scripts for:  
- Machine learning analysis (`Salek_Cambodia(SBA).R`)  
- Feature selection and interpretability (Boruta, SHAP, LIME)  
- Provincial-level spatial mapping (`MPAS_Cambodia.R`)  

## Data

- Raw DHS data is not included due to data use restrictions.  
- DHS datasets can be requested from the DHS Program (https://dhsprogram.com/) following their data use agreement.  
- Simulated or anonymized example datasets are provided in the `data/` folder for demonstration.  

## Requirements

- R version 4.2+  
- Required R packages (install using `install.packages()`):

```r
install.packages(c(
  "tidyverse",
  "caret",
  "randomForest",
  "rpart",
  "e1071",
  "shapper",
  "lime",
  "sf",
  "tmap"
))



Usage
Step 1: Prepare Data

Place DHS data in the data/ folder (or use provided example data).

Step 2: Machine Learning Analysis

Run Salek_Cambodia(SBA).R to perform Random Forest, Decision Tree, Logistic Regression, KNN, and SVM models.

Step 3: Feature Selection

Run 02_feature_selection.R for Boruta-based feature selection.

Step 4: Model Interpretation

Run 04_shap_lime.R for SHAP and LIME interpretability of ML models.

Step 5: Spatial Mapping

Run MPAS_Cambodia.R to generate choropleth maps of SBA coverage across provinces.

Output

Model performance metrics (F1, accuracy, AUC)

SHAP and LIME summary plots

Spatial maps of provincial SBA coverage

Contact

For questions or clarifications, contact:
Md Salek Miah
Department of Statistics, Shahjalal University of Science and Technology, Sylhet-3114, Bangladesh
Email: saleksta@gmail.com


-------
**Key Notes:**
- `Salek_Cambodia(SBA).R` → ML modeling  
- `MPAS_Cambodia.R` → Spatial mapping  
- All outputs (plots, maps) go to `figures/`  
- README mentions **exact file names**, so reviewers know where to find analysis  

---
