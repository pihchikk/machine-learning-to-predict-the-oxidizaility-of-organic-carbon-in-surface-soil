## Introduction

Soil carbon storage plays a critical role in maintaining soil health and stability, particularly in forest-steppe ecosystems with significant carbon sequestration potential. Traditional methods for modeling soil organic matter variability are often complex, time-consuming, and limited in scope. This repository presents an innovative approach leveraging machine learning to enhance predictions of soil organic matter distribution. By integrating ground sampling data and time series analysis of spectral indices with high-resolution remote-sensing data, our approach aims to improve the accuracy and efficiency of soil carbon modeling compared to existing methods.

---

### Features

#### Data Pre-processing
1. **Retrieving Fraction of Days Covered with Snow Raster for Each Year**:
    - **Description**: Automated process to retrieve and compute the fraction of days covered with snow for each year using Sentinel-2 surface reflectance imagery.
    - **Details**: 
        - Masks clouds and edges in the imagery.
        - Computes the Normalized Difference Snow Index (NDSI).
        - Classifies snow cover based on NDSI and other spectral thresholds.
        - Calculates the fraction of days covered with snow for each pixel over the winter season.

2. **Retrieving Median April NDWI for Each Year**:
    - **Description**: Automated process to retrieve and compute the median Normalized Difference Water Index (NDWI) for April of each year.
    - **Details**:
        - Filters Sentinel-2 imagery for the month of April.
        - Masks clouds and edges in the imagery.
        - Computes the NDWI using relevant spectral bands.
        - Calculates the median NDWI for each pixel.

3. **Computing and Retrieving Phenology Covariates Rasters for Each Year**:
    - **Description**: Automated process to compute and retrieve phenology covariates for each year.
    - **Details**:
        - Filters Sentinel-2 imagery for relevant periods.
        - Computes phenological indices as described in https://collections.sentinel-hub.com/vegetation-phenology-and-productivity-parameters-season-1/ 
        - Generates time series from available cloudless images and extracts phenological metrics such as start of season, peak of season, and end of season.
#### Model Training
- **Machine Learning Models**: Python scripts for training models (LR, RF, PLSR Catboost) to predict soil organic matter variability.

#### Uncertainty Analysis
- **Bootstrap estimation of uncertainty**: Python scripts to estimate the uncertainty in soil organic matter predictions using bootstrap resampling techniques.

**TODO**:
- **QGIS Scripts for uncertainty estimation**: Tools to assess and visualize the uncertainty in soil organic matter predictions (empty for now)
- **Monte Carlo simulations**: Scripts to perform Monte Carlo simulations for robust uncertainty analysis. (empty for now)


### Objectives
- Show how the ecotone effect (the edge effect) increases uncertainty in spatial prediction of crucial ecological parameters
- Enhance the accuracy and efficiency of spatial soil organic matter modeling.
- Provide a comprehensive suite of tools for data pre-processing, EDA, and model training.
- Facilitate better understanding and management of soil health and carbon sequestration potential.

---

### Usage

- **Python Notebooks**: Jupyter notebooks for each stage of the data pre-processing and of the modeling process. Check the [notebooks](./notebooks) folder.
- **Data**: Example of a dataset used to train the model. Check the [data](./data) folder.
- **Pre-trained Models**: Pre-trained models available for immediate use or further customization. Check the [models](./models) folder. (empty for now) 
---

### Acknowledgements

Work is greatly supported by Non-commercial Foundation for the Advancement of Science and Education INTELLECT
