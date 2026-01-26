# HCT Outcomes Ensemble Prediction Web Application

A clinical prediction tool for estimating outcomes following allogeneic hematopoietic cell transplantation (HCT) for adult patients with acute lymphoblastic leukemia (ALL).

## Overview

This Streamlit-based web application combines predictions from three different survival modeling approaches:

1. **Cox Proportional Hazards (Cox PH)** - Traditional semi-parametric survival model
2. **Random Survival Forest (RSF)** - Ensemble tree-based survival analysis
3. **XGBoost Survival** - Gradient boosted machine learning approach (with trained models)

### Outcomes Predicted (at 36 months)

| Outcome | Description | Type |
|---------|-------------|------|
| **Overall Survival (OS)** | Probability of being alive | Survival |
| **Non-Relapse Mortality (NRM)** | Death without prior relapse | Competing risk (with relapse) |
| **Relapse** | Disease relapse | Competing risk (with NRM) |
| **Chronic GVHD** | Chronic graft-versus-host disease | Competing risk (with death w/o cGVHD) |

## Key Features

- **Ensemble Predictions**: Combines Cox PH, RSF, and XGBoost models
- **Trained XGBoost Models**: Uses actual trained models on patient data
- **What-If Comparison**: Compare outcomes between different treatment scenarios
- **Interactive Visualizations**: Survival curves, gauge charts, and comparisons
- **Model Interpretation**: Feature contributions and importance analysis

## Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Setup

1. Clone or download this repository

2. Navigate to the project directory:
   ```bash
   cd HCT_Ensemble_WebApp
   ```

3. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

Start the Streamlit server:

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## Usage

### Patient Input

Enter patient characteristics in the sidebar:

1. **Disease Characteristics**
   - Disease Status (CR1 MRD−, CR1 MRD+, CR2, etc.)
   - Cytogenetic Risk
   - Immunophenotype (B-cell/T-cell)
   - Philadelphia chromosome status

2. **Patient Characteristics**
   - Age at HCT
   - Gender
   - Performance status (Karnofsky)
   - Comorbidities (HCT-CI)

3. **Donor Characteristics**
   - Donor type
   - Donor age
   - Sex match
   - CMV status

4. **Transplant Characteristics**
   - Year of HCT
   - Graft source
   - Conditioning regimen
   - GVHD prophylaxis

### Output Tabs

1. **Predictions** - Risk scores, gauge charts, and model comparison
2. **Survival Curves** - Time-dependent outcome probabilities
3. **What-If Comparison** - Compare outcomes between different treatment scenarios
4. **Model Details** - Feature contributions and model performance
5. **About** - Information about the tool and current model status

### What-If Scenario Comparison

The What-If tab allows you to compare predicted outcomes between:
- Current patient scenario
- Alternative treatment approaches (donor type, conditioning, GVHD prophylaxis, etc.)

This helps clinicians explore "what if we used a different donor?" or "what if we added ATG?" questions with quantified predictions.

## Project Structure

```
HCT_Ensemble_WebApp/
├── app.py                    # Main Streamlit application
├── config.py                 # Configuration and feature definitions
├── model_coefficients.py     # Cox PH and RSF coefficients
├── model_loader.py           # XGBoost model loader
├── prediction_engine.py      # Prediction calculation logic
├── train_and_save_models.py  # Script to train XGBoost models
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── trained_models/           # Saved XGBoost models
    ├── xgboost_OS.json
    ├── xgboost_NRM.json
    ├── xgboost_Relapse.json
    ├── xgboost_cGVHD.json
    ├── label_encoders.pkl
    ├── preprocessing_info.json
    └── model_performance.csv
```

## Trained Models

The application includes pre-trained XGBoost models for all four outcomes. These models were trained on the ALL HCT dataset using:

- **Objective**: `survival:cox` (Cox partial likelihood)
- **Hyperparameter tuning**: Grid search with 3-fold cross-validation
- **Train/Test split**: 80/20 stratified split

### Model Performance (Test Set C-index)

| Outcome | C-index |
|---------|---------|
| Overall Survival | 0.629 |
| Non-Relapse Mortality | 0.603 |
| Relapse | 0.605 |
| Chronic GVHD | 0.565 |

### Retraining Models

To retrain the models with updated data:

```bash
# Place updated data file at the expected path
python3 train_and_save_models.py
```

## Model Details

### Cox PH Model
- Traditional survival analysis approach
- Provides interpretable hazard ratios
- Fine-Gray competing risks for NRM, Relapse, cGVHD

### Random Survival Forest
- Ensemble of survival trees
- Captures non-linear relationships and interactions
- Uses variable importance for feature ranking

### XGBoost Survival
- Gradient boosted survival analysis
- `survival:cox` objective for OS and NRM
- AFT (Accelerated Failure Time) for Relapse
- Fine-Gray approach for cGVHD

### Ensemble Approach
- Weighted average of individual model predictions
- Default weights: 33% Cox, 33% RSF, 34% XGBoost
- Weights can be adjusted based on validation performance

## Data Sources

Models were developed using:
- **Cox PH**: R `survival` and `cmprsk` packages
- **RSF**: R `randomForestSRC` package (version 3.5+)
- **XGBoost**: Python `xgboost` package with survival objectives

## Limitations

- Predictions are estimates based on historical data
- Individual outcomes may vary significantly
- Not validated for pediatric patients
- Should not replace clinical judgment
- Models may not generalize to different populations

## Disclaimer

**For research and educational purposes only.** 

This tool is not intended for clinical use. Treatment decisions should always be made in consultation with the patient's healthcare team based on individual circumstances, current medical knowledge, and clinical judgment.

## Technical Notes

### Extending the Models

To use actual trained models instead of coefficient-based predictions:

1. **For XGBoost**: Save models using `model.save_model()` and load with `xgb.XGBRegressor().load_model()`

2. **For RSF**: Export predictions from R using `saveRDS()` or implement `reticulate` bridge

3. **For Cox PH**: Use `lifelines` Python package to fit models or export coefficients from R

### Customizing Ensemble Weights

Modify `ENSEMBLE_WEIGHTS` in `config.py` to adjust model contributions based on validation performance.

## License

This project is for academic and research purposes.

## Contact

For questions about the methodology or implementation, please refer to the associated publication or contact the development team.
