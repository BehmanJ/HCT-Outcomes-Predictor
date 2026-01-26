"""
Configuration file for HCT Outcomes Ensemble Prediction App
Contains feature definitions, model parameters, and reference values
"""

# Feature columns as they appear in the data
FEATURE_COLUMNS = [
    'Disease Status', 'Cytogenetic Score', 'Year of HCT',
    'Age at HCT', 'Gender', 'Race/Ethnicity', 'Karnofsky Score', 'HCT-CI',
    'Time Dx to HCT', 'Immunophenotype', 'Ph+/BCR-ABL1', 'Donor Type',
    'Donor Age', 'Donor/Recipient Sex Match', 'Donor/Recipient CMV',
    'Graft Type', 'Conditioning Regimen', 'GVHD Prophylaxis',
    'In Vivo T-cell Depletion (Yes)'
]

# Categorical feature options
FEATURE_OPTIONS = {
    'Disease Status': [
        'CR1 - MRD Negative',
        'CR1 - MRD Positive', 
        'CR1 - MRD Unknown',
        'CR2',
        'CR3 or greater'
    ],
    'Cytogenetic Score': ['Normal', 'Other', 'Poor'],
    'Gender': ['Male', 'Female'],
    'Race/Ethnicity': ['White, non-hispanic', 'Black', 'Hispanic', 'Asian', 'Other'],
    'Karnofsky Score': ['>=90', '<90'],
    'HCT-CI': ['0', '1-2', '3+'],
    'Time Dx to HCT': ['0-5 months', '6-11 months', '>= 12 months'],
    'Immunophenotype': ['B-cell', 'T-cell'],
    'Ph+/BCR-ABL1': ['No', 'Yes', 'T-cell'],
    'Donor Type': [
        'HLA-identical sibling',
        '8/8 MUD',
        '7/8 MUD',
        'Haploidentical',
        'Cord Blood'
    ],
    'Donor/Recipient Sex Match': ['Other', 'F-M'],
    'Donor/Recipient CMV': ['-/-', '-/+', '+/-', '+/+'],
    'Graft Type': ['Peripheral Blood', 'Bone Marrow', 'Cord Blood'],
    'Conditioning Regimen': ['MAC TBI', 'MAC Chemo', 'RIC/NMA'],
    'GVHD Prophylaxis': ['CNI Based', 'PTCy Based'],
    'In Vivo T-cell Depletion (Yes)': ['No', 'Yes']
}

# Default/reference values for each categorical feature
REFERENCE_CATEGORIES = {
    'Disease Status': 'CR1 - MRD Negative',
    'Cytogenetic Score': 'Normal',
    'Gender': 'Male',
    'Race/Ethnicity': 'White, non-hispanic',
    'Karnofsky Score': '>=90',
    'HCT-CI': '0',
    'Time Dx to HCT': '0-5 months',
    'Immunophenotype': 'B-cell',
    'Ph+/BCR-ABL1': 'No',
    'Donor Type': 'HLA-identical sibling',
    'Donor/Recipient Sex Match': 'Other',
    'Donor/Recipient CMV': '-/-',
    'Graft Type': 'Peripheral Blood',
    'Conditioning Regimen': 'MAC TBI',
    'GVHD Prophylaxis': 'CNI Based',
    'In Vivo T-cell Depletion (Yes)': 'No'
}

# Numeric features with their ranges
NUMERIC_FEATURES = {
    'Year of HCT': {'min': 2011, 'max': 2018, 'default': 2015},
    'Age at HCT': {'min': 18, 'max': 80, 'default': 45},
    'Donor Age': {'min': 0, 'max': 75, 'default': 35}
}

# Outcomes configuration
OUTCOMES = {
    'OS': {
        'name': 'Overall Survival',
        'description': 'Probability of survival at 36 months',
        'color': '#2E86AB',
        'competing_risk': None,
        'interpretation': 'Higher score indicates higher survival probability'
    },
    'NRM': {
        'name': 'Non-Relapse Mortality',
        'description': 'Probability of death without relapse at 36 months',
        'color': '#E94F37',
        'competing_risk': 'Relapse',
        'interpretation': 'Higher score indicates higher NRM risk'
    },
    'Relapse': {
        'name': 'Relapse',
        'description': 'Probability of disease relapse at 36 months',
        'color': '#F18F01',
        'competing_risk': 'NRM',
        'interpretation': 'Higher score indicates higher relapse risk'
    },
    'cGVHD': {
        'name': 'Chronic GVHD',
        'description': 'Probability of chronic graft-versus-host disease at 36 months',
        'color': '#6B8E23',
        'competing_risk': 'Death without cGVHD',
        'interpretation': 'Higher score indicates higher cGVHD risk'
    }
}

# Model types and their descriptions
MODEL_TYPES = {
    'Cox': {
        'name': 'Cox Proportional Hazards',
        'description': 'Traditional semi-parametric survival model',
        'pros': ['Interpretable hazard ratios', 'Well-established methodology'],
        'cons': ['Assumes proportional hazards', 'May miss non-linear effects']
    },
    'RSF': {
        'name': 'Random Survival Forest',
        'description': 'Ensemble tree-based survival model',
        'pros': ['Captures non-linear relationships', 'Handles interactions'],
        'cons': ['Less interpretable', 'Computationally intensive']
    },
    'XGBoost': {
        'name': 'XGBoost Survival',
        'description': 'Gradient boosted survival model',
        'pros': ['High predictive accuracy', 'Handles missing data'],
        'cons': ['Can overfit', 'Less interpretable']
    }
}

# Ensemble weights (can be adjusted based on validation performance)
ENSEMBLE_WEIGHTS = {
    'OS': {'Cox': 0.33, 'RSF': 0.33, 'XGBoost': 0.34},
    'NRM': {'Cox': 0.33, 'RSF': 0.33, 'XGBoost': 0.34},
    'Relapse': {'Cox': 0.33, 'RSF': 0.33, 'XGBoost': 0.34},
    'cGVHD': {'Cox': 0.33, 'RSF': 0.33, 'XGBoost': 0.34}
}

# Risk stratification thresholds
RISK_THRESHOLDS = {
    'OS': {'low': 0.8, 'high': 0.6},  # Survival probability
    'NRM': {'low': 0.15, 'high': 0.30},  # Cumulative incidence
    'Relapse': {'low': 0.20, 'high': 0.40},  # Cumulative incidence
    'cGVHD': {'low': 0.25, 'high': 0.45}   # Cumulative incidence
}

# Time points for survival curves (months)
TIME_POINTS = [0, 6, 12, 18, 24, 30, 36]

LANDMARK_TIME = 36  # months
