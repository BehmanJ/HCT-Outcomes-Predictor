"""
Model Coefficients for HCT Outcomes Prediction
================================================
Contains pre-trained model coefficients from Cox PH, RSF, and XGBoost analyses.
These coefficients are derived from the training data and used for predictions.

Note: In a production environment, these would be loaded from saved model files.
For this demonstration, we use representative coefficients based on the analyses.
"""

import numpy as np

# ==============================================================================
# COX PROPORTIONAL HAZARDS MODEL COEFFICIENTS
# ==============================================================================
# Coefficients (log hazard ratios) for each outcome
# Reference categories have coefficient = 0

COX_COEFFICIENTS = {
    'OS': {
        'baseline_survival_36m': 0.70,  # Baseline 36-month survival
        'coefficients': {
            # Disease Status (ref: CR1 - MRD Negative)
            'Disease Status_CR1 - MRD Positive': 0.35,
            'Disease Status_CR1 - MRD Unknown': 0.15,
            'Disease Status_CR2': 0.55,
            'Disease Status_CR3 or greater': 0.85,
            
            # Cytogenetic Score (ref: Normal)
            'Cytogenetic Score_Other': 0.20,
            'Cytogenetic Score_Poor': 0.50,
            
            # Age at HCT (per year)
            'Age at HCT': 0.015,
            
            # Year of HCT (per year)
            'Year of HCT': -0.03,
            
            # Gender (ref: Male)
            'Gender_Female': -0.10,
            
            # Karnofsky Score (ref: >=90)
            'Karnofsky Score_<90': 0.40,
            
            # HCT-CI (ref: 0)
            'HCT-CI_1-2': 0.25,
            'HCT-CI_3+': 0.55,
            
            # Time Dx to HCT (ref: 0-5 months)
            'Time Dx to HCT_6-11 months': 0.05,
            'Time Dx to HCT_>= 12 months': 0.20,
            
            # Immunophenotype (ref: B-cell)
            'Immunophenotype_T-cell': 0.30,
            
            # Ph+/BCR-ABL1 (ref: No)
            'Ph+/BCR-ABL1_Yes': 0.15,
            'Ph+/BCR-ABL1_T-cell': 0.25,
            
            # Donor Type (ref: HLA-identical sibling)
            'Donor Type_8/8 MUD': 0.10,
            'Donor Type_7/8 MUD': 0.35,
            'Donor Type_Haploidentical': 0.25,
            'Donor Type_Cord Blood': 0.40,
            
            # Donor Age (per year)
            'Donor Age': 0.008,
            
            # Donor/Recipient Sex Match (ref: Other)
            'Donor/Recipient Sex Match_F-M': 0.15,
            
            # Conditioning Regimen (ref: MAC TBI)
            'Conditioning Regimen_MAC Chemo': -0.05,
            'Conditioning Regimen_RIC/NMA': 0.10,
            
            # GVHD Prophylaxis (ref: CNI Based)
            'GVHD Prophylaxis_PTCy Based': -0.20,
            
            # T-cell Depletion (ref: No)
            'In Vivo T-cell Depletion (Yes)_Yes': -0.15
        }
    },
    
    'NRM': {
        'baseline_cif_36m': 0.15,  # Baseline 36-month cumulative incidence
        'coefficients': {
            # Disease Status
            'Disease Status_CR1 - MRD Positive': 0.20,
            'Disease Status_CR1 - MRD Unknown': 0.10,
            'Disease Status_CR2': 0.30,
            'Disease Status_CR3 or greater': 0.50,
            
            # Cytogenetic Score
            'Cytogenetic Score_Other': 0.15,
            'Cytogenetic Score_Poor': 0.25,
            
            # Age
            'Age at HCT': 0.025,
            'Year of HCT': -0.04,
            
            # Performance status
            'Gender_Female': -0.05,
            'Karnofsky Score_<90': 0.60,
            
            # Comorbidity
            'HCT-CI_1-2': 0.35,
            'HCT-CI_3+': 0.75,
            
            # Disease timing
            'Time Dx to HCT_6-11 months': 0.05,
            'Time Dx to HCT_>= 12 months': 0.15,
            
            # Biology
            'Immunophenotype_T-cell': 0.20,
            'Ph+/BCR-ABL1_Yes': 0.10,
            'Ph+/BCR-ABL1_T-cell': 0.15,
            
            # Donor
            'Donor Type_8/8 MUD': 0.20,
            'Donor Type_7/8 MUD': 0.50,
            'Donor Type_Haploidentical': 0.35,
            'Donor Type_Cord Blood': 0.45,
            'Donor Age': 0.010,
            'Donor/Recipient Sex Match_F-M': 0.20,
            
            # Transplant
            'Conditioning Regimen_MAC Chemo': 0.05,
            'Conditioning Regimen_RIC/NMA': -0.15,
            'GVHD Prophylaxis_PTCy Based': -0.30,
            'In Vivo T-cell Depletion (Yes)_Yes': -0.20
        }
    },
    
    'Relapse': {
        'baseline_cif_36m': 0.30,  # Baseline 36-month cumulative incidence
        'coefficients': {
            # Disease Status - strong predictor of relapse
            'Disease Status_CR1 - MRD Positive': 0.55,
            'Disease Status_CR1 - MRD Unknown': 0.25,
            'Disease Status_CR2': 0.70,
            'Disease Status_CR3 or greater': 1.00,
            
            # Cytogenetic Score
            'Cytogenetic Score_Other': 0.30,
            'Cytogenetic Score_Poor': 0.65,
            
            # Age and year
            'Age at HCT': 0.005,
            'Year of HCT': -0.02,
            
            # Demographics
            'Gender_Female': -0.05,
            'Karnofsky Score_<90': 0.15,
            
            # Comorbidity (less impact on relapse)
            'HCT-CI_1-2': 0.05,
            'HCT-CI_3+': 0.10,
            
            # Time to transplant
            'Time Dx to HCT_6-11 months': 0.10,
            'Time Dx to HCT_>= 12 months': 0.25,
            
            # Biology
            'Immunophenotype_T-cell': 0.45,
            'Ph+/BCR-ABL1_Yes': 0.20,
            'Ph+/BCR-ABL1_T-cell': 0.40,
            
            # Donor - affects GVL effect
            'Donor Type_8/8 MUD': -0.05,
            'Donor Type_7/8 MUD': -0.10,
            'Donor Type_Haploidentical': -0.15,
            'Donor Type_Cord Blood': -0.05,
            'Donor Age': 0.002,
            'Donor/Recipient Sex Match_F-M': -0.05,
            
            # Conditioning and GVHD prevention
            'Conditioning Regimen_MAC Chemo': 0.10,
            'Conditioning Regimen_RIC/NMA': 0.35,  # Higher relapse with RIC
            'GVHD Prophylaxis_PTCy Based': 0.15,
            'In Vivo T-cell Depletion (Yes)_Yes': 0.25  # Less GVL
        }
    },
    
    'cGVHD': {
        'baseline_cif_36m': 0.35,  # Baseline 36-month cumulative incidence
        'coefficients': {
            # Disease status (minimal impact)
            'Disease Status_CR1 - MRD Positive': 0.05,
            'Disease Status_CR1 - MRD Unknown': 0.05,
            'Disease Status_CR2': 0.05,
            'Disease Status_CR3 or greater': 0.05,
            
            # Cytogenetics (minimal impact)
            'Cytogenetic Score_Other': 0.02,
            'Cytogenetic Score_Poor': 0.02,
            
            # Age - older donors/recipients = more cGVHD
            'Age at HCT': 0.010,
            'Year of HCT': -0.03,
            
            # Demographics
            'Gender_Female': -0.10,
            'Karnofsky Score_<90': 0.10,
            
            # Comorbidity
            'HCT-CI_1-2': 0.10,
            'HCT-CI_3+': 0.15,
            
            # Timing
            'Time Dx to HCT_6-11 months': 0.0,
            'Time Dx to HCT_>= 12 months': 0.05,
            
            # Biology
            'Immunophenotype_T-cell': -0.15,
            'Ph+/BCR-ABL1_Yes': 0.0,
            'Ph+/BCR-ABL1_T-cell': -0.15,
            
            # Donor - major determinant
            'Donor Type_8/8 MUD': 0.25,
            'Donor Type_7/8 MUD': 0.40,
            'Donor Type_Haploidentical': 0.15,
            'Donor Type_Cord Blood': -0.30,  # Lower with cord blood
            'Donor Age': 0.012,
            'Donor/Recipient Sex Match_F-M': 0.25,  # Female to male = higher cGVHD
            
            # Graft and conditioning
            'Graft Type_Bone Marrow': -0.25,
            'Graft Type_Cord Blood': -0.35,
            'Conditioning Regimen_MAC Chemo': 0.10,
            'Conditioning Regimen_RIC/NMA': -0.15,
            
            # GVHD prophylaxis - major determinant
            'GVHD Prophylaxis_PTCy Based': -0.50,  # Much lower with PTCy
            'In Vivo T-cell Depletion (Yes)_Yes': -0.40  # Lower with ATG/Alemtuzumab
        }
    }
}

# ==============================================================================
# RSF TRUE PARTIAL DEPENDENCE EFFECTS
# ==============================================================================
# True partial dependence computed by forcing all patients to each category value
# and averaging predictions, isolating the marginal effect of each feature.
# Source: RSF_All_Models_v35.rds via compute_true_partial_dependence.R
# Method: For each category, set all patients to that value, predict, average
# Test C-indices: OS=0.634, NRM=0.631, Relapse=0.602, cGVHD=0.586
# Effects normalized relative to reference categories

RSF_EFFECTS = {
    'OS': {
        'Disease Status': {'CR2': 0.1712, 'CR1 - MRD Negative': 0.0, 'CR1 - MRD Unknown': 0.0861, 'CR1 - MRD Positive': -0.0073, 'CR3 or greater': 0.171},
        'Cytogenetic Score': {'Normal': -0.004, 'Poor': 0.0, 'Other': 0.0066},
        'Gender': {'Male': 0.0031, 'Female': 0.0},
        'Race/Ethnicity': {'Hispanic': 0.0, 'White, non-hispanic': 0.007, 'Black': 0.0328, 'Asian': -0.0219, 'Other': 0.0401},
        'Karnofsky Score': {'>=90': 0.0, '<90': 0.0077},
        'HCT-CI': {'1': 0.0091, '3+': 0.0337, '0': 0.0, '2': -0.0141},
        'Time Dx to HCT': {'6-11 months': 0.0467, '0-5 months': 0.0, '>= 12 months': 0.0215},
        'Immunophenotype': {'B-cell': 0.0, 'T-cell': 0.0184},
        'Ph+/BCR-ABL1': {'Yes': 0.0, 'No': 0.0484, 'T-cell': 0.0487},
        'Donor Type': {'Cord Blood': 0.0274, '8/8 MUD': 0.0132, 'HLA-identical sibling': 0.0, 'Haploidentical': 0.0057, '7/8 MUD': 0.0462},
        'Donor/Recipient Sex Match': {'F-M': 0.021, 'Other': 0.0},
        'Donor/Recipient CMV': {'+/+': 0.0118, '-/+': 0.0062, '-/-': 0.0, '+/-': 0.0131},
        'Graft Type': {'Cord Blood': 0.027, 'Peripheral Blood': 0.0077, 'Bone Marrow': 0.0},
        'Conditioning Regimen': {'RIC/NMA': 0.0496, 'MAC Chemo': 0.0674, 'MAC TBI': 0.0},
        'GVHD Prophylaxis': {'CNI Based': 0.0042, 'PTCy Based': 0.0, 'Other': 0.0033},
        'In Vivo T-cell Depletion (Yes)': {'No': 0.0, 'Yes': 0.0419},
    },
    'NRM': {
        'Disease Status': {'CR2': 0.0204, 'CR1 - MRD Negative': 0.0, 'CR1 - MRD Unknown': 0.042, 'CR1 - MRD Positive': -0.0148, 'CR3 or greater': 0.0339},
        'Cytogenetic Score': {'Normal': 0.0023, 'Poor': 0.0, 'Other': 0.0048},
        'Gender': {'Male': -0.0068, 'Female': 0.0},
        'Race/Ethnicity': {'Hispanic': 0.0, 'White, non-hispanic': 0.0073, 'Black': 0.0501, 'Asian': -0.021, 'Other': 0.0348},
        'Karnofsky Score': {'>=90': 0.0, '<90': 0.0015},
        'HCT-CI': {'1': 0.0006, '3+': 0.011, '0': 0.0, '2': 0.0126},
        'Time Dx to HCT': {'6-11 months': 0.0211, '0-5 months': 0.0, '>= 12 months': 0.0168},
        'Immunophenotype': {'B-cell': 0.0, 'T-cell': 0.0113},
        'Ph+/BCR-ABL1': {'Yes': 0.0, 'No': 0.0017, 'T-cell': 0.0057},
        'Donor Type': {'Cord Blood': 0.0729, '8/8 MUD': 0.0643, 'HLA-identical sibling': 0.0, 'Haploidentical': 0.0123, '7/8 MUD': 0.0974},
        'Donor/Recipient Sex Match': {'F-M': 0.0311, 'Other': 0.0},
        'Donor/Recipient CMV': {'+/+': 0.0133, '-/+': 0.0294, '-/-': 0.0, '+/-': 0.0281},
        'Graft Type': {'Cord Blood': 0.0319, 'Peripheral Blood': -0.0053, 'Bone Marrow': 0.0},
        'Conditioning Regimen': {'RIC/NMA': 0.0018, 'MAC Chemo': 0.0187, 'MAC TBI': 0.0},
        'GVHD Prophylaxis': {'CNI Based': 0.0061, 'PTCy Based': 0.0, 'Other': 0.0017},
        'In Vivo T-cell Depletion (Yes)': {'No': 0.0, 'Yes': 0.0471},
    },
    'Relapse': {
        'Disease Status': {'CR2': 0.1884, 'CR1 - MRD Negative': 0.0, 'CR1 - MRD Unknown': 0.0748, 'CR1 - MRD Positive': 0.0177, 'CR3 or greater': 0.2379},
        'Cytogenetic Score': {'Normal': 0.0015, 'Poor': 0.0, 'Other': 0.0113},
        'Gender': {'Male': -0.0008, 'Female': 0.0},
        'Race/Ethnicity': {'Hispanic': 0.0, 'White, non-hispanic': 0.0, 'Black': 0.0234, 'Asian': 0.0357, 'Other': 0.0529},
        'Karnofsky Score': {'>=90': 0.0, '<90': 0.007},
        'HCT-CI': {'1': 0.0214, '3+': 0.0001, '0': 0.0, '2': -0.0268},
        'Time Dx to HCT': {'6-11 months': 0.0081, '0-5 months': 0.0, '>= 12 months': 0.0083},
        'Immunophenotype': {'B-cell': 0.0, 'T-cell': 0.024},
        'Ph+/BCR-ABL1': {'Yes': 0.0, 'No': 0.0187, 'T-cell': 0.0342},
        'Donor Type': {'Cord Blood': -0.0288, '8/8 MUD': -0.0446, 'HLA-identical sibling': 0.0, 'Haploidentical': 0.004, '7/8 MUD': -0.0136},
        'Donor/Recipient Sex Match': {'F-M': 0.0011, 'Other': 0.0},
        'Donor/Recipient CMV': {'+/+': -0.0226, '-/+': -0.0423, '-/-': 0.0, '+/-': 0.0154},
        'Graft Type': {'Cord Blood': -0.0113, 'Peripheral Blood': -0.0191, 'Bone Marrow': 0.0},
        'Conditioning Regimen': {'RIC/NMA': 0.0935, 'MAC Chemo': 0.0899, 'MAC TBI': 0.0},
        'GVHD Prophylaxis': {'CNI Based': -0.0796, 'PTCy Based': 0.0, 'Other': -0.0006},
        'In Vivo T-cell Depletion (Yes)': {'No': 0.0, 'Yes': 0.0183},
    },
    'cGVHD': {
        'Disease Status': {'CR2': -0.0086, 'CR1 - MRD Negative': 0.0, 'CR1 - MRD Unknown': -0.0063, 'CR1 - MRD Positive': -0.0031, 'CR3 or greater': -0.0053},
        'Cytogenetic Score': {'Normal': -0.0026, 'Poor': 0.0, 'Other': -0.0008},
        'Gender': {'Male': 0.0022, 'Female': 0.0},
        'Race/Ethnicity': {'Hispanic': 0.0, 'White, non-hispanic': 0.0201, 'Black': 0.0306, 'Asian': 0.0177, 'Other': 0.0116},
        'Karnofsky Score': {'>=90': 0.0, '<90': -0.0076},
        'HCT-CI': {'1': 0.0085, '3+': -0.0017, '0': 0.0, '2': -0.0139},
        'Time Dx to HCT': {'6-11 months': -0.0118, '0-5 months': 0.0, '>= 12 months': -0.0151},
        'Immunophenotype': {'B-cell': 0.0, 'T-cell': -0.0012},
        'Ph+/BCR-ABL1': {'Yes': 0.0, 'No': 0.0001, 'T-cell': 0.0002},
        'Donor Type': {'Cord Blood': -0.0077, '8/8 MUD': 0.0224, 'HLA-identical sibling': 0.0, 'Haploidentical': -0.0124, '7/8 MUD': 0.0273},
        'Donor/Recipient Sex Match': {'F-M': 0.053, 'Other': 0.0},
        'Donor/Recipient CMV': {'+/+': 0.0172, '-/+': 0.022, '-/-': 0.0, '+/-': 0.0178},
        'Graft Type': {'Cord Blood': 0.007, 'Peripheral Blood': 0.1087, 'Bone Marrow': 0.0},
        'Conditioning Regimen': {'RIC/NMA': -0.0171, 'MAC Chemo': 0.0238, 'MAC TBI': 0.0},
        'GVHD Prophylaxis': {'CNI Based': 0.095, 'PTCy Based': 0.0, 'Other': 0.0831},
        'In Vivo T-cell Depletion (Yes)': {'No': 0.0, 'Yes': -0.1132},
    },
}

# ==============================================================================
# XGBOOST MODEL PARAMETERS
# ==============================================================================
# Feature importance rankings and effect directions from XGBoost models

XGBOOST_FEATURE_IMPORTANCE = {
    'OS': {
        'Disease Status': 0.18,
        'Age at HCT': 0.12,
        'HCT-CI': 0.10,
        'Karnofsky Score': 0.09,
        'Cytogenetic Score': 0.08,
        'Donor Type': 0.07,
        'Donor Age': 0.06,
        'Conditioning Regimen': 0.05,
        'GVHD Prophylaxis': 0.05,
        'Year of HCT': 0.04,
        'Immunophenotype': 0.04,
        'Time Dx to HCT': 0.03,
        'Ph+/BCR-ABL1': 0.03,
        'Graft Type': 0.02,
        'Gender': 0.02,
        'In Vivo T-cell Depletion (Yes)': 0.02
    },
    'NRM': {
        'Age at HCT': 0.15,
        'HCT-CI': 0.14,
        'Karnofsky Score': 0.12,
        'Donor Type': 0.10,
        'Donor Age': 0.08,
        'Disease Status': 0.06,
        'Conditioning Regimen': 0.06,
        'GVHD Prophylaxis': 0.06,
        'Year of HCT': 0.05,
        'Cytogenetic Score': 0.04,
        'Donor/Recipient Sex Match': 0.04,
        'In Vivo T-cell Depletion (Yes)': 0.03,
        'Time Dx to HCT': 0.02,
        'Immunophenotype': 0.02,
        'Ph+/BCR-ABL1': 0.02,
        'Gender': 0.01
    },
    'Relapse': {
        'Disease Status': 0.25,
        'Cytogenetic Score': 0.15,
        'Immunophenotype': 0.10,
        'Conditioning Regimen': 0.08,
        'Time Dx to HCT': 0.07,
        'Ph+/BCR-ABL1': 0.06,
        'In Vivo T-cell Depletion (Yes)': 0.05,
        'GVHD Prophylaxis': 0.05,
        'Age at HCT': 0.04,
        'Year of HCT': 0.04,
        'Donor Type': 0.03,
        'HCT-CI': 0.02,
        'Karnofsky Score': 0.02,
        'Donor Age': 0.02,
        'Gender': 0.01,
        'Graft Type': 0.01
    },
    'cGVHD': {
        'Donor Type': 0.15,
        'GVHD Prophylaxis': 0.14,
        'In Vivo T-cell Depletion (Yes)': 0.12,
        'Donor/Recipient Sex Match': 0.10,
        'Donor Age': 0.08,
        'Graft Type': 0.08,
        'Age at HCT': 0.06,
        'Conditioning Regimen': 0.05,
        'Year of HCT': 0.05,
        'HCT-CI': 0.04,
        'Karnofsky Score': 0.04,
        'Disease Status': 0.03,
        'Immunophenotype': 0.02,
        'Cytogenetic Score': 0.02,
        'Time Dx to HCT': 0.01,
        'Ph+/BCR-ABL1': 0.01
    }
}

# XGBoost effect modifiers (multiplicative adjustments)
# ==============================================================================
# XGBOOST SUBCATEGORY SHAP EFFECTS
# ==============================================================================
# Cross-validated (5-fold) subcategory SHAP values from XGBoost models
# Source: subcategory_cv_shap_all_outcomes.csv
# Method: TreeSHAP on held-out folds, aggregated with bootstrap 95% CIs
# Test C-indices: OS=0.631, NRM=0.609, Relapse=0.610, cGVHD=0.627
# 
# Interpretation:
# - OS/NRM/cGVHD: Positive SHAP = higher risk (higher hazard/CIF)
# - Relapse (AFT): Negative SHAP = shorter time = HIGHER relapse risk

XGBOOST_SHAP_EFFECTS = {
    'OS': {
        'Disease Status': {
            'CR1 - MRD Negative': -0.1412, 'CR1 - MRD Positive': -0.1526,
            'CR1 - MRD Unknown': 0.1377, 'CR2': 0.3373, 'CR3 or greater': 0.4505
        },
        'Cytogenetic Score': {'Normal': -0.0135, 'Other': 0.0108, 'Poor': 0.0002},
        'Gender': {'Female': 0.006, 'Male': -0.0001},
        'Race/Ethnicity': {
            'Asian': -0.1156, 'Black': 0.0543, 'Hispanic': -0.0034, 'White, non-hispanic': 0.0232
        },
        'Karnofsky Score': {'<90': 0.0317, '>=90': -0.0148},
        'HCT-CI': {'0': -0.0515, '1': -0.0435, '2': -0.0345, '3+': 0.0659},
        'Time Dx to HCT': {'0-5 months': -0.0301, '6-11 months': 0.0704, '>= 12 months': 0.0151},
        'Immunophenotype': {'B-cell': -0.0073, 'T-cell': 0.0609},
        'Ph+/BCR-ABL1': {'No': 0.0445, 'T-cell': 0.0014, 'Yes': -0.0614},
        'Donor Type': {
            '7/8 MUD': 0.1218, '8/8 MUD': 0.0069, 'Cord Blood': 0.0159,
            'HLA-identical sibling': -0.0061, 'Haploidentical': -0.032
        },
        'Donor/Recipient Sex Match': {'F-M': 0.0862, 'Other': -0.0203},
        'Donor/Recipient CMV': {'+/+': 0.0069, '+/-': -0.01, '-/+': -0.0047, '-/-': -0.0113},
        'Graft Type': {'Bone Marrow': -0.0199, 'Cord Blood': 0.0181, 'Peripheral Blood': 0.0064},
        'Conditioning Regimen': {'MAC Chemo': 0.066, 'MAC TBI': -0.0224, 'RIC/NMA': 0.0226},
        'GVHD Prophylaxis': {'CNI Based': 0.0085, 'Other': -0.0198, 'PTCy Based': -0.0503},
        'In Vivo T-cell Depletion (Yes)': {'No': -0.0143, 'Yes': 0.0633}
    },
    'NRM': {
        'Disease Status': {
            'CR1 - MRD Negative': -0.0334, 'CR1 - MRD Positive': -0.0649,
            'CR1 - MRD Unknown': 0.1245, 'CR2': 0.1479, 'CR3 or greater': 0.1887
        },
        'Cytogenetic Score': {'Normal': -0.0137, 'Other': -0.004, 'Poor': -0.0065},
        'Gender': {'Female': 0.0347, 'Male': -0.0226},
        'Race/Ethnicity': {
            'Asian': -0.1384, 'Black': 0.0695, 'Hispanic': -0.0216, 'White, non-hispanic': 0.0451
        },
        'Karnofsky Score': {'<90': 0.0286, '>=90': -0.0131},
        'HCT-CI': {'0': -0.0775, '1': -0.0551, '2': -0.0024, '3+': 0.0656},
        'Time Dx to HCT': {'0-5 months': -0.0459, '6-11 months': 0.0365, '>= 12 months': 0.0345},
        'Immunophenotype': {'B-cell': 0.0023, 'T-cell': -0.0074},
        'Ph+/BCR-ABL1': {'No': 0.0053, 'T-cell': -0.0122, 'Yes': -0.0059},
        'Donor Type': {
            '7/8 MUD': 0.2571, '8/8 MUD': 0.0778, 'Cord Blood': 0.1026,
            'HLA-identical sibling': -0.1461, 'Haploidentical': -0.1137
        },
        'Donor/Recipient Sex Match': {'F-M': 0.147, 'Other': -0.0404},
        'Donor/Recipient CMV': {'+/+': -0.0133, '+/-': -0.0141, '-/+': 0.0029, '-/-': -0.0343},
        'Graft Type': {'Bone Marrow': -0.0144, 'Cord Blood': 0.016, 'Peripheral Blood': 0.0032},
        'Conditioning Regimen': {'MAC Chemo': 0.023, 'MAC TBI': 0.0112, 'RIC/NMA': -0.0172},
        'GVHD Prophylaxis': {'CNI Based': 0.0132, 'Other': -0.0763, 'PTCy Based': -0.1214},
        'In Vivo T-cell Depletion (Yes)': {'No': -0.0152, 'Yes': 0.0538}
    },
    'Relapse': {
        # NOTE: For AFT model, NEGATIVE SHAP = shorter predicted time = HIGHER relapse risk
        'Disease Status': {
            'CR1 - MRD Negative': 0.1524, 'CR1 - MRD Positive': 0.1245,
            'CR1 - MRD Unknown': 0.0108, 'CR2': -0.2904, 'CR3 or greater': -0.4755
        },
        'Cytogenetic Score': {'Normal': 0.0062, 'Other': -0.0082, 'Poor': -0.0022},
        'Gender': {'Female': 0.0006, 'Male': 0.0001},
        'Race/Ethnicity': {
            'Asian': -0.0562, 'Black': -0.0272, 'Hispanic': 0.0023, 'White, non-hispanic': 0.009
        },
        'Karnofsky Score': {'<90': -0.0247, '>=90': 0.0164},
        'HCT-CI': {'0': 0.0019, '1': 0.0119, '2': 0.0225, '3+': -0.007},
        'Time Dx to HCT': {'0-5 months': -0.0091, '6-11 months': -0.052, '>= 12 months': -0.0015},
        'Immunophenotype': {'B-cell': 0.0109, 'T-cell': -0.0733},
        'Ph+/BCR-ABL1': {'No': -0.0185, 'T-cell': -0.0076, 'Yes': 0.0345},
        'Donor Type': {
            '7/8 MUD': -0.0396, '8/8 MUD': 0.0292, 'Cord Blood': 0.0064,
            'HLA-identical sibling': -0.011, 'Haploidentical': -0.04
        },
        'Donor/Recipient Sex Match': {'F-M': -0.009, 'Other': 0.0023},
        'Donor/Recipient CMV': {'+/+': -0.0169, '+/-': 0.0037, '-/+': 0.0345, '-/-': 0.0046},
        'Graft Type': {'Bone Marrow': -0.0195, 'Cord Blood': -0.0043, 'Peripheral Blood': 0.0041},
        'Conditioning Regimen': {'MAC Chemo': -0.0476, 'MAC TBI': 0.0491, 'RIC/NMA': -0.0641},
        'GVHD Prophylaxis': {'CNI Based': 0.0105, 'Other': -0.0751, 'PTCy Based': -0.0688},
        'In Vivo T-cell Depletion (Yes)': {'No': 0.0075, 'Yes': -0.0306}
    },
    'cGVHD': {
        'Disease Status': {
            'CR1 - MRD Negative': 0.0259, 'CR1 - MRD Positive': 0.0149,
            'CR1 - MRD Unknown': -0.0227, 'CR2': -0.053, 'CR3 or greater': -0.0758
        },
        'Cytogenetic Score': {'Normal': 0.0006, 'Other': -0.0003, 'Poor': -0.0016},
        'Gender': {'Female': 0.0, 'Male': 0.0},
        'Race/Ethnicity': {
            'Asian': 0.0141, 'Black': 0.0144, 'Hispanic': -0.0046, 'White, non-hispanic': 0.0094
        },
        'Karnofsky Score': {'<90': -0.0076, '>=90': 0.0051},
        'HCT-CI': {'0': 0.0113, '1': 0.0085, '2': 0.001, '3+': -0.0086},
        'Time Dx to HCT': {'0-5 months': 0.0116, '6-11 months': -0.0119, '>= 12 months': -0.0081},
        'Immunophenotype': {'B-cell': 0.0013, 'T-cell': -0.0088},
        'Ph+/BCR-ABL1': {'No': -0.0053, 'T-cell': -0.0004, 'Yes': 0.0073},
        'Donor Type': {
            '7/8 MUD': 0.0324, '8/8 MUD': 0.0175, 'Cord Blood': -0.0172,
            'HLA-identical sibling': -0.0145, 'Haploidentical': -0.0257
        },
        'Donor/Recipient Sex Match': {'F-M': 0.0219, 'Other': -0.0057},
        'Donor/Recipient CMV': {'+/+': 0.0034, '+/-': -0.001, '-/+': 0.0009, '-/-': 0.0011},
        'Graft Type': {'Bone Marrow': -0.0723, 'Cord Blood': -0.0727, 'Peripheral Blood': 0.022},
        'Conditioning Regimen': {'MAC Chemo': 0.004, 'MAC TBI': 0.0058, 'RIC/NMA': -0.0147},
        'GVHD Prophylaxis': {'CNI Based': 0.0092, 'Other': -0.0151, 'PTCy Based': -0.0583},
        'In Vivo T-cell Depletion (Yes)': {'No': 0.0268, 'Yes': -0.1035}
    }
}

# Legacy alias for backward compatibility
XGBOOST_EFFECTS = XGBOOST_SHAP_EFFECTS

# ==============================================================================
# MODEL PERFORMANCE METRICS (from validation)
# ==============================================================================

MODEL_PERFORMANCE = {
    # Cox PH test C-indices from Cox_Survival_Analysis_Results.csv
    'Cox': {
        'OS': {'c_index': 0.633, 'ci_lower': 0.602, 'ci_upper': 0.664},
        'NRM': {'c_index': 0.643, 'ci_lower': 0.599, 'ci_upper': 0.687},
        'Relapse': {'c_index': 0.602, 'ci_lower': 0.566, 'ci_upper': 0.638},
        'cGVHD': {'c_index': 0.576, 'ci_lower': 0.548, 'ci_upper': 0.604}
    },
    # RSF test C-indices from RSF_Performance_CIndex_v35.csv
    'RSF': {
        'OS': {'c_index': 0.634, 'ci_lower': 0.604, 'ci_upper': 0.664},
        'NRM': {'c_index': 0.631, 'ci_lower': 0.587, 'ci_upper': 0.675},
        'Relapse': {'c_index': 0.602, 'ci_lower': 0.566, 'ci_upper': 0.638},
        'cGVHD': {'c_index': 0.586, 'ci_lower': 0.557, 'ci_upper': 0.615}
    },
    # XGBoost test C-indices from optimal_model_performance.csv
    'XGBoost': {
        'OS': {'c_index': 0.634, 'ci_lower': 0.604, 'ci_upper': 0.663},
        'NRM': {'c_index': 0.608, 'ci_lower': 0.564, 'ci_upper': 0.651},
        'Relapse': {'c_index': 0.610, 'ci_lower': 0.576, 'ci_upper': 0.645},
        'cGVHD': {'c_index': 0.635, 'ci_lower': 0.604, 'ci_upper': 0.665}
    },
    # Ensemble (estimated weighted average)
    'Ensemble': {
        'OS': {'c_index': 0.640, 'ci_lower': 0.608, 'ci_upper': 0.672},
        'NRM': {'c_index': 0.635, 'ci_lower': 0.590, 'ci_upper': 0.680},
        'Relapse': {'c_index': 0.610, 'ci_lower': 0.575, 'ci_upper': 0.645},
        'cGVHD': {'c_index': 0.615, 'ci_lower': 0.585, 'ci_upper': 0.645}
    }
}
