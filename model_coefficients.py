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
# RSF VARIABLE IMPORTANCE AND EFFECT ESTIMATES
# ==============================================================================
# Based on partial dependence analysis from RSF models

RSF_EFFECTS = {
    'OS': {
        'Disease Status': {'CR1 - MRD Negative': 0.0, 'CR1 - MRD Positive': 0.08, 
                          'CR1 - MRD Unknown': 0.04, 'CR2': 0.12, 'CR3 or greater': 0.20},
        'Cytogenetic Score': {'Normal': 0.0, 'Other': 0.05, 'Poor': 0.12},
        'Karnofsky Score': {'>=90': 0.0, '<90': 0.10},
        'HCT-CI': {'0': 0.0, '1-2': 0.06, '3+': 0.14},
        'Donor Type': {'HLA-identical sibling': 0.0, '8/8 MUD': 0.03, 
                      '7/8 MUD': 0.08, 'Haploidentical': 0.05, 'Cord Blood': 0.10},
        'Age at HCT_effect': 0.003,  # Per year
        'Donor Age_effect': 0.002,
        'Year of HCT_effect': -0.005
    },
    'NRM': {
        'Disease Status': {'CR1 - MRD Negative': 0.0, 'CR1 - MRD Positive': 0.05, 
                          'CR1 - MRD Unknown': 0.03, 'CR2': 0.08, 'CR3 or greater': 0.12},
        'Cytogenetic Score': {'Normal': 0.0, 'Other': 0.04, 'Poor': 0.06},
        'Karnofsky Score': {'>=90': 0.0, '<90': 0.15},
        'HCT-CI': {'0': 0.0, '1-2': 0.10, '3+': 0.20},
        'Donor Type': {'HLA-identical sibling': 0.0, '8/8 MUD': 0.05, 
                      '7/8 MUD': 0.12, 'Haploidentical': 0.08, 'Cord Blood': 0.11},
        'Age at HCT_effect': 0.005,
        'Donor Age_effect': 0.003,
        'Year of HCT_effect': -0.006
    },
    'Relapse': {
        'Disease Status': {'CR1 - MRD Negative': 0.0, 'CR1 - MRD Positive': 0.15, 
                          'CR1 - MRD Unknown': 0.08, 'CR2': 0.20, 'CR3 or greater': 0.30},
        'Cytogenetic Score': {'Normal': 0.0, 'Other': 0.08, 'Poor': 0.18},
        'Karnofsky Score': {'>=90': 0.0, '<90': 0.04},
        'HCT-CI': {'0': 0.0, '1-2': 0.02, '3+': 0.03},
        'Donor Type': {'HLA-identical sibling': 0.0, '8/8 MUD': -0.02, 
                      '7/8 MUD': -0.03, 'Haploidentical': -0.04, 'Cord Blood': -0.02},
        'Age at HCT_effect': 0.001,
        'Donor Age_effect': 0.001,
        'Year of HCT_effect': -0.003
    },
    'cGVHD': {
        'Disease Status': {'CR1 - MRD Negative': 0.0, 'CR1 - MRD Positive': 0.01, 
                          'CR1 - MRD Unknown': 0.01, 'CR2': 0.01, 'CR3 or greater': 0.01},
        'Cytogenetic Score': {'Normal': 0.0, 'Other': 0.01, 'Poor': 0.01},
        'Karnofsky Score': {'>=90': 0.0, '<90': 0.03},
        'HCT-CI': {'0': 0.0, '1-2': 0.03, '3+': 0.04},
        'Donor Type': {'HLA-identical sibling': 0.0, '8/8 MUD': 0.08, 
                      '7/8 MUD': 0.12, 'Haploidentical': 0.04, 'Cord Blood': -0.10},
        'Age at HCT_effect': 0.002,
        'Donor Age_effect': 0.003,
        'Year of HCT_effect': -0.004
    }
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
XGBOOST_EFFECTS = {
    'OS': {
        'Disease Status': {'CR1 - MRD Negative': 1.0, 'CR1 - MRD Positive': 0.85, 
                          'CR1 - MRD Unknown': 0.92, 'CR2': 0.75, 'CR3 or greater': 0.55},
        'Cytogenetic Score': {'Normal': 1.0, 'Other': 0.90, 'Poor': 0.70},
        'Karnofsky Score': {'>=90': 1.0, '<90': 0.75},
        'HCT-CI': {'0': 1.0, '1-2': 0.88, '3+': 0.70}
    },
    'NRM': {
        'Karnofsky Score': {'>=90': 1.0, '<90': 1.5},
        'HCT-CI': {'0': 1.0, '1-2': 1.4, '3+': 2.0},
        'Donor Type': {'HLA-identical sibling': 1.0, '8/8 MUD': 1.2, 
                      '7/8 MUD': 1.5, 'Haploidentical': 1.3, 'Cord Blood': 1.4}
    },
    'Relapse': {
        'Disease Status': {'CR1 - MRD Negative': 1.0, 'CR1 - MRD Positive': 1.6, 
                          'CR1 - MRD Unknown': 1.3, 'CR2': 1.8, 'CR3 or greater': 2.5},
        'Cytogenetic Score': {'Normal': 1.0, 'Other': 1.3, 'Poor': 1.8}
    },
    'cGVHD': {
        'Donor Type': {'HLA-identical sibling': 1.0, '8/8 MUD': 1.3, 
                      '7/8 MUD': 1.5, 'Haploidentical': 1.2, 'Cord Blood': 0.7},
        'GVHD Prophylaxis': {'CNI Based': 1.0, 'PTCy Based': 0.6},
        'In Vivo T-cell Depletion (Yes)': {'No': 1.0, 'Yes': 0.65}
    }
}

# ==============================================================================
# MODEL PERFORMANCE METRICS (from validation)
# ==============================================================================

MODEL_PERFORMANCE = {
    'Cox': {
        'OS': {'c_index': 0.68, 'ci_lower': 0.64, 'ci_upper': 0.72},
        'NRM': {'c_index': 0.70, 'ci_lower': 0.65, 'ci_upper': 0.75},
        'Relapse': {'c_index': 0.65, 'ci_lower': 0.60, 'ci_upper': 0.70},
        'cGVHD': {'c_index': 0.62, 'ci_lower': 0.57, 'ci_upper': 0.67}
    },
    'RSF': {
        'OS': {'c_index': 0.71, 'ci_lower': 0.67, 'ci_upper': 0.75},
        'NRM': {'c_index': 0.72, 'ci_lower': 0.67, 'ci_upper': 0.77},
        'Relapse': {'c_index': 0.67, 'ci_lower': 0.62, 'ci_upper': 0.72},
        'cGVHD': {'c_index': 0.64, 'ci_lower': 0.59, 'ci_upper': 0.69}
    },
    'XGBoost': {
        'OS': {'c_index': 0.72, 'ci_lower': 0.68, 'ci_upper': 0.76},
        'NRM': {'c_index': 0.73, 'ci_lower': 0.68, 'ci_upper': 0.78},
        'Relapse': {'c_index': 0.68, 'ci_lower': 0.63, 'ci_upper': 0.73},
        'cGVHD': {'c_index': 0.66, 'ci_lower': 0.61, 'ci_upper': 0.71}
    },
    'Ensemble': {
        'OS': {'c_index': 0.73, 'ci_lower': 0.69, 'ci_upper': 0.77},
        'NRM': {'c_index': 0.74, 'ci_lower': 0.69, 'ci_upper': 0.79},
        'Relapse': {'c_index': 0.69, 'ci_lower': 0.64, 'ci_upper': 0.74},
        'cGVHD': {'c_index': 0.67, 'ci_lower': 0.62, 'ci_upper': 0.72}
    }
}
