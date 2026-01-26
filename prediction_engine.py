"""
Prediction Engine for HCT Outcomes Ensemble Model
===================================================
Implements prediction logic for Cox PH, RSF, and XGBoost models,
and combines them into an ensemble prediction.

Supports both:
1. Trained XGBoost models (if available)
2. Coefficient-based predictions (fallback)
"""

import numpy as np
from config import (
    FEATURE_OPTIONS, REFERENCE_CATEGORIES, NUMERIC_FEATURES,
    OUTCOMES, ENSEMBLE_WEIGHTS, RISK_THRESHOLDS, TIME_POINTS,
    LANDMARK_TIME
)
from model_coefficients import (
    COX_COEFFICIENTS, RSF_EFFECTS, XGBOOST_EFFECTS,
    XGBOOST_FEATURE_IMPORTANCE, MODEL_PERFORMANCE
)

# Try to import model loader for trained models
try:
    from model_loader import (
        predict_with_xgboost, check_models_available,
        preprocess_patient_for_xgboost, get_model_type,
        get_shap_importance_for_outcome, get_feature_contributions as get_shap_contributions
    )
    TRAINED_MODELS_AVAILABLE = True
except ImportError:
    TRAINED_MODELS_AVAILABLE = False
    
# Check which trained models are available
_models_status = None

def get_models_status():
    """Check and cache which trained models are available."""
    global _models_status
    if _models_status is None:
        if TRAINED_MODELS_AVAILABLE:
            _models_status = check_models_available()
        else:
            _models_status = {k: False for k in ['OS', 'NRM', 'Relapse', 'cGVHD']}
    return _models_status


def get_model_info():
    """Get information about which model type is used for each outcome."""
    if TRAINED_MODELS_AVAILABLE:
        return {
            'OS': {'model_type': get_model_type('OS'), 'description': 'Cox Proportional Hazards'},
            'NRM': {'model_type': get_model_type('NRM'), 'description': 'Cox Proportional Hazards'},
            'Relapse': {'model_type': get_model_type('Relapse'), 'description': 'Accelerated Failure Time'},
            'cGVHD': {'model_type': get_model_type('cGVHD'), 'description': 'Fine-Gray Subdistribution Hazard'}
        }
    return {
        'OS': {'model_type': 'coefficient', 'description': 'Coefficient-based'},
        'NRM': {'model_type': 'coefficient', 'description': 'Coefficient-based'},
        'Relapse': {'model_type': 'coefficient', 'description': 'Coefficient-based'},
        'cGVHD': {'model_type': 'coefficient', 'description': 'Coefficient-based'}
    }


def calculate_cox_prediction(patient_data, outcome):
    """
    Calculate Cox PH model prediction for a given patient.
    
    For OS: Returns survival probability at 36 months
    For NRM/Relapse/cGVHD: Returns cumulative incidence at 36 months
    
    Uses the formula: S(t) = S0(t)^exp(XB) for survival
                     CIF(t) = 1 - (1 - CIF0(t))^exp(XB) for competing risks (approx)
    """
    coef_dict = COX_COEFFICIENTS[outcome]
    coefficients = coef_dict['coefficients']
    
    # Calculate linear predictor (XB)
    linear_predictor = 0.0
    
    # Process categorical variables
    categorical_features = [
        'Disease Status', 'Cytogenetic Score', 'Gender', 'Karnofsky Score',
        'HCT-CI', 'Time Dx to HCT', 'Immunophenotype', 'Ph+/BCR-ABL1',
        'Donor Type', 'Donor/Recipient Sex Match', 'Graft Type',
        'Conditioning Regimen', 'GVHD Prophylaxis', 'In Vivo T-cell Depletion (Yes)'
    ]
    
    for feature in categorical_features:
        value = patient_data.get(feature)
        if value is None:
            continue
            
        # Skip if reference category
        ref_cat = REFERENCE_CATEGORIES.get(feature)
        if value == ref_cat:
            continue
            
        # Look for coefficient
        coef_key = f"{feature}_{value}"
        if coef_key in coefficients:
            linear_predictor += coefficients[coef_key]
    
    # Process numeric variables
    if 'Age at HCT' in coefficients:
        age = patient_data.get('Age at HCT', 45)
        age_ref = 45  # Reference age
        linear_predictor += coefficients['Age at HCT'] * (age - age_ref)
    
    if 'Donor Age' in coefficients:
        donor_age = patient_data.get('Donor Age', 35)
        donor_age_ref = 35  # Reference donor age
        linear_predictor += coefficients['Donor Age'] * (donor_age - donor_age_ref)
    
    if 'Year of HCT' in coefficients:
        year = patient_data.get('Year of HCT', 2020)
        year_ref = 2017  # Reference year
        linear_predictor += coefficients['Year of HCT'] * (year - year_ref)
    
    # Calculate risk score (hazard ratio relative to baseline)
    relative_hazard = np.exp(linear_predictor)
    
    if outcome == 'OS':
        # For survival: S(t) = S0(t)^HR
        baseline_surv = coef_dict['baseline_survival_36m']
        survival_prob = baseline_surv ** relative_hazard
        return max(0.01, min(0.99, survival_prob))
    else:
        # For competing risks: CIF approximation
        baseline_cif = coef_dict['baseline_cif_36m']
        # Use complementary log-log transformation approximation
        cif_prob = 1 - (1 - baseline_cif) ** relative_hazard
        return max(0.01, min(0.99, cif_prob))


def calculate_rsf_prediction(patient_data, outcome):
    """
    Calculate RSF model prediction for a given patient.
    
    Uses partial dependence effects to estimate risk.
    """
    effects = RSF_EFFECTS[outcome]
    
    # Start with baseline probability
    if outcome == 'OS':
        base_prob = 0.70  # Baseline survival
    elif outcome == 'NRM':
        base_prob = 0.15  # Baseline NRM CIF
    elif outcome == 'Relapse':
        base_prob = 0.30  # Baseline relapse CIF
    else:  # cGVHD
        base_prob = 0.35  # Baseline cGVHD CIF
    
    # Calculate additive effects
    total_effect = 0.0
    
    # Process categorical effects
    for feature in ['Disease Status', 'Cytogenetic Score', 'Karnofsky Score', 
                    'HCT-CI', 'Donor Type']:
        if feature in effects:
            value = patient_data.get(feature)
            if value in effects[feature]:
                total_effect += effects[feature][value]
    
    # Process continuous effects
    if 'Age at HCT_effect' in effects:
        age = patient_data.get('Age at HCT', 45)
        total_effect += effects['Age at HCT_effect'] * (age - 45)
    
    if 'Donor Age_effect' in effects:
        donor_age = patient_data.get('Donor Age', 35)
        total_effect += effects['Donor Age_effect'] * (donor_age - 35)
    
    if 'Year of HCT_effect' in effects:
        year = patient_data.get('Year of HCT', 2020)
        total_effect += effects['Year of HCT_effect'] * (year - 2017)
    
    if outcome == 'OS':
        # For survival, lower effect = better
        prediction = base_prob - total_effect
    else:
        # For CIF, higher effect = higher risk
        prediction = base_prob + total_effect
    
    return max(0.01, min(0.99, prediction))


def calculate_xgboost_prediction(patient_data, outcome, use_trained_model=True):
    """
    Calculate XGBoost model prediction for a given patient.
    
    Uses the optimal model type for each outcome:
    - OS: Cox PH
    - NRM: Cox PH
    - Relapse: AFT
    - cGVHD: Fine-Gray
    
    If trained models are available and use_trained_model=True,
    uses the actual trained model. Otherwise falls back to
    coefficient-based approximation.
    
    Parameters:
    -----------
    patient_data : dict
        Patient characteristics
    outcome : str
        One of 'OS', 'NRM', 'Relapse', 'cGVHD'
    use_trained_model : bool
        Whether to use trained model if available
    
    Returns:
    --------
    float : Predicted probability (0-1)
    """
    # Try to use trained model first
    if use_trained_model and TRAINED_MODELS_AVAILABLE:
        models_status = get_models_status()
        if models_status.get(outcome, False):
            result = predict_with_xgboost(patient_data, outcome)
            if result is not None:
                # Result already contains probability from model_loader
                prediction = result.get('probability', 0.5)
                return max(0.01, min(0.99, prediction))
    
    # Fallback to coefficient-based prediction
    return _calculate_xgboost_coefficient_prediction(patient_data, outcome)


def _calculate_xgboost_coefficient_prediction(patient_data, outcome):
    """
    Coefficient-based XGBoost prediction (fallback).
    Uses multiplicative effect modifiers.
    """
    effects = XGBOOST_EFFECTS[outcome]
    
    # Start with baseline
    if outcome == 'OS':
        base_prob = 0.70
    elif outcome == 'NRM':
        base_prob = 0.15
    elif outcome == 'Relapse':
        base_prob = 0.30
    else:  # cGVHD
        base_prob = 0.35
    
    # Calculate multiplicative effects
    multiplier = 1.0
    
    for feature, effect_dict in effects.items():
        value = patient_data.get(feature)
        if value in effect_dict:
            multiplier *= effect_dict[value]
    
    # Apply age effects
    age = patient_data.get('Age at HCT', 45)
    if outcome == 'OS':
        # Older age = lower survival
        age_factor = 0.995 ** (age - 45)
        multiplier *= age_factor
    elif outcome in ['NRM', 'cGVHD']:
        # Older age = higher NRM and cGVHD
        age_factor = 1.01 ** (age - 45)
        multiplier *= age_factor
    
    if outcome == 'OS':
        prediction = base_prob * multiplier
    else:
        prediction = base_prob * multiplier
    
    return max(0.01, min(0.99, prediction))


def calculate_ensemble_prediction(patient_data, outcome, weights=None):
    """
    Calculate ensemble prediction combining Cox, RSF, and XGBoost.
    
    Returns individual predictions and weighted ensemble.
    """
    if weights is None:
        weights = ENSEMBLE_WEIGHTS[outcome]
    
    # Get individual predictions
    cox_pred = calculate_cox_prediction(patient_data, outcome)
    rsf_pred = calculate_rsf_prediction(patient_data, outcome)
    xgboost_pred = calculate_xgboost_prediction(patient_data, outcome)
    
    # Calculate weighted ensemble
    ensemble_pred = (
        weights['Cox'] * cox_pred +
        weights['RSF'] * rsf_pred +
        weights['XGBoost'] * xgboost_pred
    )
    
    return {
        'cox': cox_pred,
        'rsf': rsf_pred,
        'xgboost': xgboost_pred,
        'ensemble': ensemble_pred,
        'weights': weights
    }


def calculate_survival_curves(patient_data, outcome):
    """
    Calculate survival/CIF curves at multiple time points.
    """
    # For simplicity, use exponential decay/growth model
    pred_36m = calculate_ensemble_prediction(patient_data, outcome)['ensemble']
    
    curves = {'time': TIME_POINTS}
    
    if outcome == 'OS':
        # Survival curve: S(t) = S(36)^(t/36)
        curves['survival'] = [
            pred_36m ** (t / 36) if t > 0 else 1.0 
            for t in TIME_POINTS
        ]
    else:
        # CIF curve: CIF(t) = CIF(36) * (t/36)
        curves['cif'] = [
            pred_36m * (t / 36) if t > 0 else 0.0 
            for t in TIME_POINTS
        ]
    
    return curves


def get_risk_category(prediction, outcome):
    """
    Categorize risk based on prediction and outcome-specific thresholds.
    """
    thresholds = RISK_THRESHOLDS[outcome]
    
    if outcome == 'OS':
        # For survival, higher is better
        if prediction >= thresholds['low']:
            return 'Low Risk', 'green'
        elif prediction >= thresholds['high']:
            return 'Intermediate Risk', 'orange'
        else:
            return 'High Risk', 'red'
    else:
        # For CIF, lower is better
        if prediction <= thresholds['low']:
            return 'Low Risk', 'green'
        elif prediction <= thresholds['high']:
            return 'Intermediate Risk', 'orange'
        else:
            return 'High Risk', 'red'


def get_prediction_summary(patient_data):
    """
    Get complete prediction summary for all outcomes.
    """
    summary = {}
    
    for outcome_key in ['OS', 'NRM', 'Relapse', 'cGVHD']:
        predictions = calculate_ensemble_prediction(patient_data, outcome_key)
        curves = calculate_survival_curves(patient_data, outcome_key)
        risk_cat, risk_color = get_risk_category(predictions['ensemble'], outcome_key)
        
        summary[outcome_key] = {
            'predictions': predictions,
            'curves': curves,
            'risk_category': risk_cat,
            'risk_color': risk_color,
            'outcome_info': OUTCOMES[outcome_key]
        }
    
    return summary


def get_feature_contributions(patient_data, outcome):
    """
    Calculate contribution of each feature to the prediction.
    Useful for model interpretation.
    """
    # Get baseline prediction
    ref_patient = REFERENCE_CATEGORIES.copy()
    ref_patient['Year of HCT'] = 2020
    ref_patient['Age at HCT'] = 45
    ref_patient['Donor Age'] = 35
    
    baseline_pred = calculate_ensemble_prediction(ref_patient, outcome)['ensemble']
    full_pred = calculate_ensemble_prediction(patient_data, outcome)['ensemble']
    
    contributions = {}
    importance = XGBOOST_FEATURE_IMPORTANCE[outcome]
    
    for feature in importance.keys():
        # Calculate marginal contribution
        test_patient = ref_patient.copy()
        if feature in patient_data:
            test_patient[feature] = patient_data[feature]
        
        test_pred = calculate_ensemble_prediction(test_patient, outcome)['ensemble']
        contribution = test_pred - baseline_pred
        
        contributions[feature] = {
            'value': patient_data.get(feature, 'N/A'),
            'contribution': contribution,
            'importance': importance.get(feature, 0),
            'direction': 'Higher Risk' if contribution > 0.01 else ('Lower Risk' if contribution < -0.01 else 'Neutral')
        }
    
    # Sort by absolute contribution
    sorted_contributions = sorted(
        contributions.items(),
        key=lambda x: abs(x[1]['contribution']),
        reverse=True
    )
    
    return dict(sorted_contributions)


def validate_patient_data(patient_data):
    """
    Validate patient data and fill in defaults for missing values.
    """
    validated = {}
    errors = []
    
    # Validate categorical features
    for feature, options in FEATURE_OPTIONS.items():
        value = patient_data.get(feature)
        if value is None or value == '':
            validated[feature] = REFERENCE_CATEGORIES.get(feature)
        elif value in options:
            validated[feature] = value
        else:
            errors.append(f"Invalid value '{value}' for {feature}")
            validated[feature] = REFERENCE_CATEGORIES.get(feature)
    
    # Validate numeric features
    for feature, limits in NUMERIC_FEATURES.items():
        value = patient_data.get(feature)
        if value is None:
            validated[feature] = limits['default']
        elif isinstance(value, (int, float)) and limits['min'] <= value <= limits['max']:
            validated[feature] = value
        else:
            errors.append(f"Invalid value '{value}' for {feature}")
            validated[feature] = limits['default']
    
    return validated, errors


# ==============================================================================
# WHAT-IF COMPARISON FUNCTIONS
# ==============================================================================

def compare_scenarios(scenario_a, scenario_b):
    """
    Compare predictions between two patient scenarios.
    
    Parameters:
    -----------
    scenario_a : dict
        First patient scenario (e.g., current patient)
    scenario_b : dict
        Second patient scenario (e.g., what-if scenario)
    
    Returns:
    --------
    dict : Comparison results for all outcomes
    """
    comparison = {}
    
    for outcome_key in ['OS', 'NRM', 'Relapse', 'cGVHD']:
        pred_a = calculate_ensemble_prediction(scenario_a, outcome_key)
        pred_b = calculate_ensemble_prediction(scenario_b, outcome_key)
        
        # Calculate differences
        ensemble_diff = pred_b['ensemble'] - pred_a['ensemble']
        
        comparison[outcome_key] = {
            'scenario_a': pred_a,
            'scenario_b': pred_b,
            'ensemble_diff': ensemble_diff,
            'percent_change': (ensemble_diff / pred_a['ensemble']) * 100 if pred_a['ensemble'] != 0 else 0,
            'outcome_info': OUTCOMES[outcome_key],
            'improved': (outcome_key == 'OS' and ensemble_diff > 0) or 
                       (outcome_key != 'OS' and ensemble_diff < 0)
        }
    
    return comparison


def get_scenario_differences(scenario_a, scenario_b):
    """
    Get the differences between two scenarios.
    
    Returns:
    --------
    dict : Features that differ between scenarios
    """
    differences = {}
    
    all_features = set(scenario_a.keys()) | set(scenario_b.keys())
    
    for feature in all_features:
        val_a = scenario_a.get(feature)
        val_b = scenario_b.get(feature)
        
        if val_a != val_b:
            differences[feature] = {
                'scenario_a': val_a,
                'scenario_b': val_b
            }
    
    return differences


def calculate_feature_impact(patient_data, feature, alternative_values, outcome):
    """
    Calculate the impact of changing a single feature on prediction.
    
    Parameters:
    -----------
    patient_data : dict
        Base patient characteristics
    feature : str
        Feature to vary
    alternative_values : list
        List of alternative values to test
    outcome : str
        Outcome to predict
    
    Returns:
    --------
    dict : Predictions for each value
    """
    base_pred = calculate_ensemble_prediction(patient_data, outcome)['ensemble']
    
    impacts = {
        'base_value': patient_data.get(feature),
        'base_prediction': base_pred,
        'alternatives': []
    }
    
    for value in alternative_values:
        modified = patient_data.copy()
        modified[feature] = value
        
        pred = calculate_ensemble_prediction(modified, outcome)['ensemble']
        diff = pred - base_pred
        
        impacts['alternatives'].append({
            'value': value,
            'prediction': pred,
            'difference': diff,
            'percent_change': (diff / base_pred) * 100 if base_pred != 0 else 0
        })
    
    return impacts


def get_modifiable_scenarios(patient_data):
    """
    Generate clinically meaningful alternative scenarios.
    
    Returns scenarios with modifications that could be considered
    for clinical decision-making.
    """
    scenarios = []
    
    # Alternative donor types
    current_donor = patient_data.get('Donor Type')
    donor_options = ['HLA-identical sibling', '8/8 MUD', 'Haploidentical']
    for donor in donor_options:
        if donor != current_donor:
            scenario = patient_data.copy()
            scenario['Donor Type'] = donor
            scenarios.append({
                'name': f'Alternative Donor: {donor}',
                'description': f'Using {donor} instead of {current_donor}',
                'data': scenario,
                'category': 'Donor'
            })
    
    # Alternative conditioning
    current_conditioning = patient_data.get('Conditioning Regimen')
    conditioning_options = ['MAC TBI', 'MAC Chemo', 'RIC/NMA']
    for cond in conditioning_options:
        if cond != current_conditioning:
            scenario = patient_data.copy()
            scenario['Conditioning Regimen'] = cond
            scenarios.append({
                'name': f'Conditioning: {cond}',
                'description': f'Using {cond} instead of {current_conditioning}',
                'data': scenario,
                'category': 'Conditioning'
            })
    
    # Alternative GVHD prophylaxis
    current_gvhd = patient_data.get('GVHD Prophylaxis')
    gvhd_options = ['CNI Based', 'PTCy Based']
    for gvhd in gvhd_options:
        if gvhd != current_gvhd:
            scenario = patient_data.copy()
            scenario['GVHD Prophylaxis'] = gvhd
            scenarios.append({
                'name': f'GVHD Prophylaxis: {gvhd}',
                'description': f'Using {gvhd} instead of {current_gvhd}',
                'data': scenario,
                'category': 'GVHD Prevention'
            })
    
    # T-cell depletion
    current_tcd = patient_data.get('In Vivo T-cell Depletion (Yes)')
    tcd_alt = 'Yes' if current_tcd == 'No' else 'No'
    scenario = patient_data.copy()
    scenario['In Vivo T-cell Depletion (Yes)'] = tcd_alt
    scenarios.append({
        'name': f'T-cell Depletion: {tcd_alt}',
        'description': f'With ATG/Alemtuzumab' if tcd_alt == 'Yes' else 'Without ATG/Alemtuzumab',
        'data': scenario,
        'category': 'T-cell Depletion'
    })
    
    # Graft source
    current_graft = patient_data.get('Graft Type')
    graft_options = ['Peripheral Blood', 'Bone Marrow']
    for graft in graft_options:
        if graft != current_graft:
            scenario = patient_data.copy()
            scenario['Graft Type'] = graft
            scenarios.append({
                'name': f'Graft Source: {graft}',
                'description': f'Using {graft} instead of {current_graft}',
                'data': scenario,
                'category': 'Graft'
            })
    
    return scenarios


def get_using_trained_models():
    """Check if trained models are being used."""
    if TRAINED_MODELS_AVAILABLE:
        status = get_models_status()
        return any(status.values())
    return False


# ==============================================================================
# ADJUSTABLE COVARIATE EFFECTS TABLE
# ==============================================================================

# Define which covariates are adjustable (treatment decisions)
ADJUSTABLE_COVARIATES = {
    'Conditioning Regimen': {
        'reference': 'MAC TBI',
        'label': 'Conditioning Regimen*'
    },
    'GVHD Prophylaxis': {
        'reference': 'CNI Based',
        'label': 'GVHD Prophylaxis*'
    },
    'Donor Type': {
        'reference': 'HLA-identical sibling',
        'label': 'Donor Type*'
    },
    'Graft Type': {
        'reference': 'Peripheral Blood',
        'label': 'Graft Source*'
    },
    'In Vivo T-cell Depletion (Yes)': {
        'reference': 'No',
        'label': 'T-cell Depletion (ATG)*'
    }
}


def calculate_covariate_effects_table(patient_data):
    """
    Calculate the effect of each adjustable covariate selection 
    compared to its reference value.
    
    Returns a list of dictionaries with covariate effects for display.
    """
    effects_table = []
    
    for covariate, info in ADJUSTABLE_COVARIATES.items():
        selected_value = patient_data.get(covariate)
        reference_value = info['reference']
        
        # Skip if selected value is the reference
        is_reference = (selected_value == reference_value)
        
        # Create reference patient (all same except this covariate at reference)
        ref_patient = patient_data.copy()
        ref_patient[covariate] = reference_value
        
        # Calculate effects for each outcome
        effects = {}
        for outcome in ['OS', 'NRM', 'Relapse', 'cGVHD']:
            pred_selected = calculate_ensemble_prediction(patient_data, outcome)['ensemble']
            pred_reference = calculate_ensemble_prediction(ref_patient, outcome)['ensemble']
            
            # Calculate absolute difference
            diff = pred_selected - pred_reference
            
            # Determine direction and interpretation
            if outcome == 'OS':
                # For OS, higher is better
                if diff > 0.01:
                    direction = '↑'
                    interpretation = 'benefit'
                elif diff < -0.01:
                    direction = '↓'
                    interpretation = 'risk'
                else:
                    direction = '–'
                    interpretation = 'neutral'
            else:
                # For NRM, Relapse, cGVHD - lower is better
                if diff < -0.01:
                    direction = '↓'
                    interpretation = 'benefit'
                elif diff > 0.01:
                    direction = '↑'
                    interpretation = 'risk'
                else:
                    direction = '–'
                    interpretation = 'neutral'
            
            effects[outcome] = {
                'diff': diff,
                'diff_pct': diff * 100,
                'direction': direction,
                'interpretation': interpretation
            }
        
        effects_table.append({
            'covariate': covariate,
            'label': info['label'],
            'selected': selected_value,
            'reference': reference_value,
            'is_reference': is_reference,
            'effects': effects
        })
    
    return effects_table


def format_effect_cell(effect_data, outcome):
    """Format a single effect cell for display."""
    diff_pct = effect_data['diff_pct']
    direction = effect_data['direction']
    interpretation = effect_data['interpretation']
    
    if abs(diff_pct) < 1:
        return "–"
    
    # Format with direction arrow and percentage
    sign = '+' if diff_pct > 0 else ''
    return f"{direction} {sign}{diff_pct:.1f}%"
