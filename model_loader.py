"""
Model Loader for Optimal XGBoost Models
========================================
Handles loading and prediction with different model types:
- Cox PH (OS, NRM)
- AFT (Relapse)
- Fine-Gray (cGVHD)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

# Path to trained models
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'trained_models')

# Cache for loaded models and preprocessors
_models_cache = {}
_preprocessing_info = None
_label_encoders = None
_model_config = None
_shap_importance = None


def get_preprocessing_info():
    """Load preprocessing information."""
    global _preprocessing_info
    if _preprocessing_info is None:
        path = os.path.join(MODELS_DIR, 'preprocessing_info.json')
        if os.path.exists(path):
            with open(path, 'r') as f:
                _preprocessing_info = json.load(f)
    return _preprocessing_info


def get_label_encoders():
    """Load label encoders."""
    global _label_encoders
    if _label_encoders is None:
        path = os.path.join(MODELS_DIR, 'label_encoders.pkl')
        if os.path.exists(path):
            _label_encoders = joblib.load(path)
    return _label_encoders


def get_model_config():
    """Load model configuration (model types for each outcome)."""
    global _model_config
    if _model_config is None:
        path = os.path.join(MODELS_DIR, 'model_config.json')
        if os.path.exists(path):
            with open(path, 'r') as f:
                _model_config = json.load(f)
        else:
            # Default config if file doesn't exist
            _model_config = {
                'OS': {'model_type': 'cox', 'model_file': 'xgboost_OS.json'},
                'NRM': {'model_type': 'cox', 'model_file': 'xgboost_NRM.json'},
                'Relapse': {'model_type': 'aft', 'model_file': 'xgboost_Relapse.json'},
                'cGVHD': {'model_type': 'fine_gray', 'model_file': 'xgboost_cGVHD.json'}
            }
    return _model_config


def get_shap_importance():
    """Load SHAP importance data."""
    global _shap_importance
    if _shap_importance is None:
        path = os.path.join(MODELS_DIR, 'shap_importance.csv')
        if os.path.exists(path):
            _shap_importance = pd.read_csv(path)
    return _shap_importance


def get_shap_importance_for_outcome(outcome):
    """Get SHAP importance for a specific outcome."""
    shap_df = get_shap_importance()
    if shap_df is None:
        return None
    
    outcome_map = {
        'Overall Survival': 'OS',
        'Non-Relapse Mortality': 'NRM',
        'Relapse': 'Relapse',
        'Chronic GVHD': 'cGVHD'
    }
    outcome_key = outcome_map.get(outcome, outcome)
    
    filtered = shap_df[shap_df['outcome'] == outcome_key].copy()
    if len(filtered) == 0:
        return None
    
    return filtered.sort_values('mean_abs_shap', ascending=False)


def load_xgboost_model(outcome):
    """Load an XGBoost model for a specific outcome."""
    global _models_cache
    
    if outcome in _models_cache:
        return _models_cache[outcome]
    
    config = get_model_config()
    
    # Map outcome names
    outcome_map = {
        'Overall Survival': 'OS',
        'Non-Relapse Mortality': 'NRM',
        'Relapse': 'Relapse',
        'Chronic GVHD': 'cGVHD'
    }
    outcome_key = outcome_map.get(outcome, outcome)
    
    if outcome_key not in config:
        return None
    
    model_file = config[outcome_key]['model_file']
    model_path = os.path.join(MODELS_DIR, model_file)
    
    if not os.path.exists(model_path):
        return None
    
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    
    _models_cache[outcome] = model
    return model


def get_model_type(outcome):
    """Get the model type for a specific outcome."""
    config = get_model_config()
    
    outcome_map = {
        'Overall Survival': 'OS',
        'Non-Relapse Mortality': 'NRM',
        'Relapse': 'Relapse',
        'Chronic GVHD': 'cGVHD'
    }
    outcome_key = outcome_map.get(outcome, outcome)
    
    if outcome_key in config:
        return config[outcome_key].get('model_type', 'cox')
    return 'cox'


def check_models_available():
    """Check which models are available."""
    available = {}
    config = get_model_config()
    
    for outcome_key, info in config.items():
        model_file = info.get('model_file', f'xgboost_{outcome_key}.json')
        model_path = os.path.join(MODELS_DIR, model_file)
        available[outcome_key] = os.path.exists(model_path)
    
    return available


def preprocess_patient_for_xgboost(patient_data):
    """
    Preprocess patient data for XGBoost prediction.
    
    Args:
        patient_data: dict with patient characteristics
        
    Returns:
        DataFrame ready for model.predict()
    """
    preprocessing_info = get_preprocessing_info()
    label_encoders = get_label_encoders()
    
    if preprocessing_info is None:
        raise ValueError("Preprocessing info not found")
    
    feature_columns = preprocessing_info.get('feature_columns', [])
    encoding_maps = preprocessing_info.get('encoding_maps', {})
    numeric_medians = preprocessing_info.get('numeric_medians', {})
    
    # Map from UI field names to model feature names
    field_mapping = {
        'Patient Sex': 'Gender',
        'Donor/Recipient Sex Match': 'Donor/Recipient Sex Match',
    }
    
    # Create feature dict
    features = {}
    
    for col in feature_columns:
        # Get value from patient data (try direct name or mapped name)
        ui_name = next((k for k, v in field_mapping.items() if v == col), col)
        value = patient_data.get(col) or patient_data.get(ui_name)
        
        if value is None:
            # Use median/mode for missing
            if col in numeric_medians:
                value = numeric_medians[col]
            elif col in encoding_maps:
                # Use first class as default
                value = list(encoding_maps[col].keys())[0]
        
        # Process the value
        if col in encoding_maps:
            # Categorical - encode
            if str(value) in encoding_maps[col]:
                features[col] = encoding_maps[col][str(value)]
            else:
                # Unknown category - use 0
                features[col] = 0
        elif col == 'HCT-CI':
            # Special handling
            if value == '3+':
                value = 3
            try:
                features[col] = float(value)
            except:
                features[col] = numeric_medians.get(col, 0)
        elif col == 'Karnofsky Score':
            # Special handling
            if value == '>=90':
                features[col] = 90
            elif value == '<90':
                features[col] = 80
            else:
                try:
                    features[col] = float(value)
                except:
                    features[col] = 90
        else:
            # Numeric
            try:
                features[col] = float(value)
            except:
                features[col] = numeric_medians.get(col, 0)
    
    # Create DataFrame in correct column order
    df = pd.DataFrame([features])[feature_columns]
    
    return df


def predict_with_xgboost(patient_data, outcome):
    """
    Make prediction for a patient using the optimal XGBoost model.
    
    Args:
        patient_data: dict with patient characteristics
        outcome: str, one of 'Overall Survival', 'Non-Relapse Mortality', 'Relapse', 'Chronic GVHD'
        
    Returns:
        dict with 'risk_score' and 'probability'
    """
    model = load_xgboost_model(outcome)
    if model is None:
        return None
    
    model_type = get_model_type(outcome)
    
    # Preprocess patient data
    X = preprocess_patient_for_xgboost(patient_data)
    
    # Get raw prediction
    raw_pred = model.predict(X)[0]
    
    # Interpret based on model type
    if model_type == 'cox':
        # Cox: output is log-hazard ratio (risk score)
        # Higher = worse prognosis
        risk_score = float(raw_pred)
        
        # Convert to probability using baseline survival
        # S(t) = S0(t)^exp(risk_score)
        # Using approximate baseline survival at 36 months
        baseline_survival = 0.5  # Approximate median survival
        
        if outcome in ['Overall Survival', 'OS']:
            # For OS, probability of death
            survival_prob = baseline_survival ** np.exp(risk_score)
            probability = 1 - survival_prob
        else:
            # For NRM, probability of event
            baseline_cif = 0.22  # Approximate NRM CIF
            probability = 1 - (1 - baseline_cif) ** np.exp(risk_score)
            probability = np.clip(probability, 0, 1)
        
    elif model_type == 'aft':
        # AFT: output is log(predicted time)
        # Shorter time = worse prognosis
        # We already trained with negated SHAP, so negate for risk
        log_time = float(raw_pred)
        predicted_time = np.exp(log_time)
        
        # Convert to probability of event by 36 months
        # Using exponential approximation
        risk_score = -log_time  # Higher risk = shorter time
        
        # Probability based on predicted time vs landmark
        if predicted_time >= 36:
            probability = 0.2  # Low risk
        else:
            # Approximate: probability increases as predicted time decreases
            probability = 0.3 + 0.5 * (1 - predicted_time / 36)
        
        probability = np.clip(probability, 0.1, 0.9)
        
    elif model_type == 'fine_gray':
        # Fine-Gray: output is predicted CIF (cumulative incidence)
        # This is directly the probability
        probability = float(np.clip(raw_pred, 0, 1))
        risk_score = probability  # For Fine-Gray, risk score IS the probability
        
    else:
        # Unknown model type, use raw prediction
        risk_score = float(raw_pred)
        probability = float(np.clip(raw_pred, 0, 1))
    
    return {
        'risk_score': risk_score,
        'probability': probability,
        'model_type': model_type
    }


def get_feature_contributions(patient_data, outcome):
    """
    Get SHAP-based feature contributions for a prediction.
    
    Note: This returns pre-computed average SHAP importance,
    not individual prediction explanations (which would require SHAP runtime).
    """
    shap_df = get_shap_importance_for_outcome(outcome)
    if shap_df is None:
        return None
    
    # Return top features
    return shap_df.head(10)[['feature', 'mean_abs_shap']].to_dict('records')


# Check model availability on import
def print_model_status():
    """Print status of available models."""
    available = check_models_available()
    config = get_model_config()
    
    print("Model Status:")
    print("-" * 50)
    for outcome, is_available in available.items():
        model_type = config.get(outcome, {}).get('model_type', 'unknown')
        status = "✓ Available" if is_available else "✗ Not found"
        print(f"  {outcome}: {model_type} - {status}")
    print("-" * 50)


if __name__ == "__main__":
    print_model_status()
