"""
Model Loader Module
====================
Loads trained XGBoost models and preprocessing info for predictions.
Falls back to coefficient-based predictions if models are not available.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

# Try importing xgboost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_SUBDIR = os.path.join(MODEL_DIR, 'trained_models')

# Cache for loaded models
_model_cache = {}
_preprocessing_cache = {}


def get_preprocessing_info():
    """Load preprocessing information (encoding maps, medians)."""
    if 'preprocessing' in _preprocessing_cache:
        return _preprocessing_cache['preprocessing']
    
    info_path = os.path.join(MODELS_SUBDIR, 'preprocessing_info.json')
    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            info = json.load(f)
        _preprocessing_cache['preprocessing'] = info
        return info
    return None


def get_label_encoders():
    """Load label encoders for categorical features."""
    if 'encoders' in _preprocessing_cache:
        return _preprocessing_cache['encoders']
    
    encoders_path = os.path.join(MODELS_SUBDIR, 'label_encoders.pkl')
    if os.path.exists(encoders_path):
        encoders = joblib.load(encoders_path)
        _preprocessing_cache['encoders'] = encoders
        return encoders
    return None


def load_xgboost_model(outcome):
    """
    Load XGBoost model for a specific outcome.
    
    Parameters:
    -----------
    outcome : str
        One of 'OS', 'NRM', 'Relapse', 'cGVHD'
    
    Returns:
    --------
    model : XGBRegressor or None
    """
    cache_key = f'xgboost_{outcome}'
    if cache_key in _model_cache:
        return _model_cache[cache_key]
    
    if not XGBOOST_AVAILABLE:
        return None
    
    model_path = os.path.join(MODELS_SUBDIR, f'xgboost_{outcome}.json')
    
    if os.path.exists(model_path):
        try:
            model = xgb.XGBRegressor()
            model.load_model(model_path)
            _model_cache[cache_key] = model
            return model
        except Exception as e:
            print(f"Warning: Could not load {outcome} model: {e}")
            return None
    return None


def preprocess_patient_for_xgboost(patient_data):
    """
    Preprocess patient data for XGBoost prediction.
    
    Converts categorical values to encoded integers and handles
    special feature transformations to match training data format.
    """
    from config import FEATURE_COLUMNS
    
    preprocessing = get_preprocessing_info()
    encoders = get_label_encoders()
    
    # If no preprocessing info, fall back to simple encoding
    if preprocessing is None or encoders is None:
        return preprocess_patient_simple(patient_data)
    
    encoding_maps = preprocessing.get('encoding_maps', {})
    numeric_medians = preprocessing.get('numeric_medians', {})
    
    processed = {}
    
    for feature in FEATURE_COLUMNS:
        value = patient_data.get(feature)
        
        # Handle numeric features
        if feature in ['Year of HCT', 'Age at HCT', 'Donor Age']:
            if value is None:
                processed[feature] = numeric_medians.get(feature, 0)
            else:
                processed[feature] = float(value)
        
        # Handle HCT-CI (convert to numeric)
        elif feature == 'HCT-CI':
            if value is None:
                processed[feature] = numeric_medians.get(feature, 0)
            elif value == '3+':
                processed[feature] = 3.0
            else:
                try:
                    processed[feature] = float(value.replace('+', ''))
                except:
                    processed[feature] = numeric_medians.get(feature, 0)
        
        # Handle Karnofsky Score (convert to numeric)
        elif feature == 'Karnofsky Score':
            if value == '>=90':
                processed[feature] = 90.0
            elif value == '<90':
                processed[feature] = 80.0
            else:
                processed[feature] = numeric_medians.get(feature, 90)
        
        # Handle categorical features
        elif feature in encoding_maps:
            if value is None:
                value = 'Missing'
            if value in encoding_maps[feature]:
                processed[feature] = encoding_maps[feature][value]
            else:
                # Unknown category - use 0 or 'Missing' encoding
                processed[feature] = encoding_maps[feature].get('Missing', 0)
        
        else:
            # Unknown feature type
            processed[feature] = 0
    
    return processed


def preprocess_patient_simple(patient_data):
    """
    Simple preprocessing when full preprocessing info is not available.
    Uses basic encoding based on feature options.
    """
    from config import FEATURE_COLUMNS, FEATURE_OPTIONS
    
    processed = {}
    
    for feature in FEATURE_COLUMNS:
        value = patient_data.get(feature)
        
        # Numeric features
        if feature in ['Year of HCT', 'Age at HCT', 'Donor Age']:
            processed[feature] = float(value) if value is not None else 0.0
        
        elif feature == 'HCT-CI':
            if value == '0':
                processed[feature] = 0.0
            elif value == '1-2':
                processed[feature] = 1.5
            elif value == '3+':
                processed[feature] = 3.0
            else:
                processed[feature] = 0.0
        
        elif feature == 'Karnofsky Score':
            processed[feature] = 90.0 if value == '>=90' else 80.0
        
        # Categorical features - use index in options list
        elif feature in FEATURE_OPTIONS:
            options = FEATURE_OPTIONS[feature]
            if value in options:
                processed[feature] = options.index(value)
            else:
                processed[feature] = 0
        
        else:
            processed[feature] = 0
    
    return processed


def predict_with_xgboost(patient_data, outcome):
    """
    Make prediction using trained XGBoost model.
    
    Returns risk score (higher = higher risk of event).
    For OS, this is inverted (higher score = higher mortality risk).
    """
    model = load_xgboost_model(outcome)
    
    if model is None:
        return None
    
    # Preprocess patient data
    processed = preprocess_patient_for_xgboost(patient_data)
    
    # Create feature array in correct order
    from config import FEATURE_COLUMNS
    features = np.array([[processed[f] for f in FEATURE_COLUMNS]])
    
    # Predict
    try:
        risk_score = model.predict(features)[0]
        return risk_score
    except Exception as e:
        print(f"Warning: Prediction failed for {outcome}: {e}")
        return None


def check_models_available():
    """Check which models are available."""
    available = {}
    
    for outcome in ['OS', 'NRM', 'Relapse', 'cGVHD']:
        model = load_xgboost_model(outcome)
        available[outcome] = model is not None
    
    return available


def get_model_performance():
    """Load model performance metrics if available."""
    perf_path = os.path.join(MODELS_SUBDIR, 'model_performance.csv')
    
    if os.path.exists(perf_path):
        return pd.read_csv(perf_path)
    return None


if __name__ == "__main__":
    print("Checking available models...")
    available = check_models_available()
    
    print("\nXGBoost Models:")
    for outcome, is_available in available.items():
        status = "Available" if is_available else "Not found"
        print(f"  {outcome}: {status}")
    
    perf = get_model_performance()
    if perf is not None:
        print("\nModel Performance:")
        print(perf.to_string(index=False))
