"""
Train and Save XGBoost Models for Web Application
==================================================
This script trains XGBoost models for all outcomes and saves them
along with the label encoders for use in the web application.

Run this script once to train models, then the web app will load them.
"""

import pandas as pd
import numpy as np
import os
import joblib
import json
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sksurv.metrics import concordance_index_censored
import warnings
warnings.filterwarnings('ignore')

# Configuration
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
LANDMARK_TIME = 36

# Paths
DATA_PATH = '/Users/j/Desktop/XGBoost Jan 20 2026/ALL Expanded .csv'
OUTPUT_DIR = '/Users/j/Desktop/HCT_Ensemble_WebApp/trained_models'
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURE_COLUMNS = [
    'Disease Status', 'Cytogenetic Score', 'Year of HCT',
    'Age at HCT', 'Gender', 'Race/Ethnicity', 'Karnofsky Score', 'HCT-CI',
    'Time Dx to HCT', 'Immunophenotype', 'Ph+/BCR-ABL1', 'Donor Type',
    'Donor Age', 'Donor/Recipient Sex Match', 'Donor/Recipient CMV',
    'Graft Type', 'Conditioning Regimen', 'GVHD Prophylaxis',
    'In Vivo T-cell Depletion (Yes)'
]


def load_and_preprocess_data():
    """Load and preprocess the data."""
    print("=" * 80)
    print("LOADING AND PREPROCESSING DATA")
    print("=" * 80)
    
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} patients")
    
    # Create outcome variables
    df['OS_status'] = (df['Overall Survival'] == 'Event').astype(int)
    df['Relapse_status'] = (df['Relapse'] == 'Event').astype(int)
    df['cGVHD_status'] = (df['Chronic GVHD'] == 'Event').astype(int)
    
    df['OS_time'] = df['Time to Follow-up (months)'].clip(upper=LANDMARK_TIME)
    df['Relapse_time'] = df['Time to Relapse (months)'].fillna(df['Time to Follow-up (months)']).clip(upper=LANDMARK_TIME)
    df['cGVHD_time'] = df['Time to cGVHD (months)'].fillna(df['Time to Follow-up (months)']).clip(upper=LANDMARK_TIME)
    
    # OS event
    df['OS_event'] = ((df['Overall Survival'] == 'Event') & 
                      (df['Time to Follow-up (months)'] <= LANDMARK_TIME)).astype(int)
    
    # NRM
    df['relapse_related_death'] = (
        (df['OS_status'] == 1) & 
        (df['Relapse_status'] == 1) & 
        (df['Time to Relapse (months)'].fillna(999) <= df['Time to Follow-up (months)'])
    ).astype(int)
    
    df['NRM_event'] = (
        (df['OS_status'] == 1) & 
        (df['Time to Follow-up (months)'] <= LANDMARK_TIME) & 
        (df['relapse_related_death'] == 0)
    ).astype(int)
    
    df['NRM_time'] = df['OS_time'].copy()
    relapse_first = (df['Relapse_status'] == 1) & (df['Relapse_time'] < df['OS_time'])
    df.loc[relapse_first, 'NRM_time'] = df.loc[relapse_first, 'Relapse_time']
    df.loc[relapse_first, 'NRM_event'] = 0
    
    # Relapse
    df['Relapse_event'] = (
        (df['Relapse_status'] == 1) & 
        (df['Relapse_time'] <= LANDMARK_TIME)
    ).astype(int)
    
    # cGVHD
    df['cGVHD_event'] = (
        (df['cGVHD_status'] == 1) & 
        (df['cGVHD_time'] <= LANDMARK_TIME)
    ).astype(int)
    
    return df


def preprocess_features(df):
    """Preprocess features and create label encoders."""
    print("\nPreprocessing features...")
    
    X = df[FEATURE_COLUMNS].copy()
    
    numeric_cols = ['Year of HCT', 'Age at HCT', 'Donor Age']
    
    # Handle special conversions
    if 'Year of HCT' in X.columns:
        X['Year of HCT'] = pd.to_numeric(X['Year of HCT'], errors='coerce')
    if 'HCT-CI' in X.columns:
        X['HCT-CI'] = X['HCT-CI'].astype(str).replace('3+', '3')
        X['HCT-CI'] = pd.to_numeric(X['HCT-CI'], errors='coerce')
    if 'Karnofsky Score' in X.columns:
        X['Karnofsky Score'] = X['Karnofsky Score'].apply(
            lambda x: 90 if x == '>=90' else (80 if x == '<90' else np.nan)
        )
    if 'Donor Age' in X.columns:
        X['Donor Age'] = pd.to_numeric(X['Donor Age'], errors='coerce')
    if 'Age at HCT' in X.columns:
        X['Age at HCT'] = pd.to_numeric(X['Age at HCT'], errors='coerce')
    
    # Identify categorical columns
    categorical_cols = [col for col in FEATURE_COLUMNS if col not in numeric_cols 
                        and col not in ['HCT-CI', 'Karnofsky Score']]
    
    # Encode categorical variables
    label_encoders = {}
    encoding_maps = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = X[col].fillna('Missing').astype(str)
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
        encoding_maps[col] = dict(zip(le.classes_, range(len(le.classes_))))
    
    # Fill numeric missing values
    numeric_medians = {}
    for col in X.columns:
        if col in categorical_cols:
            continue
        X[col] = pd.to_numeric(X[col], errors='coerce')
        if X[col].isna().sum() > 0:
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val)
            numeric_medians[col] = median_val
    
    print(f"Features preprocessed: {X.shape}")
    
    return X, label_encoders, encoding_maps, numeric_medians


def create_cox_labels(time, event):
    """Create labels for XGBoost survival:cox objective."""
    time = np.asarray(time)
    event = np.asarray(event)
    labels = np.where(event == 1, time, -time)
    labels = np.where(labels == 0, 0.001, labels)
    return labels


def calculate_cindex(event_indicator, event_time, risk_score):
    """Calculate Harrell's C-index."""
    try:
        c_index, _, _, _, _ = concordance_index_censored(
            event_indicator.astype(bool), event_time, risk_score
        )
        return c_index
    except:
        return np.nan


def train_cox_model(X_train, time_train, event_train, X_test, time_test, event_test, outcome_name):
    """Train XGBoost Cox survival model."""
    print(f"\n--- Training {outcome_name} ---")
    print(f"  Train: {len(X_train)} ({event_train.sum()} events)")
    print(f"  Test: {len(X_test)} ({event_test.sum()} events)")
    
    y_train = create_cox_labels(time_train, event_train)
    y_test = create_cox_labels(time_test, event_test)
    
    # Hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.05, 0.1],
        'min_child_weight': [1, 3],
        'subsample': [0.8],
        'colsample_bytree': [0.8]
    }
    
    best_cindex = -np.inf
    best_params = None
    
    from itertools import product
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    
    print("  Tuning hyperparameters...")
    for combo in product(*param_values):
        params = dict(zip(param_names, combo))
        
        kf = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = []
        
        for train_idx, val_idx in kf.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr = y_train[train_idx]
            time_val = time_train[val_idx]
            event_val = event_train[val_idx]
            
            model = xgb.XGBRegressor(
                objective='survival:cox',
                eval_metric='cox-nloglik',
                random_state=RANDOM_STATE,
                verbosity=0,
                **params
            )
            model.fit(X_tr, y_tr, verbose=False)
            risk_scores = model.predict(X_val)
            c_idx = calculate_cindex(event_val, time_val, risk_scores)
            if not np.isnan(c_idx):
                cv_scores.append(c_idx)
        
        if cv_scores:
            mean_cindex = np.mean(cv_scores)
            if mean_cindex > best_cindex:
                best_cindex = mean_cindex
                best_params = params.copy()
    
    print(f"  Best CV C-index: {best_cindex:.4f}")
    print(f"  Best params: {best_params}")
    
    # Train final model
    final_model = xgb.XGBRegressor(
        objective='survival:cox',
        eval_metric='cox-nloglik',
        random_state=RANDOM_STATE,
        verbosity=0,
        **best_params
    )
    final_model.fit(X_train, y_train, verbose=False)
    
    # Evaluate
    train_risk = final_model.predict(X_train)
    test_risk = final_model.predict(X_test)
    train_cindex = calculate_cindex(event_train, time_train, train_risk)
    test_cindex = calculate_cindex(event_test, time_test, test_risk)
    
    print(f"  Train C-index: {train_cindex:.4f}")
    print(f"  Test C-index: {test_cindex:.4f}")
    
    return final_model, {
        'best_params': best_params,
        'train_cindex': train_cindex,
        'test_cindex': test_cindex,
        'cv_cindex': best_cindex
    }


def main():
    """Main training function."""
    print("\n" + "=" * 80)
    print("TRAINING AND SAVING XGBOOST MODELS FOR WEB APP")
    print("=" * 80)
    
    # Load data
    df = load_and_preprocess_data()
    
    # Preprocess features
    X, label_encoders, encoding_maps, numeric_medians = preprocess_features(df)
    
    # Save preprocessing info
    preprocessing_info = {
        'encoding_maps': encoding_maps,
        'numeric_medians': {k: float(v) for k, v in numeric_medians.items()},
        'feature_columns': FEATURE_COLUMNS
    }
    
    with open(os.path.join(OUTPUT_DIR, 'preprocessing_info.json'), 'w') as f:
        json.dump(preprocessing_info, f, indent=2)
    print(f"\nSaved: preprocessing_info.json")
    
    # Save label encoders
    joblib.dump(label_encoders, os.path.join(OUTPUT_DIR, 'label_encoders.pkl'))
    print(f"Saved: label_encoders.pkl")
    
    # Train models for each outcome
    outcomes = {
        'OS': ('OS_time', 'OS_event'),
        'NRM': ('NRM_time', 'NRM_event'),
        'Relapse': ('Relapse_time', 'Relapse_event'),
        'cGVHD': ('cGVHD_time', 'cGVHD_event')
    }
    
    all_results = {}
    
    for outcome_name, (time_col, event_col) in outcomes.items():
        time_data = df[time_col].values
        event_data = df[event_col].values
        
        # 80/20 split
        X_train, X_test, time_train, time_test, event_train, event_test = train_test_split(
            X, time_data, event_data,
            test_size=0.2, random_state=RANDOM_STATE, stratify=event_data
        )
        
        model, results = train_cox_model(
            X_train, time_train, event_train,
            X_test, time_test, event_test,
            outcome_name
        )
        
        # Save model
        model_path = os.path.join(OUTPUT_DIR, f'xgboost_{outcome_name}.json')
        model.save_model(model_path)
        print(f"  Saved: xgboost_{outcome_name}.json")
        
        all_results[outcome_name] = results
    
    # Save results summary
    results_df = pd.DataFrame([
        {
            'Outcome': k,
            'Train_Cindex': v['train_cindex'],
            'CV_Cindex': v['cv_cindex'],
            'Test_Cindex': v['test_cindex'],
            **v['best_params']
        }
        for k, v in all_results.items()
    ])
    results_df.to_csv(os.path.join(OUTPUT_DIR, 'model_performance.csv'), index=False)
    print(f"\nSaved: model_performance.csv")
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"\nAll models saved to: {OUTPUT_DIR}")
    print("\nModel Performance Summary:")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
