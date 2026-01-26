"""
Train and Save Optimal XGBoost Models for Web Application
==========================================================
Uses the best-performing model type for each outcome:
- Overall Survival: Cox PH (C-index 0.634)
- NRM: Cox PH (C-index 0.608)
- Relapse: AFT (C-index 0.610)
- Chronic GVHD: Fine-Gray (C-index 0.635)

Includes OOF SHAP computation for unbiased feature importance.
"""

import pandas as pd
import numpy as np
import os
import joblib
import json
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sksurv.metrics import concordance_index_censored
import shap
import warnings
warnings.filterwarnings('ignore')

# Configuration
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
LANDMARK_TIME = 36
N_FOLDS = 5

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

# Best hyperparameters from prior GridSearchCV tuning
BEST_PARAMS = {
    'OS': {
        'n_estimators': 300, 'max_depth': 7, 'learning_rate': 0.01,
        'min_child_weight': 1, 'subsample': 0.8, 'colsample_bytree': 0.8
    },
    'NRM': {
        'n_estimators': 300, 'max_depth': 7, 'learning_rate': 0.01,
        'min_child_weight': 1, 'subsample': 0.7, 'colsample_bytree': 0.8
    },
    'Relapse': {
        'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.05,
        'min_child_weight': 1, 'subsample': 0.8, 'colsample_bytree': 0.8
    },
    'cGVHD': {
        'n_estimators': 200, 'max_depth': 7, 'learning_rate': 0.1,
        'min_child_weight': 1, 'subsample': 0.7, 'colsample_bytree': 0.8
    }
}

MODEL_TYPES = {
    'OS': 'cox',
    'NRM': 'cox',
    'Relapse': 'aft',
    'cGVHD': 'fine_gray'
}


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
    
    # cGVHD competing event (death without cGVHD)
    df['cGVHD_competing'] = (
        (df['OS_status'] == 1) & 
        (df['Time to Follow-up (months)'] <= LANDMARK_TIME) & 
        (df['cGVHD_status'] == 0)
    ).astype(int)
    
    print(f"\nOutcome Summary at {LANDMARK_TIME} months:")
    print(f"  OS Events: {df['OS_event'].sum()} ({100*df['OS_event'].mean():.1f}%)")
    print(f"  NRM Events: {df['NRM_event'].sum()} ({100*df['NRM_event'].mean():.1f}%)")
    print(f"  Relapse Events: {df['Relapse_event'].sum()} ({100*df['Relapse_event'].mean():.1f}%)")
    print(f"  cGVHD Events: {df['cGVHD_event'].sum()} ({100*df['cGVHD_event'].mean():.1f}%)")
    
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


def calculate_cindex(event_indicator, event_time, risk_score):
    """Calculate Harrell's C-index."""
    try:
        c_index, _, _, _, _ = concordance_index_censored(
            np.asarray(event_indicator).astype(bool), 
            np.asarray(event_time), 
            np.asarray(risk_score)
        )
        return c_index
    except:
        return np.nan


def compute_pseudo_observations(time, event, competing, landmark):
    """Compute Aalen-Johansen pseudo-observations for Fine-Gray."""
    n = len(time)
    time = np.asarray(time)
    event = np.asarray(event)
    competing = np.asarray(competing)
    
    # State: 0=censored, 1=event of interest, 2=competing event
    state = np.zeros(n, dtype=int)
    state[event == 1] = 1
    state[competing == 1] = 2
    
    def aalen_johansen_cif(t, s, tau):
        """Compute CIF at tau."""
        order = np.argsort(t)
        t_sorted = t[order]
        s_sorted = s[order]
        
        unique_times = np.unique(t_sorted[t_sorted <= tau])
        
        surv = 1.0
        cif = 0.0
        
        for tj in unique_times:
            at_risk = np.sum(t_sorted >= tj)
            if at_risk == 0:
                break
            
            d1 = np.sum((t_sorted == tj) & (s_sorted == 1))
            d2 = np.sum((t_sorted == tj) & (s_sorted == 2))
            
            cif += surv * (d1 / at_risk)
            surv *= (1 - (d1 + d2) / at_risk)
        
        return cif
    
    # Overall CIF
    cif_overall = aalen_johansen_cif(time, state, landmark)
    
    # Pseudo-observations using jackknife
    pseudo_obs = np.zeros(n)
    
    print(f"    Computing {n} pseudo-observations...")
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        
        cif_minus_i = aalen_johansen_cif(time[mask], state[mask], landmark)
        pseudo_obs[i] = n * cif_overall - (n - 1) * cif_minus_i
        
        if (i + 1) % 1000 == 0:
            print(f"      Processed {i+1}/{n}...")
    
    # Clip to [0, 1]
    pseudo_obs = np.clip(pseudo_obs, 0, 1)
    
    return pseudo_obs, cif_overall


def train_cox_model(X, time, event, outcome_name, params):
    """Train Cox PH model with CV and compute OOF SHAP."""
    print(f"\n{'='*70}")
    print(f"Training {outcome_name}: Cox Proportional Hazards")
    print(f"{'='*70}")
    
    X = X.reset_index(drop=True)
    time = np.asarray(time)
    event = np.asarray(event)
    n_samples = len(X)
    n_features = X.shape[1]
    
    # Storage for OOF results
    oof_shap_values = np.zeros((n_samples, n_features), dtype=np.float64)
    oof_predictions = np.zeros(n_samples, dtype=np.float64)
    processed = np.zeros(n_samples, dtype=bool)
    
    fold_metrics = []
    models = []
    
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    print(f"Running {N_FOLDS}-Fold CV with OOF SHAP...")
    print(f"Samples: {n_samples}, Events: {event.sum()} ({100*event.mean():.1f}%)")
    
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, event)):
        print(f"\n  Fold {fold_idx + 1}/{N_FOLDS}")
        
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        time_train = time[train_idx]
        time_test = time[test_idx]
        event_train = event[train_idx]
        event_test = event[test_idx]
        
        # Cox survival format: positive = uncensored, negative = censored
        y_train = np.where(event_train == 1, time_train, -time_train)
        y_train = np.where(y_train == 0, 0.001, y_train)
        
        model = xgb.XGBRegressor(
            objective='survival:cox',
            eval_metric='cox-nloglik',
            random_state=RANDOM_STATE,
            verbosity=0,
            **params
        )
        model.fit(X_train, y_train, verbose=False)
        models.append(model)
        
        # Predictions
        risk_scores = model.predict(X_test)
        oof_predictions[test_idx] = risk_scores
        
        # SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        oof_shap_values[test_idx] = shap_values
        
        processed[test_idx] = True
        
        c_index = calculate_cindex(event_test, time_test, risk_scores)
        fold_metrics.append(c_index)
        
        print(f"    Train: {len(train_idx)}, Test: {len(test_idx)}, C-index: {c_index:.4f}")
    
    # Train final model on all data
    print("\n  Training final model on all data...")
    y_all = np.where(event == 1, time, -time)
    y_all = np.where(y_all == 0, 0.001, y_all)
    
    final_model = xgb.XGBRegressor(
        objective='survival:cox',
        eval_metric='cox-nloglik',
        random_state=RANDOM_STATE,
        verbosity=0,
        **params
    )
    final_model.fit(X, y_all, verbose=False)
    
    cv_cindex = np.mean(fold_metrics)
    overall_cindex = calculate_cindex(event, time, oof_predictions)
    
    print(f"\n  CV C-index: {cv_cindex:.4f} (±{np.std(fold_metrics):.4f})")
    print(f"  Overall OOF C-index: {overall_cindex:.4f}")
    
    # Compute SHAP importance
    shap_importance = pd.DataFrame({
        'feature': list(X.columns),
        'mean_abs_shap': np.abs(oof_shap_values).mean(axis=0)
    }).sort_values('mean_abs_shap', ascending=False)
    
    return {
        'model': final_model,
        'model_type': 'cox',
        'cv_cindex': cv_cindex,
        'overall_cindex': overall_cindex,
        'shap_importance': shap_importance,
        'oof_shap_values': oof_shap_values,
        'params': params
    }


def train_aft_model(X, time, event, outcome_name, params):
    """Train AFT model with CV and compute OOF SHAP."""
    print(f"\n{'='*70}")
    print(f"Training {outcome_name}: Accelerated Failure Time")
    print(f"{'='*70}")
    
    X = X.reset_index(drop=True)
    time = np.asarray(time)
    event = np.asarray(event)
    time = np.maximum(time, 0.001)
    
    n_samples = len(X)
    n_features = X.shape[1]
    
    oof_shap_values = np.zeros((n_samples, n_features), dtype=np.float64)
    oof_predictions = np.zeros(n_samples, dtype=np.float64)
    processed = np.zeros(n_samples, dtype=bool)
    
    fold_metrics = []
    models = []
    
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    print(f"Running {N_FOLDS}-Fold CV with OOF SHAP...")
    print(f"Samples: {n_samples}, Events: {event.sum()} ({100*event.mean():.1f}%)")
    
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, event)):
        print(f"\n  Fold {fold_idx + 1}/{N_FOLDS}")
        
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        time_train = time[train_idx]
        time_test = time[test_idx]
        event_train = event[train_idx]
        event_test = event[test_idx]
        
        # For sklearn-compatible SHAP, use log-transformed time
        y_train_aft = np.log(time_train + 0.001)
        
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            random_state=RANDOM_STATE,
            verbosity=0,
            **params
        )
        model.fit(X_train, y_train_aft, verbose=False)
        models.append(model)
        
        # Predictions (negative for risk ordering)
        pred_log_time = model.predict(X_test)
        risk_scores = -pred_log_time
        oof_predictions[test_idx] = risk_scores
        
        # SHAP values (negate so positive = higher risk)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        oof_shap_values[test_idx] = -shap_values
        
        processed[test_idx] = True
        
        c_index = calculate_cindex(event_test, time_test, risk_scores)
        fold_metrics.append(c_index)
        
        print(f"    Train: {len(train_idx)}, Test: {len(test_idx)}, C-index: {c_index:.4f}")
    
    # Train final model on all data
    print("\n  Training final model on all data...")
    y_all_aft = np.log(time + 0.001)
    
    final_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=RANDOM_STATE,
        verbosity=0,
        **params
    )
    final_model.fit(X, y_all_aft, verbose=False)
    
    cv_cindex = np.mean(fold_metrics)
    overall_cindex = calculate_cindex(event, time, oof_predictions)
    
    print(f"\n  CV C-index: {cv_cindex:.4f} (±{np.std(fold_metrics):.4f})")
    print(f"  Overall OOF C-index: {overall_cindex:.4f}")
    
    shap_importance = pd.DataFrame({
        'feature': list(X.columns),
        'mean_abs_shap': np.abs(oof_shap_values).mean(axis=0)
    }).sort_values('mean_abs_shap', ascending=False)
    
    return {
        'model': final_model,
        'model_type': 'aft',
        'cv_cindex': cv_cindex,
        'overall_cindex': overall_cindex,
        'shap_importance': shap_importance,
        'oof_shap_values': oof_shap_values,
        'params': params
    }


def train_fine_gray_model(X, time, event, competing, outcome_name, params):
    """Train Fine-Gray model with CV and compute OOF SHAP."""
    print(f"\n{'='*70}")
    print(f"Training {outcome_name}: Fine-Gray Subdistribution Hazard")
    print(f"{'='*70}")
    
    X = X.reset_index(drop=True)
    time = np.asarray(time)
    event = np.asarray(event)
    competing = np.asarray(competing)
    
    n_samples = len(X)
    n_features = X.shape[1]
    
    # Compute pseudo-observations
    print("  Computing pseudo-observations...")
    pseudo_obs, cif_overall = compute_pseudo_observations(time, event, competing, LANDMARK_TIME)
    print(f"  Overall CIF: {cif_overall:.4f}")
    
    oof_shap_values = np.zeros((n_samples, n_features), dtype=np.float64)
    oof_predictions = np.zeros(n_samples, dtype=np.float64)
    processed = np.zeros(n_samples, dtype=bool)
    
    fold_metrics = []
    models = []
    
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    print(f"\nRunning {N_FOLDS}-Fold CV with OOF SHAP...")
    print(f"Samples: {n_samples}, Events: {event.sum()} ({100*event.mean():.1f}%)")
    
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, event)):
        print(f"\n  Fold {fold_idx + 1}/{N_FOLDS}")
        
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = pseudo_obs[train_idx]
        time_test = time[test_idx]
        event_test = event[test_idx]
        
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            random_state=RANDOM_STATE,
            verbosity=0,
            **params
        )
        model.fit(X_train, y_train, verbose=False)
        models.append(model)
        
        # Predictions
        pred = model.predict(X_test)
        oof_predictions[test_idx] = pred
        
        # SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        oof_shap_values[test_idx] = shap_values
        
        processed[test_idx] = True
        
        c_index = calculate_cindex(event_test, time_test, pred)
        fold_metrics.append(c_index)
        
        print(f"    Train: {len(train_idx)}, Test: {len(test_idx)}, C-index: {c_index:.4f}")
    
    # Train final model on all data
    print("\n  Training final model on all data...")
    final_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=RANDOM_STATE,
        verbosity=0,
        **params
    )
    final_model.fit(X, pseudo_obs, verbose=False)
    
    cv_cindex = np.mean(fold_metrics)
    overall_cindex = calculate_cindex(event, time, oof_predictions)
    
    print(f"\n  CV C-index: {cv_cindex:.4f} (±{np.std(fold_metrics):.4f})")
    print(f"  Overall OOF C-index: {overall_cindex:.4f}")
    
    shap_importance = pd.DataFrame({
        'feature': list(X.columns),
        'mean_abs_shap': np.abs(oof_shap_values).mean(axis=0)
    }).sort_values('mean_abs_shap', ascending=False)
    
    return {
        'model': final_model,
        'model_type': 'fine_gray',
        'cv_cindex': cv_cindex,
        'overall_cindex': overall_cindex,
        'cif_overall': cif_overall,
        'shap_importance': shap_importance,
        'oof_shap_values': oof_shap_values,
        'params': params
    }


def main():
    print("\n" + "=" * 80)
    print("TRAINING OPTIMAL XGBOOST MODELS FOR WEB APP")
    print("=" * 80)
    print("""
    Model Selection:
    - Overall Survival: Cox PH
    - NRM: Cox PH
    - Relapse: AFT
    - Chronic GVHD: Fine-Gray
    """)
    
    # Load and preprocess data
    df = load_and_preprocess_data()
    X, label_encoders, encoding_maps, numeric_medians = preprocess_features(df)
    
    # Save preprocessing info
    preprocessing_info = {
        'encoding_maps': encoding_maps,
        'numeric_medians': {k: float(v) for k, v in numeric_medians.items()},
        'feature_columns': FEATURE_COLUMNS,
        'model_types': MODEL_TYPES
    }
    with open(os.path.join(OUTPUT_DIR, 'preprocessing_info.json'), 'w') as f:
        json.dump(preprocessing_info, f, indent=2)
    print(f"\nSaved: preprocessing_info.json")
    
    joblib.dump(label_encoders, os.path.join(OUTPUT_DIR, 'label_encoders.pkl'))
    print(f"Saved: label_encoders.pkl")
    
    all_results = {}
    all_shap_importance = []
    
    # =========================================================================
    # 1. OVERALL SURVIVAL - Cox PH
    # =========================================================================
    results_os = train_cox_model(
        X=X.copy(),
        time=df['OS_time'].values,
        event=df['OS_event'].values,
        outcome_name='Overall Survival',
        params=BEST_PARAMS['OS']
    )
    results_os['model'].save_model(os.path.join(OUTPUT_DIR, 'xgboost_OS.json'))
    print(f"  Saved: xgboost_OS.json")
    all_results['OS'] = results_os
    
    fi = results_os['shap_importance'].copy()
    fi['outcome'] = 'OS'
    all_shap_importance.append(fi)
    
    # =========================================================================
    # 2. NRM - Cox PH
    # =========================================================================
    results_nrm = train_cox_model(
        X=X.copy(),
        time=df['NRM_time'].values,
        event=df['NRM_event'].values,
        outcome_name='NRM',
        params=BEST_PARAMS['NRM']
    )
    results_nrm['model'].save_model(os.path.join(OUTPUT_DIR, 'xgboost_NRM.json'))
    print(f"  Saved: xgboost_NRM.json")
    all_results['NRM'] = results_nrm
    
    fi = results_nrm['shap_importance'].copy()
    fi['outcome'] = 'NRM'
    all_shap_importance.append(fi)
    
    # =========================================================================
    # 3. RELAPSE - AFT
    # =========================================================================
    results_rel = train_aft_model(
        X=X.copy(),
        time=df['Relapse_time'].values,
        event=df['Relapse_event'].values,
        outcome_name='Relapse',
        params=BEST_PARAMS['Relapse']
    )
    results_rel['model'].save_model(os.path.join(OUTPUT_DIR, 'xgboost_Relapse.json'))
    print(f"  Saved: xgboost_Relapse.json")
    all_results['Relapse'] = results_rel
    
    fi = results_rel['shap_importance'].copy()
    fi['outcome'] = 'Relapse'
    all_shap_importance.append(fi)
    
    # =========================================================================
    # 4. CHRONIC GVHD - Fine-Gray
    # =========================================================================
    results_cgvhd = train_fine_gray_model(
        X=X.copy(),
        time=df['cGVHD_time'].values,
        event=df['cGVHD_event'].values,
        competing=df['cGVHD_competing'].values,
        outcome_name='Chronic GVHD',
        params=BEST_PARAMS['cGVHD']
    )
    results_cgvhd['model'].save_model(os.path.join(OUTPUT_DIR, 'xgboost_cGVHD.json'))
    print(f"  Saved: xgboost_cGVHD.json")
    all_results['cGVHD'] = results_cgvhd
    
    fi = results_cgvhd['shap_importance'].copy()
    fi['outcome'] = 'cGVHD'
    all_shap_importance.append(fi)
    
    # =========================================================================
    # Save performance summary and SHAP importance
    # =========================================================================
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    
    # Performance summary
    performance_data = []
    for outcome, res in all_results.items():
        performance_data.append({
            'Outcome': outcome,
            'Model_Type': res['model_type'],
            'CV_Cindex': res['cv_cindex'],
            'OOF_Cindex': res['overall_cindex'],
            **res['params']
        })
    
    performance_df = pd.DataFrame(performance_data)
    performance_df.to_csv(os.path.join(OUTPUT_DIR, 'model_performance.csv'), index=False)
    print(f"Saved: model_performance.csv")
    
    # Combined SHAP importance
    combined_shap = pd.concat(all_shap_importance, ignore_index=True)
    combined_shap.to_csv(os.path.join(OUTPUT_DIR, 'shap_importance.csv'), index=False)
    print(f"Saved: shap_importance.csv")
    
    # Model config for web app
    model_config = {
        'OS': {'model_type': 'cox', 'model_file': 'xgboost_OS.json'},
        'NRM': {'model_type': 'cox', 'model_file': 'xgboost_NRM.json'},
        'Relapse': {'model_type': 'aft', 'model_file': 'xgboost_Relapse.json'},
        'cGVHD': {'model_type': 'fine_gray', 'model_file': 'xgboost_cGVHD.json'}
    }
    with open(os.path.join(OUTPUT_DIR, 'model_config.json'), 'w') as f:
        json.dump(model_config, f, indent=2)
    print(f"Saved: model_config.json")
    
    # =========================================================================
    # Final Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    
    print(f"\nAll models saved to: {OUTPUT_DIR}")
    
    print("\n" + "-" * 60)
    print("MODEL PERFORMANCE SUMMARY")
    print("-" * 60)
    print(f"{'Outcome':<15} {'Model Type':<12} {'CV C-index':<15} {'OOF C-index':<15}")
    print("-" * 60)
    for outcome, res in all_results.items():
        print(f"{outcome:<15} {res['model_type']:<12} {res['cv_cindex']:<15.4f} {res['overall_cindex']:<15.4f}")
    
    print("\n" + "-" * 60)
    print("TOP 5 FEATURES BY OOF SHAP (per outcome)")
    print("-" * 60)
    for outcome, res in all_results.items():
        print(f"\n{outcome} ({res['model_type']}):")
        for _, row in res['shap_importance'].head(5).iterrows():
            print(f"  {row['feature']:30s} {row['mean_abs_shap']:.4f}")
    
    return all_results


if __name__ == "__main__":
    results = main()
