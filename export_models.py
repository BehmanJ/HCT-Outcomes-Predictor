"""
Export Trained Models Script
==============================
This script exports trained models from the original analysis scripts
for use in the web application.

Run this script after training models to save them for the web app.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib

# Add paths to model directories
COX_DIR = '/Users/j/Desktop/Cox PH 2026'
RSF_DIR = '/Users/j/Desktop/Version Jan 20'
XGBOOST_DIR = '/Users/j/Desktop/XGBoost Jan 20 2026'
OUTPUT_DIR = '/Users/j/Desktop/HCT_Ensemble_WebApp/trained_models'

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("MODEL EXPORT UTILITY")
print("=" * 80)

# ==============================================================================
# EXPORT XGBOOST MODELS
# ==============================================================================

def export_xgboost_models():
    """
    Export trained XGBoost models.
    
    Note: This requires running the XGBoost training scripts first,
    then modifying them to save the models using joblib or xgboost's
    native save_model() function.
    """
    print("\n--- XGBoost Models ---")
    print("To export XGBoost models, add the following to your training scripts:")
    print("""
    # At the end of training:
    import joblib
    
    # Save the model
    joblib.dump(model, f'{OUTPUT_DIR}/xgboost_{outcome_name}.pkl')
    
    # Or use XGBoost's native format:
    model.save_model(f'{OUTPUT_DIR}/xgboost_{outcome_name}.json')
    """)
    
    # Check if models exist
    for outcome in ['OS', 'NRM', 'Relapse', 'cGVHD']:
        pkl_path = os.path.join(OUTPUT_DIR, f'xgboost_{outcome}.pkl')
        json_path = os.path.join(OUTPUT_DIR, f'xgboost_{outcome}.json')
        
        if os.path.exists(pkl_path):
            print(f"  Found: xgboost_{outcome}.pkl")
        elif os.path.exists(json_path):
            print(f"  Found: xgboost_{outcome}.json")
        else:
            print(f"  Missing: xgboost_{outcome} model")

# ==============================================================================
# EXPORT COX PH COEFFICIENTS FROM R
# ==============================================================================

def export_cox_coefficients():
    """
    Instructions for exporting Cox PH coefficients from R.
    """
    print("\n--- Cox PH Models ---")
    print("To export Cox PH coefficients from R, add the following to your R script:")
    print("""
    # In R, after fitting Cox models:
    
    # For each outcome model (e.g., os_results$model):
    cox_coefs <- list(
      coefficients = coef(os_results$model),
      baseline_hazard = basehaz(os_results$model, centered = FALSE)
    )
    
    # Save to JSON or CSV
    write.csv(
      data.frame(
        Variable = names(cox_coefs$coefficients),
        Coefficient = unname(cox_coefs$coefficients)
      ),
      "cox_os_coefficients.csv",
      row.names = FALSE
    )
    
    # Save baseline hazard
    write.csv(
      cox_coefs$baseline_hazard,
      "cox_os_baseline_hazard.csv",
      row.names = FALSE
    )
    """)
    
    # Check for existing coefficient files
    coef_files = [
        'cox_os_coefficients.csv',
        'cox_nrm_coefficients.csv',
        'cox_relapse_coefficients.csv',
        'cox_cgvhd_coefficients.csv'
    ]
    
    for f in coef_files:
        path = os.path.join(OUTPUT_DIR, f)
        if os.path.exists(path):
            print(f"  Found: {f}")
        else:
            print(f"  Missing: {f}")

# ==============================================================================
# EXPORT RSF MODEL INFO FROM R
# ==============================================================================

def export_rsf_info():
    """
    Instructions for exporting RSF model information from R.
    """
    print("\n--- RSF Models ---")
    print("To export RSF partial effects from R, add the following to your R script:")
    print("""
    # In R, after fitting RSF models:
    library(randomForestSRC)
    
    # For each model:
    rsf_model <- all_results$OS$model
    
    # Export variable importance
    vimp <- data.frame(
      Variable = names(rsf_model$importance),
      Importance = rsf_model$importance
    )
    write.csv(vimp, "rsf_os_vimp.csv", row.names = FALSE)
    
    # Export partial effects for key variables
    for (var in c("Disease.Status", "Cytogenetic.Score", etc.)) {
      pd <- partial(rsf_model, partial.xvar = var)
      write.csv(pd, paste0("rsf_os_partial_", var, ".csv"), row.names = FALSE)
    }
    
    # Alternatively, use reticulate to interface R and Python directly
    """)
    
    # Check for existing RSF files
    rds_path = os.path.join(RSF_DIR, 'Results/RSF_v3.5/RSF_All_Models_v35.rds')
    if os.path.exists(rds_path):
        print(f"  Found: RSF_All_Models_v35.rds")
        print("  Note: RDS files cannot be loaded directly in Python.")
        print("        Use rpy2 or export to CSV/JSON format.")
    else:
        print(f"  Missing: RSF_All_Models_v35.rds")

# ==============================================================================
# LOAD EXISTING RESULTS
# ==============================================================================

def load_existing_results():
    """
    Load and summarize existing results from analysis directories.
    """
    print("\n" + "=" * 80)
    print("EXISTING ANALYSIS RESULTS")
    print("=" * 80)
    
    # XGBoost results
    xgb_results = os.path.join(XGBOOST_DIR, 'Results')
    if os.path.exists(xgb_results):
        print("\n--- XGBoost Results ---")
        for f in os.listdir(xgb_results):
            if f.endswith('.csv'):
                print(f"  {f}")
    
    # Cox PH results
    cox_results = os.path.join(COX_DIR, 'Results')
    if os.path.exists(cox_results):
        print("\n--- Cox PH Results ---")
        for subdir in ['Overall_Survival', 'NRM', 'Relapse', 'cGVHD', 'Summary']:
            subdir_path = os.path.join(cox_results, subdir)
            if os.path.exists(subdir_path):
                print(f"  {subdir}/")
                for f in os.listdir(subdir_path):
                    if f.endswith('.csv'):
                        print(f"    {f}")
    
    # RSF results
    rsf_results = os.path.join(RSF_DIR, 'Results/RSF_v3.5')
    if os.path.exists(rsf_results):
        print("\n--- RSF Results ---")
        for f in sorted(os.listdir(rsf_results)):
            if f.endswith('.csv') or f.endswith('.rds'):
                print(f"  {f}")

# ==============================================================================
# CREATE SAMPLE MODEL LOADING CODE
# ==============================================================================

def create_model_loader():
    """
    Create a Python module for loading exported models.
    """
    loader_code = '''"""
Model Loader Module
====================
Loads trained models for the web application.
"""

import os
import pandas as pd
import numpy as np
import joblib

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_SUBDIR = os.path.join(MODEL_DIR, 'trained_models')

def load_xgboost_model(outcome):
    """Load XGBoost model for a specific outcome."""
    pkl_path = os.path.join(MODELS_SUBDIR, f'xgboost_{outcome}.pkl')
    json_path = os.path.join(MODELS_SUBDIR, f'xgboost_{outcome}.json')
    
    if os.path.exists(pkl_path):
        return joblib.load(pkl_path)
    elif os.path.exists(json_path):
        import xgboost as xgb
        model = xgb.XGBRegressor()
        model.load_model(json_path)
        return model
    else:
        return None

def load_cox_coefficients(outcome):
    """Load Cox PH coefficients for a specific outcome."""
    coef_path = os.path.join(MODELS_SUBDIR, f'cox_{outcome.lower()}_coefficients.csv')
    
    if os.path.exists(coef_path):
        return pd.read_csv(coef_path)
    else:
        return None

def load_rsf_vimp(outcome):
    """Load RSF variable importance for a specific outcome."""
    vimp_path = os.path.join(MODELS_SUBDIR, f'rsf_{outcome.lower()}_vimp.csv')
    
    if os.path.exists(vimp_path):
        return pd.read_csv(vimp_path)
    else:
        return None

def check_models_available():
    """Check which models are available."""
    available = {
        'xgboost': {},
        'cox': {},
        'rsf': {}
    }
    
    for outcome in ['OS', 'NRM', 'Relapse', 'cGVHD']:
        available['xgboost'][outcome] = load_xgboost_model(outcome) is not None
        available['cox'][outcome] = load_cox_coefficients(outcome) is not None
        available['rsf'][outcome] = load_rsf_vimp(outcome) is not None
    
    return available

if __name__ == "__main__":
    print("Checking available models...")
    available = check_models_available()
    
    for model_type, outcomes in available.items():
        print(f"\\n{model_type.upper()}:")
        for outcome, is_available in outcomes.items():
            status = "✓" if is_available else "✗"
            print(f"  {outcome}: {status}")
'''
    
    loader_path = os.path.join(OUTPUT_DIR, '..', 'model_loader.py')
    with open(loader_path, 'w') as f:
        f.write(loader_code)
    
    print(f"\nCreated: model_loader.py")

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    export_xgboost_models()
    export_cox_coefficients()
    export_rsf_info()
    load_existing_results()
    create_model_loader()
    
    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("""
1. Run the XGBoost training scripts with model saving enabled
2. Run the R scripts to export Cox PH and RSF coefficients
3. Place exported files in: {}/
4. Run this script again to verify models are available
5. Update prediction_engine.py to use loaded models

For immediate use, the web app uses pre-defined coefficients 
based on the original analyses, which provide good approximations
of the trained model predictions.
""".format(OUTPUT_DIR))

if __name__ == "__main__":
    main()
