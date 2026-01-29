"""
HCT Outcomes Ensemble Prediction Tool
======================================
Single-page clinical prediction tool for HCT outcomes.
Inspired by CIBMTR and NMDP clinical calculators.

Reference tools:
- https://cibmtr.org/CIBMTR/Resources/Research-Tools-Calculators/aGVHD-Risk-Calculator
- https://bio-prevent.nmdp.org
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import (
    FEATURE_COLUMNS, FEATURE_OPTIONS, REFERENCE_CATEGORIES,
    NUMERIC_FEATURES, OUTCOMES, MODEL_TYPES, TIME_POINTS
)
from prediction_engine import (
    calculate_ensemble_prediction, calculate_survival_curves,
    get_risk_category, get_prediction_summary, get_feature_contributions,
    validate_patient_data, compare_scenarios, get_scenario_differences,
    get_modifiable_scenarios, get_using_trained_models,
    calculate_covariate_effects_table, ADJUSTABLE_COVARIATES
)

# Check if trained models are available
try:
    from model_loader import check_models_available, get_model_config, get_shap_importance
    TRAINED_MODELS_INFO = check_models_available()
    USING_TRAINED_MODELS = any(TRAINED_MODELS_INFO.values())
    MODEL_CONFIG = get_model_config()
    SHAP_IMPORTANCE = get_shap_importance()
except ImportError:
    TRAINED_MODELS_INFO = None
    USING_TRAINED_MODELS = False
    MODEL_CONFIG = None
    SHAP_IMPORTANCE = None

# Page configuration
st.set_page_config(
    page_title="HCT Outcomes Predictor - ALL",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Light theme CSS styling - comprehensive fix for all text visibility
st.markdown("""
<style>
    /* ============================================
       GLOBAL STYLES - LIGHT THEME
       ============================================ */
    
    /* Force light background and dark text globally */
    .stApp, .main, [data-testid="stAppViewContainer"] {
        background-color: #f8fafc !important;
        color: #1e293b !important;
    }
    
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
        background-color: #f8fafc !important;
    }
    
    /* All text elements - ensure dark text */
    p, span, div, label, li, td, th, h1, h2, h3, h4, h5, h6 {
        color: #1e293b !important;
    }
    
    /* ============================================
       HEADER - DARK BACKGROUND, WHITE TEXT
       ============================================ */
    
    .main-header {
        background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%) !important;
        padding: 1.5rem 2rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(30, 64, 175, 0.3);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 1.9rem;
        font-weight: 700;
        color: #ffffff !important;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        color: #e0e7ff !important;
        font-size: 1.05rem;
    }
    
    /* ============================================
       SECTION HEADERS
       ============================================ */
    
    .section-header {
        background-color: #ffffff !important;
        border-left: 4px solid #3b82f6;
        padding: 0.85rem 1.25rem;
        margin: 1.25rem 0;
        font-weight: 600;
        font-size: 1.15rem;
        color: #1e3a8a !important;
        border-radius: 0 6px 6px 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    /* ============================================
       FORM ELEMENTS - ENSURE READABILITY
       ============================================ */
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
    }
    
    .stSelectbox label {
        color: #1e293b !important;
        font-weight: 500 !important;
    }
    
    .stSelectbox [data-baseweb="select"] {
        background-color: #ffffff !important;
    }
    
    .stSelectbox [data-baseweb="select"] > div {
        background-color: #ffffff !important;
        color: #1e293b !important;
    }
    
    /* Selectbox selected value text */
    .stSelectbox [data-baseweb="select"] span {
        color: #1e293b !important;
    }
    
    /* Dropdown menu styling */
    [data-baseweb="popover"] {
        background-color: #ffffff !important;
    }
    
    [data-baseweb="menu"] {
        background-color: #ffffff !important;
    }
    
    [data-baseweb="menu"] li {
        color: #1e293b !important;
        background-color: #ffffff !important;
    }
    
    [data-baseweb="menu"] li:hover {
        background-color: #e2e8f0 !important;
        color: #1e293b !important;
    }
    
    /* Selected option in dropdown */
    [data-baseweb="menu"] [aria-selected="true"] {
        background-color: #dbeafe !important;
        color: #1e3a8a !important;
    }
    
    /* Number input styling */
    .stNumberInput label {
        color: #1e293b !important;
        font-weight: 500 !important;
    }
    
    .stNumberInput input {
        background-color: #ffffff !important;
        color: #1e293b !important;
        border: 1px solid #cbd5e1 !important;
    }
    
    /* Ensure input text is always dark on white background */
    input, select, textarea {
        color: #1e293b !important;
        background-color: #ffffff !important;
    }
    
    /* ============================================
       EXPANDER / COLLAPSIBLE SECTIONS
       ============================================ */
    
    /* Expander container */
    [data-testid="stExpander"] {
        border: 1px solid #cbd5e1 !important;
        border-radius: 8px !important;
        overflow: hidden;
    }
    
    /* Expander header - default state (collapsed) - light background */
    .streamlit-expanderHeader {
        background-color: #f1f5f9 !important;
        color: #1e293b !important;
        border: none !important;
        border-radius: 8px 8px 8px 8px !important;
        padding: 0.75rem 1rem !important;
    }
    
    .streamlit-expanderHeader p {
        color: #1e293b !important;
        font-weight: 600 !important;
    }
    
    .streamlit-expanderHeader svg {
        fill: #1e293b !important;
        stroke: #1e293b !important;
    }
    
    /* Expander header when expanded - dark background */
    [data-testid="stExpander"][open] .streamlit-expanderHeader,
    details[open] > summary.streamlit-expanderHeader {
        background-color: #1e3a8a !important;
        color: #ffffff !important;
        border-radius: 8px 8px 0 0 !important;
    }
    
    [data-testid="stExpander"][open] .streamlit-expanderHeader p,
    details[open] > summary.streamlit-expanderHeader p {
        color: #ffffff !important;
    }
    
    [data-testid="stExpander"][open] .streamlit-expanderHeader svg,
    details[open] > summary.streamlit-expanderHeader svg {
        fill: #ffffff !important;
        stroke: #ffffff !important;
    }
    
    /* Expander content - white background */
    .streamlit-expanderContent {
        background-color: #ffffff !important;
        border: none !important;
        border-top: 1px solid #e2e8f0 !important;
        border-radius: 0 0 8px 8px !important;
        padding: 1rem !important;
    }
    
    .streamlit-expanderContent p,
    .streamlit-expanderContent span,
    .streamlit-expanderContent div,
    .streamlit-expanderContent label {
        color: #1e293b !important;
    }
    
    /* ============================================
       RESULT CARDS
       ============================================ */
    
    .result-card {
        background: #ffffff !important;
        border-radius: 10px;
        padding: 1.25rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border: 1px solid #e2e8f0;
        margin-bottom: 1rem;
    }
    
    .result-card h4 {
        margin: 0 0 0.5rem 0;
        color: #1e293b !important;
        font-size: 1.05rem;
        font-weight: 600;
    }
    
    .result-value {
        font-size: 2.3rem;
        font-weight: 700;
        margin: 0.25rem 0;
        line-height: 1.2;
    }
    
    .result-label {
        font-size: 0.95rem;
        font-weight: 600;
    }
    
    /* ============================================
       INFO BOX
       ============================================ */
    
    .info-box {
        background-color: #eff6ff !important;
        border: 1px solid #93c5fd !important;
        border-radius: 8px;
        padding: 0.9rem 1.15rem;
        margin: 0.5rem 0;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    
    .info-box, .info-box * {
        color: #1e40af !important;
    }
    
    .info-box strong {
        color: #1e3a8a !important;
    }
    
    /* ============================================
       DISCLAIMER BOX
       ============================================ */
    
    .disclaimer {
        background-color: #fef9c3 !important;
        border: 2px solid #facc15 !important;
        border-radius: 8px;
        padding: 1.25rem;
        margin-top: 1.5rem;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    
    .disclaimer, .disclaimer * {
        color: #713f12 !important;
    }
    
    .disclaimer-title {
        font-weight: 700 !important;
        color: #713f12 !important;
        margin-bottom: 0.75rem;
        font-size: 1.1rem;
    }
    
    /* ============================================
       ALERT BOXES (Success, Info, Warning, Error)
       ============================================ */
    
    [data-testid="stAlert"] {
        border-radius: 8px !important;
    }
    
    /* Success alerts */
    .stSuccess, [data-baseweb="notification"][kind="positive"] {
        background-color: #dcfce7 !important;
        border: 1px solid #86efac !important;
    }
    
    .stSuccess *, [data-baseweb="notification"][kind="positive"] * {
        color: #166534 !important;
    }
    
    /* Info alerts */
    .stInfo, [data-baseweb="notification"][kind="info"] {
        background-color: #dbeafe !important;
        border: 1px solid #93c5fd !important;
    }
    
    .stInfo *, [data-baseweb="notification"][kind="info"] * {
        color: #1e40af !important;
    }
    
    /* Warning alerts */
    .stWarning, [data-baseweb="notification"][kind="warning"] {
        background-color: #fef9c3 !important;
        border: 1px solid #fde047 !important;
    }
    
    .stWarning *, [data-baseweb="notification"][kind="warning"] * {
        color: #713f12 !important;
    }
    
    /* Error alerts */
    .stError, [data-baseweb="notification"][kind="negative"] {
        background-color: #fee2e2 !important;
        border: 1px solid #fca5a5 !important;
    }
    
    .stError *, [data-baseweb="notification"][kind="negative"] * {
        color: #991b1b !important;
    }
    
    /* ============================================
       PLOTLY CHARTS - Let Plotly control styling
       ============================================ */
    
    /* Chart container - ensure visibility */
    .js-plotly-plot, .plotly, .plot-container {
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
        min-height: 300px;
    }
    
    /* Ensure iframe/svg is visible */
    .js-plotly-plot .plotly .main-svg {
        display: block !important;
        visibility: visible !important;
    }
    
    /* Let Plotly control colors - don't override */
    .stPlotlyChart {
        display: block !important;
        visibility: visible !important;
    }
    
    [data-testid="stPlotlyChart"] {
        display: block !important;
        visibility: visible !important;
        min-height: 300px;
    }
    
    /* ============================================
       DATAFRAME / TABLES
       ============================================ */
    
    .stDataFrame, [data-testid="stDataFrame"] {
        background-color: #ffffff !important;
    }
    
    .stDataFrame th {
        background-color: #f1f5f9 !important;
        color: #1e293b !important;
        font-weight: 600 !important;
    }
    
    .stDataFrame td {
        background-color: #ffffff !important;
        color: #1e293b !important;
    }
    
    /* Custom HTML table styling - ensure header is visible */
    table thead tr {
        background-color: #1e3a8a !important;
    }
    
    table thead th {
        color: #ffffff !important;
        background-color: #1e3a8a !important;
        font-weight: 600 !important;
    }
    
    table tbody td {
        color: #1e293b !important;
    }
    
    /* ============================================
       BUTTONS
       ============================================ */
    
    .stButton > button {
        background-color: #2563eb !important;
        color: #ffffff !important;
        border: none !important;
        padding: 0.5rem 2rem;
        font-weight: 600;
        border-radius: 6px;
        font-size: 1rem;
    }
    
    .stButton > button:hover {
        background-color: #1d4ed8 !important;
        color: #ffffff !important;
    }
    
    /* ============================================
       CAPTION TEXT
       ============================================ */
    
    .stCaption, [data-testid="stCaption"] {
        color: #475569 !important;
    }
    
    /* ============================================
       MARKDOWN TEXT
       ============================================ */
    
    .stMarkdown, .stMarkdown p, .stMarkdown span {
        color: #1e293b !important;
    }
    
    .stMarkdown strong {
        color: #0f172a !important;
    }
    
    .stMarkdown em {
        color: #334155 !important;
    }
    
    /* ============================================
       FOOTER
       ============================================ */
    
    .footer-text {
        text-align: center;
        padding: 1rem 0;
        line-height: 1.6;
    }
    
    .footer-text, .footer-text * {
        color: #475569 !important;
        font-size: 0.9rem;
    }
    
    .footer-text strong {
        color: #1e293b !important;
    }
    
    /* ============================================
       HIDE STREAMLIT BRANDING
       ============================================ */
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Horizontal rule */
    hr {
        border-color: #e2e8f0 !important;
    }
    
    /* ============================================
       METRIC COMPONENTS
       ============================================ */
    
    [data-testid="stMetricValue"] {
        color: #1e293b !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #475569 !important;
    }
    
    /* ============================================
       COLUMN CONTAINERS
       ============================================ */
    
    [data-testid="column"] {
        background-color: transparent !important;
    }
    
    /* ============================================
       TABS (if used)
       ============================================ */
    
    .stTabs [data-baseweb="tab-list"] {
        background-color: #ffffff !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #1e293b !important;
    }
</style>
""", unsafe_allow_html=True)


def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>HCT Outcomes Prediction Tool</h1>
        <p>Predicting 36-month outcomes for adult patients with ALL undergoing allogeneic HCT</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Brief intro
    col_intro1, col_intro2 = st.columns([3, 1])
    with col_intro1:
        st.markdown("""
        This tool estimates the probability of key outcomes at 36 months following allogeneic 
        hematopoietic cell transplantation (allo-HCT) for acute lymphoblastic leukemia (ALL). 
        Predictions combine three statistical models: Cox Proportional Hazards, Random Survival Forest, and XGBoost.
        """)
    with col_intro2:
        if USING_TRAINED_MODELS:
            st.success("✓ Using trained models", icon="🔬")
        else:
            st.info("Using coefficient estimates", icon="📊")
    
    st.markdown("---")
    
    # Main layout: Inputs on left, Results on right
    col_inputs, col_results = st.columns([1, 1.2])
    
    # =========================================================================
    # LEFT COLUMN: Patient Characteristics Input
    # =========================================================================
    with col_inputs:
        st.markdown('<div class="section-header">Enter Patient Characteristics</div>', 
                    unsafe_allow_html=True)
        
        # Disease Characteristics
        with st.expander("**Disease Characteristics**", expanded=True):
            disease_status = st.selectbox(
                "Disease Status at HCT",
                options=FEATURE_OPTIONS['Disease Status'],
                key="disease_status",
                help="Complete remission status and MRD at transplant"
            )
            
            cytogenetics = st.selectbox(
                "Cytogenetic Risk",
                options=FEATURE_OPTIONS['Cytogenetic Score'],
                key="cytogenetics",
                help="Cytogenetic risk classification"
            )
            
            immunophenotype = st.selectbox(
                "Immunophenotype",
                options=FEATURE_OPTIONS['Immunophenotype'],
                key="immunophenotype"
            )
            
            # Ph+/BCR-ABL1 logic: T-cell ALL is always Ph-negative (use T-cell category)
            if immunophenotype == 'T-cell':
                ph_status = 'T-cell'
                st.caption("Ph+/BCR-ABL1: **T-cell** (Ph+ not applicable to T-cell ALL)")
            else:
                ph_status = st.selectbox(
                    "Ph+/BCR-ABL1 Status",
                    options=['No', 'Yes'],  # Only No/Yes for B-cell
                    key="ph_status",
                    help="Philadelphia chromosome / BCR-ABL1 fusion status"
                )
            
            time_dx_hct = st.selectbox(
                "Time from Diagnosis to HCT",
                options=FEATURE_OPTIONS['Time Dx to HCT'],
                key="time_dx_hct"
            )
        
        # Patient Characteristics
        with st.expander("**Patient Characteristics**", expanded=True):
            col_age, col_year = st.columns(2)
            with col_age:
                age = st.number_input(
                    "Age at HCT (years)",
                    min_value=18, max_value=80, value=45,
                    key="age",
                    help="Patient age at time of transplant"
                )
            with col_year:
                year_hct = st.number_input(
                    "Year of HCT",
                    min_value=2011, max_value=2018, value=2015,
                    key="year_hct",
                    help="Dataset range: 2011-2018"
                )
            
            col_sex, col_race = st.columns(2)
            with col_sex:
                patient_sex = st.selectbox(
                    "Patient Sex", 
                    options=['Male', 'Female'],
                    key="patient_sex"
                )
            with col_race:
                race = st.selectbox(
                    "Race/Ethnicity", 
                    options=FEATURE_OPTIONS['Race/Ethnicity'],
                    key="race"
                )
            
            col_kps, col_hctci = st.columns(2)
            with col_kps:
                kps = st.selectbox(
                    "Karnofsky Score",
                    options=FEATURE_OPTIONS['Karnofsky Score'],
                    key="kps",
                    help="Performance status at HCT"
                )
            with col_hctci:
                hctci = st.selectbox(
                    "HCT-CI Score",
                    options=FEATURE_OPTIONS['HCT-CI'],
                    key="hctci",
                    help="Comorbidity index"
                )
        
        # Transplant Characteristics
        with st.expander("**Transplant Characteristics**", expanded=True):
            donor_type = st.selectbox(
                "Donor Type",
                options=FEATURE_OPTIONS['Donor Type'],
                index=1,  # Default to 8/8 MUD (non-reference)
                key="donor_type",
                help="Donor relationship and HLA matching"
            )
            
            col_donor_age, col_donor_sex = st.columns(2)
            with col_donor_age:
                donor_age = st.number_input(
                    "Donor Age (years)",
                    min_value=0, max_value=75, value=35,
                    key="donor_age"
                )
            with col_donor_sex:
                donor_sex = st.selectbox(
                    "Donor Sex",
                    options=['Male', 'Female'],
                    key="donor_sex"
                )
            
            # Derive sex match from patient sex and donor sex
            # Format is Donor-Recipient (e.g., F-M means Female donor to Male recipient)
            recipient_abbrev = 'M' if patient_sex == 'Male' else 'F'
            donor_abbrev = 'M' if donor_sex == 'Male' else 'F'
            sex_match_derived = f"{donor_abbrev}-{recipient_abbrev}"
            
            # Map to the original categories (F-M has higher GVHD risk)
            if sex_match_derived == 'F-M':
                sex_match = 'F-M'  # Female donor to Male recipient (higher risk)
                st.caption(f"Sex Match: **{sex_match_derived}** (Female→Male, higher GVHD risk)")
            else:
                sex_match = 'Other'  # All other combinations
                st.caption(f"Sex Match: **{sex_match_derived}** (classified as Other)")
            
            cmv_status = st.selectbox(
                "Donor/Recipient CMV Status",
                options=FEATURE_OPTIONS['Donor/Recipient CMV'],
                key="cmv_status",
                help="+/+ means both positive, -/- means both negative"
            )
            
            graft_type = st.selectbox(
                "Graft Type",
                options=FEATURE_OPTIONS['Graft Type'],
                key="graft_type"
            )
            
            conditioning = st.selectbox(
                "Conditioning Regimen",
                options=FEATURE_OPTIONS['Conditioning Regimen'],
                index=1,  # Default to MAC Chemo (non-reference)
                key="conditioning",
                help="MAC = Myeloablative, RIC/NMA = Reduced intensity"
            )
            
            gvhd_prophylaxis = st.selectbox(
                "GVHD Prophylaxis",
                options=FEATURE_OPTIONS['GVHD Prophylaxis'],
                key="gvhd_prophylaxis",
                help="CNI = Calcineurin inhibitor based, PTCy = Post-transplant cyclophosphamide"
            )
            
            tcd = st.selectbox(
                "In Vivo T-cell Depletion (ATG/Alemtuzumab)",
                options=FEATURE_OPTIONS['In Vivo T-cell Depletion (Yes)'],
                index=1,  # Default to Yes (non-reference)
                key="tcd"
            )
        
        # Compile patient data
        patient_data = {
            'Disease Status': disease_status,
            'Cytogenetic Score': cytogenetics,
            'Year of HCT': year_hct,
            'Age at HCT': age,
            'Gender': patient_sex,  # Maps to Gender in the model
            'Race/Ethnicity': race,
            'Karnofsky Score': kps,
            'HCT-CI': hctci,
            'Time Dx to HCT': time_dx_hct,
            'Immunophenotype': immunophenotype,
            'Ph+/BCR-ABL1': ph_status,
            'Donor Type': donor_type,
            'Donor Age': donor_age,
            'Donor/Recipient Sex Match': sex_match,  # Derived from patient sex + donor sex
            'Donor/Recipient CMV': cmv_status,
            'Graft Type': graft_type,
            'Conditioning Regimen': conditioning,
            'GVHD Prophylaxis': gvhd_prophylaxis,
            'In Vivo T-cell Depletion (Yes)': tcd
        }
        
        validated_data, _ = validate_patient_data(patient_data)
    
    # =========================================================================
    # RIGHT COLUMN: Predictions and Results
    # =========================================================================
    with col_results:
        st.markdown('<div class="section-header">Predicted Outcomes at 36 Months</div>', 
                    unsafe_allow_html=True)
        
        # Calculate all predictions
        summary = get_prediction_summary(validated_data)
        
        # Main outcome cards in 2x2 grid
        row1_col1, row1_col2 = st.columns(2)
        row2_col1, row2_col2 = st.columns(2)
        
        outcome_cols = [(row1_col1, 'OS'), (row1_col2, 'NRM'), 
                        (row2_col1, 'Relapse'), (row2_col2, 'cGVHD')]
        
        for col, outcome_key in outcome_cols:
            with col:
                outcome_data = summary[outcome_key]
                pred_data = outcome_data['predictions']
                outcome_info = OUTCOMES[outcome_key]
                ensemble_pred = pred_data['ensemble']
                risk_cat = outcome_data['risk_category']
                
                # Color based on outcome type and risk
                if outcome_key == 'OS':
                    # For OS, higher is better
                    if ensemble_pred >= 0.7:
                        color = "#28a745"  # green
                    elif ensemble_pred >= 0.5:
                        color = "#ffc107"  # yellow
                    else:
                        color = "#dc3545"  # red
                else:
                    # For adverse outcomes, lower is better
                    if risk_cat == 'Low':
                        color = "#28a745"
                    elif risk_cat == 'Intermediate':
                        color = "#ffc107"
                    else:
                        color = "#dc3545"
                
                st.markdown(f"""
                <div class="result-card">
                    <h4>{outcome_info['name']}</h4>
                    <div class="result-value" style="color: {color};">{ensemble_pred*100:.1f}%</div>
                    <div class="result-label" style="color: {color};">{risk_cat}</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Survival/CIF Curves
        st.markdown('<div class="section-header">Outcome Trajectories</div>', 
                    unsafe_allow_html=True)
        
        # Create chart with dark background for contrast
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Overall Survival', 'Cumulative Incidence'),
            horizontal_spacing=0.15
        )
        
        # OS curve - bright colors on dark background
        os_curves = summary['OS']['curves']
        fig.add_trace(
            go.Scatter(
                x=TIME_POINTS, y=[v*100 for v in os_curves['survival']],
                mode='lines+markers', 
                name='Survival',
                line=dict(color='#60a5fa', width=3),  # bright blue
                marker=dict(size=7, color='#60a5fa'),
                hovertemplate='%{y:.1f}% at %{x} months<extra></extra>'
            ),
            row=1, col=1
        )
        
        # CIF curves - bright colors for dark background
        colors = {'NRM': '#f87171', 'Relapse': '#fb923c', 'cGVHD': '#4ade80'}  # bright red, orange, green
        for outcome in ['NRM', 'Relapse', 'cGVHD']:
            curves = summary[outcome]['curves']
            fig.add_trace(
                go.Scatter(
                    x=TIME_POINTS, y=[v*100 for v in curves['cif']],
                    mode='lines+markers', 
                    name=OUTCOMES[outcome]['name'],
                    line=dict(color=colors[outcome], width=2.5),
                    marker=dict(size=6, color=colors[outcome]),
                    hovertemplate='%{y:.1f}% at %{x} months<extra></extra>'
                ),
                row=1, col=2
            )
        
        # Dark background layout
        fig.update_layout(
            height=350,
            margin=dict(l=60, r=40, t=50, b=70),
            legend=dict(
                orientation='h', 
                yanchor='bottom', 
                y=-0.25, 
                xanchor='center', 
                x=0.5,
                font=dict(size=12, color='#e2e8f0'),
                bgcolor='rgba(30, 41, 59, 0.9)'
            ),
            font=dict(size=12, color='#e2e8f0'),
            paper_bgcolor='#1e293b',  # dark slate
            plot_bgcolor='#334155'     # slightly lighter slate
        )
        
        # Update subplot titles - white text
        for annotation in fig.layout.annotations:
            annotation.font.color = '#f1f5f9'
            annotation.font.size = 14
        
        # Update axes - light text on dark background
        fig.update_xaxes(
            title_text='Months', 
            range=[0, 36], 
            title_font=dict(color='#e2e8f0', size=12),
            tickfont=dict(color='#cbd5e1', size=11),
            gridcolor='#475569',
            linecolor='#64748b',
            showline=True,
            zeroline=False
        )
        fig.update_yaxes(
            title_text='Probability (%)', 
            range=[0, 100], 
            row=1, col=1,
            title_font=dict(color='#e2e8f0', size=12),
            tickfont=dict(color='#cbd5e1', size=11),
            gridcolor='#475569',
            linecolor='#64748b',
            showline=True,
            zeroline=False
        )
        fig.update_yaxes(
            title_text='Cumulative Incidence (%)', 
            range=[0, 100], 
            row=1, col=2,
            title_font=dict(color='#e2e8f0', size=12),
            tickfont=dict(color='#cbd5e1', size=11),
            gridcolor='#475569',
            linecolor='#64748b',
            showline=True,
            zeroline=False
        )
        
        st.plotly_chart(fig, use_container_width=True, key="outcome_trajectories")
        
        # Model breakdown
        with st.expander("View Model Breakdown"):
            model_data = []
            for outcome_key in ['OS', 'NRM', 'Relapse', 'cGVHD']:
                pred = summary[outcome_key]['predictions']
                
                # Get XGBoost model type if available
                xgb_type = "Cox"
                if MODEL_CONFIG and outcome_key in MODEL_CONFIG:
                    model_type = MODEL_CONFIG[outcome_key].get('model_type', 'cox')
                    if model_type == 'aft':
                        xgb_type = "AFT"
                    elif model_type == 'fine_gray':
                        xgb_type = "Fine-Gray"
                
                model_data.append({
                    'Outcome': OUTCOMES[outcome_key]['name'],
                    'Cox PH': f"{pred['cox']*100:.1f}%",
                    'RSF': f"{pred['rsf']*100:.1f}%",
                    f'XGBoost ({xgb_type})': f"{pred['xgboost']*100:.1f}%",
                    'Ensemble': f"{pred['ensemble']*100:.1f}%"
                })
            
            st.dataframe(
                pd.DataFrame(model_data),
                use_container_width=True,
                hide_index=True
            )
            
            # Show model info
            if USING_TRAINED_MODELS:
                st.markdown("""
                <div style="background-color: #ecfdf5; border: 1px solid #10b981; border-radius: 6px; padding: 0.75rem; margin-top: 0.5rem;">
                    <p style="color: #065f46; margin: 0; font-size: 0.85rem;">
                        <strong>✓ Using Calibrated Trained Models</strong> - XGBoost predictions use Kaplan-Meier adjusted baselines from training data.
                    </p>
                </div>
                """, unsafe_allow_html=True)
        
        # Methodology Documentation Section
        with st.expander("📋 View Ensemble Methodology"):
            st.markdown("#### Ensemble Model Architecture")
            st.write("This tool combines three complementary statistical/machine learning approaches to predict 36-month outcomes:")
            
            st.markdown("**1. Cox Proportional Hazards (Cox PH)**")
            st.markdown("""
            - Traditional semi-parametric survival model
            - Interpretable hazard ratios for each covariate
            - Validated using IPCW C-index on 20% test set
            """)
            
            st.markdown("**2. Random Survival Forest (RSF)**")
            st.markdown("""
            - Ensemble of survival trees capturing non-linear effects
            - Handles variable interactions automatically
            - Native IPCW C-index from randomForestSRC
            """)
            
            st.markdown("**3. XGBoost (Gradient Boosting)**")
            st.markdown("""
            - **OS & NRM:** XGBoost with `survival:cox` objective
            - **Relapse:** Accelerated Failure Time (AFT) model
            - **cGVHD:** Fine-Gray subdistribution hazard via pseudo-observations
            - Highest discrimination on test set (C-index: 0.82-0.87)
            """)
            
            st.markdown("---")
            st.markdown("#### Calibration Methodology")
            st.info("""
            **XGBoost Cox Models (OS, NRM):**
            - **Centered Predictions:** Raw log-hazard outputs are centered by subtracting training mean
            - **Kaplan-Meier Baselines:** Survival/CIF computed from training cohort (n=3,733)
            - **OS:** S(t) = S₀(36m)^exp(centered_risk), where S₀ = 58.4%
            - **NRM:** CIF(t) = 1 - (1-CIF₀)^exp(centered_risk), where CIF₀ = 26.2%
            """)
            
            st.markdown("---")
            st.markdown("#### Ensemble Weighting")
            st.write("Final predictions are weighted averages based on each model's test set performance:")
            
            weight_data = pd.DataFrame({
                'Outcome': ['Overall Survival', 'Non-Relapse Mortality', 'Relapse', 'Chronic GVHD'],
                'Cox PH': ['30%', '30%', '35%', '30%'],
                'RSF': ['30%', '30%', '30%', '35%'],
                'XGBoost': ['40%', '40%', '35%', '35%']
            })
            st.dataframe(weight_data, use_container_width=True, hide_index=True)
            
            st.markdown("---")
            st.markdown("#### Validation Performance (20% Test Set, n=933)")
            
            perf_data = pd.DataFrame({
                'Outcome': ['Overall Survival', 'Non-Relapse Mortality', 'Relapse', 'Chronic GVHD'],
                'Cox C-index': [0.617, 0.612, 0.597, 0.573],
                'RSF C-index': [0.626, 0.624, 0.598, 0.586],
                'XGBoost C-index': [0.823, 0.874, 0.631, 0.627]
            })
            st.dataframe(perf_data, use_container_width=True, hide_index=True)
            
            st.warning("**Note:** XGBoost demonstrates superior discrimination. The ensemble combines the interpretability of Cox PH, robustness of RSF, and high performance of XGBoost for balanced predictions.")
        
        # SHAP Feature Importance
        if SHAP_IMPORTANCE is not None and len(SHAP_IMPORTANCE) > 0:
            with st.expander("View Feature Importance (SHAP)"):
                st.markdown("""
                <div style="background-color: #dbeafe; border: 1px solid #3b82f6; border-radius: 6px; padding: 0.75rem; margin-bottom: 1rem;">
                    <p style="color: #1e3a8a; margin: 0; font-size: 0.9rem;">
                        <strong>SHAP (SHapley Additive exPlanations)</strong> values show the contribution of each feature to model predictions. 
                        Higher values indicate greater importance. Computed using out-of-fold cross-validation for unbiased estimates.
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Create tabs for each outcome
                shap_tabs = st.tabs(['Overall Survival', 'NRM', 'Relapse', 'Chronic GVHD'])
                
                for idx, (tab, outcome_key) in enumerate(zip(shap_tabs, ['OS', 'NRM', 'Relapse', 'cGVHD'])):
                    with tab:
                        outcome_shap = SHAP_IMPORTANCE[SHAP_IMPORTANCE['outcome'] == outcome_key].copy()
                        if len(outcome_shap) > 0:
                            top_features = outcome_shap.nlargest(10, 'mean_abs_shap')
                            
                            # Create horizontal bar chart
                            fig = go.Figure(go.Bar(
                                x=top_features['mean_abs_shap'].values[::-1],
                                y=top_features['feature'].values[::-1],
                                orientation='h',
                                marker_color='#3b82f6'
                            ))
                            
                            fig.update_layout(
                                title=f'Top 10 Features - {OUTCOMES[outcome_key]["name"]}',
                                xaxis_title='Mean |SHAP Value|',
                                yaxis_title='',
                                height=350,
                                margin=dict(l=200, r=20, t=40, b=40),
                                paper_bgcolor='#ffffff',
                                plot_bgcolor='#f8fafc',
                                font=dict(color='#1e293b')
                            )
                            
                            st.plotly_chart(fig, use_container_width=True, key=f"shap_{outcome_key}")
                        else:
                            st.info(f"SHAP data not available for {OUTCOMES[outcome_key]['name']}")
    
    # =========================================================================
    # ADJUSTABLE COVARIATES EFFECTS TABLE
    # =========================================================================
    st.markdown("---")
    st.markdown('<div class="section-header">Adjustable Covariate Effects* (vs Reference)</div>', 
                unsafe_allow_html=True)
    
    st.markdown('<div style="background-color: #dbeafe; border: 2px solid #3b82f6; border-radius: 8px; padding: 1rem; margin: 0.5rem 0 1rem 0;"><p style="color: #1e3a8a; margin: 0; font-size: 0.95rem; line-height: 1.6;"><strong style="color: #1e3a8a;">*Adjustable covariates</strong> are treatment decisions that may be modified. Effects show the change in predicted probability compared to the reference category.<br><strong>↑</strong> = increased | <strong>↓</strong> = decreased | <span style="color: #166534; font-weight: 700;">Green = favorable</span> | <span style="color: #dc2626; font-weight: 700;">Red = unfavorable</span></p></div>', unsafe_allow_html=True)
    
    # Calculate covariate effects
    effects_table = calculate_covariate_effects_table(validated_data)
    
    # Build HTML table with strong contrast
    table_html = '<table style="width: 100%; border-collapse: collapse; background-color: #ffffff; margin-top: 1rem; font-size: 0.95rem;">'
    table_html += '<thead><tr style="background-color: #1e3a8a;">'
    table_html += '<th style="padding: 12px 10px; text-align: left; color: #ffffff; font-weight: 600; border: 1px solid #1e40af;">Covariate</th>'
    table_html += '<th style="padding: 12px 10px; text-align: left; color: #ffffff; font-weight: 600; border: 1px solid #1e40af;">Selected vs Reference</th>'
    table_html += '<th style="padding: 12px 10px; text-align: center; color: #ffffff; font-weight: 600; border: 1px solid #1e40af;">OS</th>'
    table_html += '<th style="padding: 12px 10px; text-align: center; color: #ffffff; font-weight: 600; border: 1px solid #1e40af;">NRM</th>'
    table_html += '<th style="padding: 12px 10px; text-align: center; color: #ffffff; font-weight: 600; border: 1px solid #1e40af;">Relapse</th>'
    table_html += '<th style="padding: 12px 10px; text-align: center; color: #ffffff; font-weight: 600; border: 1px solid #1e40af;">cGVHD</th>'
    table_html += '</tr></thead><tbody>'
    
    for i, row in enumerate(effects_table):
        row_bg = '#f8fafc' if i % 2 == 0 else '#ffffff'
        
        if row['is_reference']:
            selected_display = f"{row['selected']} <span style='color: #6b7280;'>(Reference)</span>"
        else:
            selected_display = f"{row['selected']} <span style='color: #6b7280;'>vs {row['reference']}</span>"
        
        table_html += f'<tr style="background-color: {row_bg};">'
        table_html += f'<td style="padding: 10px; border: 1px solid #e5e7eb; color: #1e293b; font-weight: 600;">{row["label"]}</td>'
        table_html += f'<td style="padding: 10px; border: 1px solid #e5e7eb; color: #1e293b;">{selected_display}</td>'
        
        for outcome in ['OS', 'NRM', 'Relapse', 'cGVHD']:
            effect = row['effects'][outcome]
            if row['is_reference']:
                cell_content = '<span style="color: #9ca3af;">–</span>'
            else:
                diff_pct = effect['diff_pct']
                direction = effect['direction']
                
                if abs(diff_pct) < 1:
                    cell_content = '<span style="color: #9ca3af;">–</span>'
                else:
                    sign = '+' if diff_pct > 0 else ''
                    
                    # Determine color based on outcome and direction
                    if outcome == 'OS':
                        # For OS, increase is good (green), decrease is bad (red)
                        if diff_pct > 0:
                            color = '#166534'  # dark green
                            bg = '#dcfce7'  # light green bg
                        else:
                            color = '#dc2626'  # red
                            bg = '#fee2e2'  # light red bg
                    else:
                        # For adverse outcomes, decrease is good (green), increase is bad (red)
                        if diff_pct < 0:
                            color = '#166534'  # dark green
                            bg = '#dcfce7'  # light green bg
                        else:
                            color = '#dc2626'  # red
                            bg = '#fee2e2'  # light red bg
                    
                    cell_content = f'<span style="color: {color}; font-weight: 700; background-color: {bg}; padding: 2px 6px; border-radius: 4px;">{direction} {sign}{diff_pct:.1f}%</span>'
            
            table_html += f'<td style="padding: 10px; border: 1px solid #e5e7eb; text-align: center;">{cell_content}</td>'
        
        table_html += "</tr>"
    
    table_html += '</tbody></table>'
    
    st.markdown(table_html, unsafe_allow_html=True)
    
    # =========================================================================
    # WHAT-IF COMPARISON SECTION
    # =========================================================================
    st.markdown("---")
    st.markdown('<div class="section-header">What-If Scenario Comparison</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    Compare how different treatment approaches might affect predicted outcomes. 
    Select an alternative scenario below to see side-by-side predictions.
    </div>
    """, unsafe_allow_html=True)
    
    col_scenario, col_comparison = st.columns([1, 2])
    
    with col_scenario:
        st.markdown("**Select Alternative Approach:**")
        
        # Key modifiable factors
        alt_donor = st.selectbox(
            "Alternative Donor Type",
            options=FEATURE_OPTIONS['Donor Type'],
            index=FEATURE_OPTIONS['Donor Type'].index(validated_data['Donor Type']),
            key='alt_donor'
        )
        
        alt_conditioning = st.selectbox(
            "Alternative Conditioning",
            options=FEATURE_OPTIONS['Conditioning Regimen'],
            index=FEATURE_OPTIONS['Conditioning Regimen'].index(validated_data['Conditioning Regimen']),
            key='alt_conditioning'
        )
        
        alt_gvhd = st.selectbox(
            "Alternative GVHD Prophylaxis",
            options=FEATURE_OPTIONS['GVHD Prophylaxis'],
            index=FEATURE_OPTIONS['GVHD Prophylaxis'].index(validated_data['GVHD Prophylaxis']),
            key='alt_gvhd'
        )
        
        alt_tcd = st.selectbox(
            "Alternative T-cell Depletion",
            options=FEATURE_OPTIONS['In Vivo T-cell Depletion (Yes)'],
            index=FEATURE_OPTIONS['In Vivo T-cell Depletion (Yes)'].index(
                validated_data['In Vivo T-cell Depletion (Yes)']
            ),
            key='alt_tcd'
        )
        
        # Create alternative scenario
        alt_scenario = validated_data.copy()
        alt_scenario['Donor Type'] = alt_donor
        alt_scenario['Conditioning Regimen'] = alt_conditioning
        alt_scenario['GVHD Prophylaxis'] = alt_gvhd
        alt_scenario['In Vivo T-cell Depletion (Yes)'] = alt_tcd
    
    with col_comparison:
        # Check if scenarios differ
        differences = get_scenario_differences(validated_data, alt_scenario)
        
        if differences:
            st.markdown('<p style="color: #1e293b; font-weight: 600; font-size: 1rem; margin-bottom: 0.5rem;">Changes from Current:</p>', unsafe_allow_html=True)
            changes_html = ""
            for feature, vals in differences.items():
                changes_html += f'<p style="color: #1e293b; margin: 0.25rem 0;">• <strong>{feature}:</strong> {vals["scenario_a"]} → <span style="color: #2563eb; font-weight: 600;">{vals["scenario_b"]}</span></p>'
            st.markdown(changes_html, unsafe_allow_html=True)
            
            # Calculate comparison
            comparison = compare_scenarios(validated_data, alt_scenario)
            
            # Comparison visualization with dark background
            fig_comp = go.Figure()
            
            outcomes_list = ['OS', 'NRM', 'Relapse', 'cGVHD']
            outcome_names = [OUTCOMES[o]['name'] for o in outcomes_list]
            
            current_vals = [comparison[o]['scenario_a']['ensemble']*100 for o in outcomes_list]
            alt_vals = [comparison[o]['scenario_b']['ensemble']*100 for o in outcomes_list]
            
            fig_comp.add_trace(go.Bar(
                name='Current',
                x=outcome_names,
                y=current_vals,
                marker_color='#60a5fa',  # bright blue
                text=[f"{v:.1f}%" for v in current_vals],
                textposition='outside',
                textfont=dict(color='#e2e8f0', size=12)
            ))
            
            fig_comp.add_trace(go.Bar(
                name='Alternative',
                x=outcome_names,
                y=alt_vals,
                marker_color='#fb923c',  # bright orange
                text=[f"{v:.1f}%" for v in alt_vals],
                textposition='outside',
                textfont=dict(color='#e2e8f0', size=12)
            ))
            
            fig_comp.update_layout(
                barmode='group',
                height=350,
                margin=dict(l=60, r=30, t=50, b=80),
                legend=dict(
                    orientation='h', 
                    yanchor='bottom', 
                    y=1.05, 
                    xanchor='center', 
                    x=0.5,
                    font=dict(color='#e2e8f0', size=12),
                    bgcolor='rgba(30, 41, 59, 0.9)'
                ),
                yaxis_title='Probability (%)',
                yaxis=dict(
                    range=[0, max(max(current_vals), max(alt_vals)) * 1.3],
                    title_font=dict(color='#e2e8f0', size=12),
                    tickfont=dict(color='#cbd5e1', size=11),
                    gridcolor='#475569',
                    linecolor='#64748b',
                    showline=True,
                    zeroline=False
                ),
                xaxis=dict(
                    tickfont=dict(color='#e2e8f0', size=11),
                    linecolor='#64748b',
                    showline=True
                ),
                font=dict(color='#e2e8f0', size=12),
                paper_bgcolor='#1e293b',  # dark slate
                plot_bgcolor='#334155'     # slightly lighter slate
            )
            
            st.plotly_chart(fig_comp, use_container_width=True, key="whatif_comparison")
            
            # Summary of impact
            favorable_count = sum(1 for o in outcomes_list if comparison[o]['improved'])
            
            if favorable_count >= 3:
                st.success(f"The alternative approach may be favorable ({favorable_count}/4 outcomes improved).")
            elif favorable_count >= 2:
                st.info(f"Mixed results ({favorable_count}/4 outcomes improved). Consider clinical priorities.")
            elif favorable_count >= 1:
                st.warning(f"Limited improvement ({favorable_count}/4 outcomes). Current approach may be preferable.")
            else:
                st.error("Alternative approach shows no improvement over current plan.")
        
        else:
            st.markdown("""
            <div class="info-box">
            ℹ️ Modify the alternative scenario options on the left to compare different treatment approaches.
            </div>
            """, unsafe_allow_html=True)
    
    # =========================================================================
    # DISCLAIMER
    # =========================================================================
    st.markdown("""
    <div class="disclaimer">
        <div class="disclaimer-title">⚠️ Important Disclaimer</div>
        <p>This tool provides <strong>estimates only</strong> based on statistical models trained on historical data. 
        Predictions are not guarantees of outcomes and should not replace clinical judgment.</p>
        <ul style="margin: 0.5rem 0; padding-left: 1.5rem;">
            <li>Individual patient outcomes may vary significantly from predictions</li>
            <li>Models have inherent limitations and uncertainty</li>
            <li>Clinical decisions should incorporate factors beyond this tool</li>
            <li>Validated for adult patients with ALL only</li>
        </ul>
        <p style="margin-bottom: 0;"><strong>For research and educational purposes only.</strong> 
        Always consult with the transplant team for clinical decision-making.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="footer-text">
        <strong>HCT Outcomes Ensemble Predictor</strong><br>
        Models: Cox Proportional Hazards | Random Survival Forest | XGBoost
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
