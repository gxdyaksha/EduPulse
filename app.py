import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# --- Page Configuration ---
st.set_page_config(
    page_title="EduPulse | Student Analytics & Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling (Dark UI & Metric Cards) ---
st.markdown("""
<style>
    .main { background-color: #0E1117; }
    .metric-card {
        background: linear-gradient(135deg, #1E222D 0%, #262C3A 100%);
        border: 1px solid #333C4E;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #4F8BF9; }
    .metric-title { font-size: 0.9rem; color: #9AA0A6; margin-bottom: 5px; }
    .stButton>button {
        background: linear-gradient(90deg, #4F8BF9, #00C6FF);
        color: white;
        font-weight: 600;
        border: none;
        padding: 10px 24px;
        border-radius: 8px;
        width: 100%;
        transition: 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 12px rgba(79, 139, 249, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# --- Load & Cache Data ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('StudentsPerformance.csv')
    except FileNotFoundError:
        # Fallback synthetic sample if file is missing
        np.random.seed(42)
        n = 1000
        df = pd.DataFrame({
            'gender': np.random.choice(['female', 'male'], n),
            'race/ethnicity': np.random.choice(['group A', 'group B', 'group C', 'group D', 'group E'], n),
            'parental level of education': np.random.choice(["bachelor's degree", 'some college', "master's degree", "associate's degree", 'high school', 'some high school'], n),
            'lunch': np.random.choice(['standard', 'free/reduced'], n),
            'test preparation course': np.random.choice(['none', 'completed'], n),
            'math score': np.random.randint(20, 100, n),
            'reading score': np.random.randint(25, 100, n),
            'writing score': np.random.randint(20, 100, n)
        })
    df['average_score'] = (df['math score'] + df['reading score'] + df['writing score']) / 3.0
    return df

df = load_data()

# --- Model Training Pipeline ---
@st.cache_resource
def train_model(data):
    X = data[['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']]
    y = data['average_score']
    
    cat_features = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
    preprocessor = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(drop='first', sparse_output=False), cat_features)]
    )
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42))
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    metrics = {
        'r2': r2_score(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
    }
    return pipeline, metrics

pipeline, model_metrics = train_model(df)

# --- Sidebar Navigation ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3135/3135810.png", width=70)
st.sidebar.title("EduPulse Analytics")
app_mode = st.sidebar.radio("Navigation Menu", ["Executive Dashboard", "Exploratory Insights", "Live Score Predictor", "Model Diagnostics"])

st.sidebar.divider()
st.sidebar.info("💡 **Tech Stack:** Python 3.10+, Scikit-Learn, Streamlit, Plotly")

# ==========================================
# PAGE 1: EXECUTIVE DASHBOARD
# ==========================================
if app_mode == "Executive Dashboard":
    st.title("🎓 Institutional Overview & Student Metrics")
    st.write("Aggregated visual reporting across demographics, study preparation, and subject-wise benchmarks.")
    
    # KPI Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-title">Total Records</div>
            <div class="metric-value">{len(df):,}</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-title">Overall Mean Score</div>
            <div class="metric-value">{df['average_score'].mean():.1f}%</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        pass_rate = (len(df[df['average_score'] >= 40]) / len(df)) * 100
        st.markdown(f"""<div class="metric-card">
            <div class="metric-title">Pass Rate (≥40%)</div>
            <div class="metric-value">{pass_rate:.1f}%</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-title">Model R² Accuracy</div>
            <div class="metric-value">{model_metrics['r2']*100:.1f}%</div>
        </div>""", unsafe_allow_html=True)

    st.write("")
    
    # Row 2: Distribution & Demographics
    r2_col1, r2_col2 = st.columns([1.2, 1])
    with r2_col1:
        fig_dist = px.histogram(
            df, x="average_score", nbins=30, marginal="box",
            title="Distribution of Overall Student Academic Scores",
            color_discrete_sequence=['#4F8BF9'], template="plotly_dark"
        )
        st.plotly_chart(fig_dist, use_container_width=True)
    with r2_col2:
        parental_counts = df['parental level of education'].value_counts().reset_index()
        fig_pie = px.pie(
            parental_counts, values='count', names='parental level of education',
            title="Cohort by Parental Education Level", hole=0.45,
            color_discrete_sequence=px.colors.sequential.Teal, template="plotly_dark"
        )
        st.plotly_chart(fig_pie, use_container_width=True)

# ==========================================
# PAGE 2: EXPLORATORY INSIGHTS
# ==========================================
elif app_mode == "Exploratory Insights":
    st.title("📊 Deep-Dive Exploratory Data Analysis")
    st.write("Understand how socioeconomic variables correlate with test achievements.")
    
    tab1, tab2 = st.tabs(["Preparation & Nutrition", "Subject Correlations"])
    
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            fig_prep = px.box(
                df, x="test preparation course", y="average_score", color="test preparation course",
                title="Impact of Test Preparation Course", template="plotly_dark",
                color_discrete_sequence=['#FF7F0E', '#2CA02C']
            )
            st.plotly_chart(fig_prep, use_container_width=True)
        with c2:
            fig_lunch = px.violin(
                df, x="lunch", y="average_score", color="gender", box=True,
                title="Score Variation by Lunch Program & Gender", template="plotly_dark"
            )
            st.plotly_chart(fig_lunch, use_container_width=True)
            
    with tab2:
        fig_scatter = px.scatter(
            df, x="reading score", y="writing score", color="math score",
            size="average_score", hover_data=['gender'],
            title="Subject Synergy: Reading vs. Writing vs. Math Performance",
            template="plotly_dark", color_continuous_scale="Plasma"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

# ==========================================
# PAGE 3: LIVE SCORE PREDICTOR
# ==========================================
elif app_mode == "Live Score Predictor":
    st.title("🎯 Machine Learning Student Score Predictor")
    st.write("Enter individual background details to simulate anticipated academic performance.")
    
    with st.form("prediction_form"):
        f_col1, f_col2 = st.columns(2)
        with f_col1:
            gender = st.selectbox("Gender", sorted(df['gender'].unique()))
            race = st.selectbox("Race / Ethnicity Group", sorted(df['race/ethnicity'].unique()))
            parent_edu = st.selectbox("Parental Education Level", sorted(df['parental level of education'].unique()))
        with f_col2:
            lunch = st.selectbox("Lunch Support", sorted(df['lunch'].unique()))
            prep = st.selectbox("Test Preparation Course", sorted(df['test preparation course'].unique()))
            
        submitted = st.form_submit_button("Run Prediction Inference")
        
    if submitted:
        input_data = pd.DataFrame([{
            'gender': gender,
            'race/ethnicity': race,
            'parental level of education': parent_edu,
            'lunch': lunch,
            'test preparation course': prep
        }])
        
        predicted_score = pipeline.predict(input_data)[0]
        
        st.divider()
        res_col1, res_col2 = st.columns([1, 1.5])
        with res_col1:
            st.subheader("Predicted Benchmark")
            st.metric(label="Expected Mean Score", value=f"{predicted_score:.2f} / 100")
            
            if predicted_score >= 75:
                st.success("🟢 Category: Advanced Tier (Distinction)")
            elif predicted_score >= 50:
                st.info("🟡 Category: Intermediate Tier (Pass)")
            else:
                st.error("🔴 Category: At-Risk / Needs Remedial Attention")
                
        with res_col2:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=predicted_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Academic Proficiency Gauge"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#4F8BF9"},
                    'steps': [
                        {'range': [0, 40], 'color': "#3F1818"},
                        {'range': [40, 70], 'color': "#3C3217"},
                        {'range': [70, 100], 'color': "#173C27"}
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 3},
                        'thickness': 0.75,
                        'value': predicted_score
                    }
                }
            ))
            fig_gauge.update_layout(template="plotly_dark", height=280, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

# ==========================================
# PAGE 4: MODEL DIAGNOSTICS
# ==========================================
elif app_mode == "Model Diagnostics":
    st.title("⚙️ Model Pipeline Performance & Validation")
    st.write("Validation metrics evaluated against a 20% randomized unseen test split.")
    
    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("Coefficient of Determination (R²)", f"{model_metrics['r2']:.4f}")
    m_col2.metric("Mean Absolute Error (MAE)", f"{model_metrics['mae']:.2f} marks")
    m_col3.metric("Root Mean Squared Error (RMSE)", f"{model_metrics['rmse']:.2f} marks")
    
    st.divider()
    st.markdown("""
    ### Pipeline Architecture
    1. **Pre-processing Block:** One-Hot Encoding (`drop='first'`) applied on nominal features to prevent multicollinearity.
    2. **Estimator:** Tuned Random Forest Regressor (`n_estimators=150`, `max_depth=10`, random seed locked for reproducibility).
    3. **Target Variable:** Continuous float representing the arithmetic mean across all three subject scores.
    """)