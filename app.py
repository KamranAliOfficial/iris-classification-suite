import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report
import time

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Iris AI Analytics Suite",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
    }
    .metric-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .sidebar-card {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 15px;
        border-left: 4px solid #007bff;
    }
    .footer {
        text-align: center;
        color: #6c757d;
        padding: 20px;
        border-top: 1px solid #dee2e6;
        margin-top: 50px;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 10px 20px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD RESOURCES ---
@st.cache_resource
def load_model_and_data():
    model = joblib.load('iris_model.pkl')
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target
    return model, iris, X, y

try:
    model, iris_data, X, y = load_model_and_data()
    # Calculate model metrics
    train_predictions = model.predict(X)
    accuracy = accuracy_score(y, train_predictions)
    report = classification_report(y, train_predictions, target_names=iris_data.target_names, output_dict=True)
except Exception as e:
    st.error(f"❌ Error loading resources: {e}")
    st.error("Please run 'python train_model.py' to train the model first.")
    st.stop()

# --- HEADER ---
st.markdown("""
    <div class="main-header">
        <h1>🌸 Iris Species Classification Suite</h1>
        <p>Advanced Machine Learning Analytics for Botanical Research</p>
    </div>
    """, unsafe_allow_html=True)

# --- SIDEBAR ---
st.sidebar.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
st.sidebar.header("🔬 Input Parameters")
st.sidebar.markdown("Adjust the flower measurements below:")

# Input controls
sepal_length = st.sidebar.slider('Sepal Length (cm)', 4.3, 7.9, 5.4, 0.1,
                                help="Length of the sepal from base to tip")
sepal_width = st.sidebar.slider('Sepal Width (cm)', 2.0, 4.4, 3.4, 0.1,
                               help="Width of the sepal at its widest point")
petal_length = st.sidebar.slider('Petal Length (cm)', 1.0, 6.9, 1.3, 0.1,
                                help="Length of the petal from base to tip")
petal_width = st.sidebar.slider('Petal Width (cm)', 0.1, 2.5, 0.2, 0.1,
                               help="Width of the petal at its widest point")

st.sidebar.markdown('</div>', unsafe_allow_html=True)

# Reset button
if st.sidebar.button("🔄 Reset to Defaults"):
    st.rerun()

# Create input dataframe
input_data = {
    'sepal length (cm)': sepal_length,
    'sepal width (cm)': sepal_width,
    'petal length (cm)': petal_length,
    'petal width (cm)': petal_width
}
input_df = pd.DataFrame(input_data, index=[0])

# --- PREDICTION SECTION ---
st.header("🎯 Prediction Results")

col1, col2, col3 = st.columns(3)

with st.spinner("Analyzing your input..."):
    time.sleep(0.5)  # Simulate processing
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

species_name = iris_data.target_names[prediction[0]]
confidence = np.max(prediction_proba) * 100

with col1:
    st.markdown(f"""
        <div class="metric-card">
            <h3>Predicted Species</h3>
            <h1>{species_name.upper()}</h1>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
        <div class="metric-card">
            <h3>Confidence Score</h3>
            <h1>{confidence:.1f}%</h1>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
        <div class="metric-card">
            <h3>Model Accuracy</h3>
            <h1>{accuracy*100:.1f}%</h1>
        </div>
    """, unsafe_allow_html=True)

# --- PROBABILITY DISTRIBUTION ---
st.subheader("📊 Class Probabilities")
prob_df = pd.DataFrame(prediction_proba, columns=[name.capitalize() for name in iris_data.target_names])
fig_prob = px.bar(prob_df.T, orientation='h', title="Prediction Probabilities",
                  labels={'value': 'Probability', 'index': 'Species'},
                  color_discrete_sequence=['#667eea', '#764ba2', '#f093fb'])
fig_prob.update_layout(showlegend=False)
st.plotly_chart(fig_prob, use_container_width=True)

# --- VISUALIZATIONS ---
st.header("📈 Data Visualizations")

tab1, tab2, tab3, tab4 = st.tabs(["Feature Radar", "3D Clusters", "Feature Importance", "Model Metrics"])

with tab1:
    st.subheader("📡 Feature Profile")
    categories = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=[input_df[col][0] for col in iris_data.feature_names],
        theta=categories,
        fill='toself',
        name='Your Input',
        line_color='#667eea'
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 8])),
        showlegend=False
    )
    st.plotly_chart(fig_radar, use_container_width=True)

with tab2:
    st.subheader("🌌 3D Feature Space")
    # Prepare data
    X_plot = X.copy()
    X_plot['Species'] = [iris_data.target_names[i] for i in y]
    user_point = input_df.copy()
    user_point['Species'] = 'Your Sample'
    combined_df = pd.concat([X_plot, user_point])

    fig_3d = px.scatter_3d(combined_df, x='sepal length (cm)', y='sepal width (cm)', z='petal length (cm)',
                          color='Species', symbol='Species', size='petal width (cm)',
                          opacity=0.7, title="3D Feature Distribution")
    st.plotly_chart(fig_3d, use_container_width=True)

with tab3:
    st.subheader("🔍 Feature Importance")
    # Get feature importance from the RandomForest classifier
    feature_importance = model.named_steps['classifier'].feature_importances_
    importance_df = pd.DataFrame({
        'Feature': iris_data.feature_names,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=True)

    fig_imp = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                     title="Feature Importance in Prediction",
                     color='Importance', color_continuous_scale='Blues')
    st.plotly_chart(fig_imp, use_container_width=True)

with tab4:
    st.subheader("📋 Model Performance Metrics")
    col_a, col_b = st.columns(2)

    with col_a:
        st.metric("Training Accuracy", f"{accuracy*100:.2f}%")
        st.metric("Precision (Setosa)", f"{report['setosa']['precision']*100:.2f}%")
        st.metric("Recall (Setosa)", f"{report['setosa']['recall']*100:.2f}%")

    with col_b:
        st.metric("F1-Score (Setosa)", f"{report['setosa']['f1-score']*100:.2f}%")
        st.metric("Precision (Macro Avg)", f"{report['macro avg']['precision']*100:.2f}%")
        st.metric("Recall (Macro Avg)", f"{report['macro avg']['recall']*100:.2f}%")

# --- FOOTER ---
st.markdown("""
    <div class="footer">
        <p>Built with ❤️ using Streamlit & Scikit-learn | Iris Dataset Classification Demo</p>
        <p>© 2024 AI Analytics Suite</p>
    </div>
    """, unsafe_allow_html=True)
