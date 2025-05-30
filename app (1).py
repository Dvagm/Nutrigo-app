import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import joblib
import os

# Load ML model
@st.cache_resource
def load_model():
    model_path = os.path.join("model", "nutrition_model.pkl")
    return joblib.load(model_path)

model = load_model()

# Set page config
st.set_page_config(
    page_title="Nutrigo - Nutrition Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {
        background-color: #f7f8fa;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stSidebar {
        background-color: #ffffff;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🥗 Nutrigo - Nutrition & Diet Tracker")
st.markdown("Minimalist dashboard to visualize and understand your meal's nutritional values.")

# Sidebar for input
st.sidebar.header("🍽️ Enter Meal Details")
st.sidebar.markdown("Fill in the nutritional values of your meal.")

# User Profile Section
st.sidebar.subheader("👤 User Profile")
user_name = st.sidebar.text_input("Name", "John Doe")
age = st.sidebar.slider("Age", 10, 80, 30)
gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
goals = st.sidebar.selectbox("Health Goal", ["Maintain", "Lose Weight", "Gain Muscle", "Improve Health"])

# Meal Input
st.sidebar.subheader("🍲 Meal Information")
meal_name = st.sidebar.text_input("Meal/Food Name", "Grilled Chicken Salad")
calories = st.sidebar.slider("Calories (kcal)", 0, 1000, 350)
protein = st.sidebar.slider("Protein (g)", 0, 100, 30)
carbs = st.sidebar.slider("Carbohydrates (g)", 0, 100, 20)
fat = st.sidebar.slider("Fat (g)", 0, 100, 15)
fiber = st.sidebar.slider("Fiber (g)", 0, 50, 8)
sugar = st.sidebar.slider("Sugars (g)", 0, 50, 5)
sodium = st.sidebar.slider("Sodium (mg)", 0, 1500, 500)
chol = st.sidebar.slider("Cholesterol (mg)", 0, 300, 75)
water = st.sidebar.slider("Water Intake (ml)", 0, 1000, 500)

# DataFrame from input
data = pd.DataFrame({
    "Nutrient": ["Calories", "Protein", "Carbs", "Fat", "Fiber", "Sugar", "Sodium", "Cholesterol", "Water"],
    "Value": [calories, protein, carbs, fat, fiber, sugar, sodium, chol, water],
    "Unit": ["kcal", "g", "g", "g", "g", "g", "mg", "mg", "ml"]
})

# ML Model Prediction
X_input = np.array([[calories, protein, carbs, fat, fiber, sugar, sodium, chol, water]])
predicted_score = model.predict(X_input)[0]
score_text = "✅ Balanced" if predicted_score >= 30 else "⚠️ Needs Improvement"

# Layout
st.subheader(f"Nutrition Overview for: {meal_name}")
col1, col2 = st.columns([1, 2])

with col1:
    st.metric("Health Score (ML)", f"{predicted_score:.2f}", score_text)
    st.markdown(f"**Name:** {user_name}  ")
    st.markdown(f"**Age:** {age} | **Gender:** {gender}  ")
    st.markdown(f"**Goal:** {goals}")
    st.dataframe(data.set_index("Nutrient"))

with col2:
    fig = px.line_polar(
        data,
        r="Value",
        theta="Nutrient",
        line_close=True,
        title="Radar Chart of Nutrients",
        color_discrete_sequence=["#4CAF50"]
    )
    fig.update_traces(fill='toself')
    fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# Additional visualizations
st.subheader("📈 Additional Visualizations")
col3, col4 = st.columns(2)

with col3:
    fig_bar = px.bar(
        data.sort_values("Value", ascending=False),
        x="Nutrient", y="Value", color="Nutrient",
        title="Bar Chart of Nutritional Content",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with col4:
    fig_pie = px.pie(
        data.query("Nutrient != 'Calories' and Nutrient != 'Water'"),
        names="Nutrient", values="Value",
        title="Macronutrient Composition",
        color_discrete_sequence=px.colors.sequential.RdBu
    )
    st.plotly_chart(fig_pie, use_container_width=True)

# Daily Recommendation Radar
st.subheader("🎯 Nutrient Intake vs Daily Recommendation")
daily_rec = pd.DataFrame({
    "Nutrient": ["Protein", "Carbs", "Fat", "Fiber", "Sugar"],
    "Recommended": [50, 275, 70, 28, 25],
    "Actual": [protein, carbs, fat, fiber, sugar]
})

fig_compare = px.line_polar(
    daily_rec.melt(id_vars="Nutrient", var_name="Type", value_name="Value"),
    r="Value", theta="Nutrient", color="Type", line_close=True,
    title="Actual Intake vs Recommended",
    color_discrete_sequence=["#00bcd4", "#f44336"]
)
fig_compare.update_traces(fill='toself')
fig_compare.update_layout(showlegend=True)
st.plotly_chart(fig_compare, use_container_width=True)

# Footer
st.markdown("""
---
Built with ❤️ by Nutrigo Team
""")
