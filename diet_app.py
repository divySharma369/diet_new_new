import streamlit as st
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
import random
df_core = pd.read_csv("nutrition.csv")


st.set_page_config(page_title="AI Diet Recommendation System", layout="centered")

st.title("🥗 AI-Powered Diet Recommendation System")

st.sidebar.header("User Profile")
age = st.sidebar.number_input("Age", 10, 100, 25)
gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
height = st.sidebar.number_input("Height (cm)", 100, 220, 170)
weight = st.sidebar.number_input("Weight (kg)", 30, 200, 70)
activity = st.sidebar.selectbox("Activity Level", 
    ["Sedentary", "Light", "Moderate", "Active", "Very Active"])
goal = st.sidebar.selectbox("Goal", 
    ["muscle_gain", "weight_loss", "fiber_rich", "balanced"])

# --- Calorie Estimation (Mifflin–St Jeor) ---
if gender == "Male":
    bmr = 10 * weight + 6.25 * height - 5 * age + 5
else:
    bmr = 10 * weight + 6.25 * height - 5 * age - 161

activity_factor = {
    "Sedentary": 1.2, "Light": 1.375, "Moderate": 1.55, 
    "Active": 1.725, "Very Active": 1.9
}[activity]

calories_needed = bmr * activity_factor
st.metric("Estimated Daily Calorie Need", f"{int(calories_needed)} kcal")

# --- Recommendation section ---
st.subheader("Your Personalized Diet Recommendations")

def recommend_food(goal, top_n=5):
    goal_map = {
        'muscle_gain': 'High-Protein',
        'weight_loss': 'Low-Carb',
        'fiber_rich': 'High-Fiber',
        'balanced': 'Balanced'
    }
    target_category = goal_map.get(goal, 'Balanced')
    subset = df_core[df_core['category'] == target_category]
    if len(subset) == 0:
        return pd.DataFrame({'name': ['No foods found for this goal.']})
    return subset.sample(n=min(top_n, len(subset)))[
        ['name','calories','protein','fat','carbohydrate','fiber','sugars','category']
    ]

if st.button("Generate Diet Plan"):
    recs = recommend_food(goal, top_n=5)
    st.dataframe(recs.reset_index(drop=True))
    st.success("✅ Diet plan generated successfully!")

st.markdown("---")
st.caption("Built with 💚 PyTorch + Streamlit on Kaggle Cloud CPU")
