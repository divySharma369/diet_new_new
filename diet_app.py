"""
import streamlit as st
import pandas as pd
import numpy as np
import torch
import google.generativeai as genai
import time
import json
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="AI Diet Recommendation + Gemini", layout="wide")
st.title("🥗 AI Diet Recommendation System — Gemini Edition")

@st.cache_data
def load_data(path="nutrition.csv"):
    return pd.read_csv(path)

df_core = load_data()

model = None
try:
    class DummyModel: pass
except Exception:
    model = None

GEN_AVAILABLE = False
if "GEMINI_API_KEY" in st.secrets:
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        GEN_AVAILABLE = True
    except Exception:
        GEN_AVAILABLE = False

with st.sidebar:
    st.header("User Profile")
    age = st.number_input("Age", value=25, min_value=10, max_value=120)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    height = st.number_input("Height (cm)", value=170, min_value=100, max_value=250)
    weight = st.number_input("Weight (kg)", value=70, min_value=20, max_value=300)
    activity = st.selectbox("Activity Level", ["Sedentary","Light","Moderate","Active","Very Active"])
    goal = st.selectbox("Goal", ["muscle_gain", "weight_loss", "fiber_rich", "balanced"])
    allergies = st.text_input("Allergies / Avoid (comma separated)", "")
    cuisine_pref = st.selectbox("Cuisine preference", ["Any","Indian","Mediterranean","Western","Asian","Low-cost"])
    budget = st.selectbox("Budget", ["Low","Medium","High"])
    tone = st.selectbox("Tone for AI", ["Friendly","Clinical","Encouraging"])
    use_dataset_context = st.checkbox("Include dataset/model suggestions in prompt", value=True)
    st.divider()
    generate_plan = st.button("Generate Diet Plan (dataset model)")
    get_ai = st.button("💬 Get AI Suggestions (Gemini)")

def compute_bmr(age, gender, weight, height):
    if gender == "Male":
        bmr = 10*weight + 6.25*height - 5*age + 5
    else:
        bmr = 10*weight + 6.25*height - 5*age - 161
    return bmr

activity_factor = {"Sedentary":1.2,"Light":1.375,"Moderate":1.55,"Active":1.725,"Very Active":1.9}
bmr = compute_bmr(age, gender, weight, height)
calories_needed = int(bmr * activity_factor.get(activity, 1.2))
st.metric("Estimated Daily Calorie Need", f"{calories_needed} kcal")

def recommend_food(goal, top_n=5):
    goal_map = {
        "muscle_gain": "High-Protein",
        "weight_loss": "Low-Carb",
        "fiber_rich": "High-Fiber",
        "balanced": "Balanced",
    }
    target_category = goal_map.get(goal, "Balanced")
    subset = df_core[df_core.get("category","") == target_category] if "category" in df_core.columns else df_core
    if len(subset) == 0:
        return pd.DataFrame([{"name":"No foods found","calories":0,"protein":0,"fat":0,"carbohydrate":0,"fiber":0}])
    return subset.sample(n=min(top_n, len(subset)))[['name','calories','protein','fat','carbohydrate','fiber','sugars']]

if "ai_chat" not in st.session_state:
    st.session_state.ai_chat = []

def add_to_history(role, text):
    st.session_state.ai_chat.append({"role": role, "text": text, "time": time.time()})

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Quick Recommendations")
    if generate_plan:
        recs = recommend_food(goal, top_n=6)
        st.dataframe(recs.reset_index(drop=True))
        add_to_history("system", f"Generated dataset recommendations for goal: {goal}")
    else:
        st.info("Click 'Generate Diet Plan' to show dataset-backed suggestions.")
    st.divider()
    st.subheader("Saved Chat")
    if st.session_state.ai_chat:
        for msg in reversed(st.session_state.ai_chat[-6:]):
            st.markdown(f"**{msg['role'].title()}:** {msg['text']}")
    else:
        st.write("No AI suggestions yet. Click 'Get AI Suggestions'.")

with col2:
    st.subheader("AI Nutritionist — Gemini")
    recs_table_str = ""
    if use_dataset_context and 'recs' in locals():
        recs_table_str = recs.to_string(index=False)
    base_context = (
        f"User profile:\\n- Age: {age}\\n- Gender: {gender}\\n- Height: {height} cm\\n- Weight: {weight} kg\\n"
        f"- Activity: {activity}\\n- Goal: {goal.replace('_',' ')}\\n- Target Calories: {calories_needed} kcal/day\\n"
        f"- Allergies: {allergies}\\n- Cuisine preference: {cuisine_pref}\\n- Budget: {budget}\\n"
    )
    system_prompt = (
        "You are a professional, evidence-based nutrition coach. Provide practical, realistic, non-medical dietary guidance. "
        "Do not give medical diagnoses. Focus on food choices, portions, simple recipes, and macro targets."
    )
    def build_user_prompt():
        p = base_context
        if use_dataset_context and recs_table_str:
            p += "\\nDataset/model suggested foods:\\n" + recs_table_str + "\\n"
        p += (
            "\\nTask: Create a 1-day meal plan (breakfast, lunch, dinner, snacks) matching the calorie target, respecting allergies and budget. "
            "Include portions, macros, and brief preparation notes. Keep it concise and in the selected tone."
        )
        p += f"\\nTone: {tone}. Return JSON with keys: meals, total_calories, macros, notes."
        return p

    prompt_text = build_user_prompt()
    with st.expander("Preview Gemini prompt"):
        st.code(prompt_text, language="text")

    if get_ai:
        if not GEN_AVAILABLE:
            st.error("Gemini API key not configured in Streamlit secrets.")
        else:
            add_to_history("user", "Requested AI Suggestions")
            with st.spinner("Gemini is generating suggestions..."):
                try:
                    model = genai.GenerativeModel("gemini-1.5-flash")
                    resp = model.generate_content(prompt_text)
                    response_text = getattr(resp, "text", str(resp))
                    st.markdown("<div style='background:#f9f9f9;border-radius:10px;padding:12px;'>"
                                f"<b>Gemini's Response:</b><br>{response_text}</div>", unsafe_allow_html=True)
                    add_to_history("assistant", response_text)
                except Exception as e:
                    st.error(f"Gemini call failed: {e}")
                    add_to_history("assistant", f"Error: {e}")

st.markdown("---")
st.caption("Built with ❤️ using PyTorch + Kaggle dataset + Gemini API")
"""
