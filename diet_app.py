import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import json

# ML libs
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Optional Gemini
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except Exception:
    GENAI_AVAILABLE = False

# ----------------- Page config -----------------
st.set_page_config(page_title="AI Diet Recommendation System (Gemini + Model)", layout="wide")
st.markdown("<h1 style='text-align:center;color:#2E8B57;'>🥗 AI Diet Recommendation System</h1>", unsafe_allow_html=True)

# ----------------- Load data -----------------
@st.cache_data
def load_nutrition(path="nutrition.csv"):
    if not os.path.exists(path):
        st.error(f"Required file '{path}' not found. Upload it to the app folder.")
        st.stop()
    df = pd.read_csv(path)
    return df

df_core = load_nutrition("nutrition.csv")

# If category column missing, create same simple categories logic used earlier
if "category" not in df_core.columns:
    def food_category(row):
        if row.get('protein',0) >= 20 and row.get('fat',0) <= 15 and row.get('carbohydrate',0) <= 20:
            return 'High-Protein'
        elif row.get('carbohydrate',0) <= 10 and row.get('fat',0) <= 20:
            return 'Low-Carb'
        elif row.get('fiber',0) >= 5:
            return 'High-Fiber'
        else:
            return 'Balanced'
    df_core['category'] = df_core.apply(food_category, axis=1)

# ----------------- Prepare label encoder and scaler -----------------
categorical_labels = sorted(df_core['category'].unique().astype(str).tolist())
le = LabelEncoder().fit(categorical_labels)
# prepare features matrix used for training previously
FEATURE_COLS = ['calories', 'protein', 'fat', 'carbohydrate', 'fiber', 'sugars']
X_all = df_core[FEATURE_COLS].fillna(0).values
scaler = StandardScaler().fit(X_all)

# ----------------- Define the neural network (same architecture used earlier) -----------------
class DietNet(nn.Module):
    def __init__(self, input_size, hidden1=64, hidden2=32, output_size=4):
        super(DietNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.1)
        self.output = nn.Linear(hidden2, output_size)
    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu2(self.fc2(x))
        x = self.dropout2(x)
        return self.output(x)

# ----------------- Try to load model.pth if present -----------------
model = None
MODEL_LOADED = False
model_path = "model.pth"
if os.path.exists(model_path):
    try:
        input_size = len(FEATURE_COLS)
        output_size = len(le.classes_)
        model = DietNet(input_size, 64, 32, output_size)
        state = torch.load(model_path, map_location="cpu")
        # if user saved only state_dict, load; else try to load whole object safely
        if isinstance(state, dict):
            model.load_state_dict(state)
        else:
            # unexpected but attempt
            model = state
        model.eval()
        MODEL_LOADED = True
    except Exception as e:
        st.warning("Model file found but failed to load: " + str(e))
        MODEL_LOADED = False

# ----------------- Helper: recommend by dataset category -----------------
def recommend_by_category(category, top_n=6):
    subset = df_core[df_core['category'] == category]
    if subset.empty:
        return pd.DataFrame([{"name":"No foods found","calories":0,"protein":0,"fat":0,"carbohydrate":0,"fiber":0,"sugars":0}])
    return subset.sample(n=min(top_n, len(subset)))[['name','calories','protein','fat','carbohydrate','fiber','sugars','category']]

# ----------------- Helper: model-rank foods for a target category -----------------
def model_rank_for_category(target_category, top_n=6):
    if not MODEL_LOADED or model is None:
        return None
    try:
        X = df_core[FEATURE_COLS].fillna(0).values
        Xs = scaler.transform(X)
        with torch.no_grad():
            logits = model(torch.tensor(Xs, dtype=torch.float32))
            probs = F.softmax(logits, dim=1).numpy()
        target_idx = int(le.transform([target_category])[0])
        scores = probs[:, target_idx]
        df_scores = df_core.copy()
        df_scores['score'] = scores
        df_sorted = df_scores.sort_values('score', ascending=False)
        return df_sorted.head(top_n)[['name','calories','protein','fat','carbohydrate','fiber','sugars','category','score']]
    except Exception:
        return None

# ----------------- Gemini config -----------------
GEN_AVAILABLE = False
if GENAI_AVAILABLE and "GEMINI_API_KEY" in st.secrets:
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        GEN_AVAILABLE = True
    except Exception:
        GEN_AVAILABLE = False

# ----------------- Sidebar inputs -----------------
with st.sidebar:
    st.header("User Profile")
    age = st.number_input("Age", value=25, min_value=10, max_value=120)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    height = st.number_input("Height (cm)", value=170, min_value=100, max_value=250)
    weight = st.number_input("Weight (kg)", value=70, min_value=20, max_value=300)
    activity = st.selectbox("Activity Level", ["Sedentary","Light","Moderate","Active","Very Active"])
    goal = st.selectbox("Goal", ["muscle_gain","weight_loss","fiber_rich","balanced"])
    allergies = st.text_input("Allergies / avoid (comma separated)", "")
    cuisine_pref = st.selectbox("Cuisine preference", ["Any","Indian","Mediterranean","Western","Asian","Low-cost"])
    budget = st.selectbox("Budget", ["Low","Medium","High"])
    tone = st.selectbox("Tone for AI", ["Friendly","Clinical","Encouraging"])
    use_dataset_context = st.checkbox("Include dataset/model suggestions in Gemini prompt", value=True)
    st.markdown("---")
    btn_generate = st.button("Generate Dataset Plan")
    btn_model_rank = st.button("Model: Rank foods for goal")
    btn_ai = st.button("💬 Get AI Suggestions (Gemini)")

# ----------------- Compute calories -----------------
def compute_bmr(age, gender, weight, height):
    if gender == "Male":
        return 10*weight + 6.25*height - 5*age + 5
    return 10*weight + 6.25*height - 5*age - 161

activity_factor = {"Sedentary":1.2,"Light":1.375,"Moderate":1.55,"Active":1.725,"Very Active":1.9}
bmr = compute_bmr(age, gender, weight, height)
calories_needed = int(bmr * activity_factor.get(activity,1.2))
st.metric("Estimated Daily Calorie Need", f"{calories_needed} kcal")

# ----------------- Session state for chat -----------------
if "ai_chat" not in st.session_state:
    st.session_state.ai_chat = []

def add_history(role, text):
    st.session_state.ai_chat.append({"role":role, "text":text, "time":time.time()})

# ----------------- Main layout -----------------
left, right = st.columns([1, 2])

with left:
    st.subheader("Quick dataset recommendations")
    goal_map = {"muscle_gain":"High-Protein","weight_loss":"Low-Carb","fiber_rich":"High-Fiber","balanced":"Balanced"}
    target_cat = goal_map.get(goal, "Balanced")
    if btn_generate:
        recs = recommend_by_category(target_cat, top_n=6)
        st.dataframe(recs.reset_index(drop=True))
        add_history("system", f"Dataset recommendations for {goal}")
    else:
        st.info("Click 'Generate Dataset Plan' to show dataset-based suggestions.")

    st.markdown("---")
    st.subheader("Model ranking (top matches)")
    if MODEL_LOADED:
        if btn_model_rank:
            model_recs = model_rank_for_category(target_cat, top_n=6)
            if model_recs is not None:
                st.dataframe(model_recs.reset_index(drop=True))
                add_history("system", f"Model-ranked recommendations for {goal}")
            else:
                st.warning("Model ranking failed. Check model compatibility.")
        else:
            st.write("Click 'Model: Rank foods for goal' to get model-ranked suggestions.")
    else:
        st.info("No model loaded. Upload model.pth in app folder to enable model ranking.")

    st.markdown("---")
    st.subheader("Chat history")
    if st.session_state.ai_chat:
        for msg in reversed(st.session_state.ai_chat[-8:]):
            role = msg['role'].title()
            st.markdown(f"**{role}:** {msg['text']}")
    else:
        st.write("No AI messages yet.")

with right:
    st.subheader("AI Nutritionist (Gemini)")
    # Build context
    dataset_section = ""
    if use_dataset_context and 'recs' in locals():
        dataset_section = recs.to_string(index=False)
    elif use_dataset_context and MODEL_LOADED and 'model_recs' in locals() and model_recs is not None:
        dataset_section = model_recs.to_string(index=False)

    base_context = (
        f"User profile:\n- Age: {age}\n- Gender: {gender}\n- Height: {height} cm\n- Weight: {weight} kg\n"
        f"- Activity: {activity}\n- Goal: {goal.replace('_',' ')}\n- Target Calories: {calories_needed} kcal/day\n"
        f"- Allergies: {allergies}\n- Cuisine pref: {cuisine_pref}\n- Budget: {budget}\n"
    )

    prompt = base_context
    if use_dataset_context and dataset_section:
        prompt += "\nDataset/model suggested foods (table):\n" + dataset_section + "\n"
    prompt += (
        "Task: Create a 1-day meal plan (breakfast, lunch, dinner, 1-2 snacks) matching the calorie target approximately. "
        "Provide portion sizes (household units), approximate macro totals (protein, carbs, fat in grams), and short prep notes. "
        f"Tone: {tone}. Keep it realistic, non-medical, and concise. Return JSON with keys: meals, total_calories, macros, notes."
    )

    with st.expander("Preview prompt"):
        st.code(prompt, language="text")

    if btn_ai:
    if not GEN_AVAILABLE:
        st.error("Gemini not configured or google-generativeai not installed. Add GEMINI_API_KEY to Streamlit secrets.")
    else:
        add_history("user", "Requested Gemini suggestions")
        with st.spinner("Contacting Gemini..."):
            try:
                # Correct API usage for Gemini v1
                model_handle = genai.GenerativeModel("models/gemini-1.5-flash")
                response = model_handle.generate_content(prompt)
                response_text = response.text.strip() if hasattr(response, "text") else str(response)
                
                try:
                    parsed = json.loads(response_text)
                    st.success("Gemini returned structured plan:")
                    st.code(json.dumps(parsed, indent=2), language="json")
                    add_history("assistant", json.dumps(parsed))
                except Exception:
                    st.markdown(
                        f"<div style='background:#f9f9f9;border-radius:8px;padding:12px;'>{response_text}</div>",
                        unsafe_allow_html=True
                    )
                    add_history("assistant", response_text)
            except Exception as e:
                st.error(f"Gemini call failed: {e}")
                add_history("assistant", f"Gemini call failed: {e}")
    st.markdown("---")
    st.caption("Tip: Use 'Include dataset/model suggestions' to give Gemini concrete foods to include or modify.")

st.markdown("---")
st.caption("Built with PyTorch + Kaggle nutrition dataset + Gemini (optional).")
