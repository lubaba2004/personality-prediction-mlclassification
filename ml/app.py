import warnings
warnings.filterwarnings("ignore")
import streamlit as st
import joblib
import pandas as pd

# ----------- PAGE CONFIG -----------
st.set_page_config(page_title="Personality AI",
                   page_icon="🧠",
                   layout="centered")

# ----------- LOAD MODEL -----------
model = joblib.load("personality.pkl")

# ----------- TITLE -----------
st.markdown("""
<h1 style='text-align: center; color:#FF4B4B;'>
🧠 Personality Prediction System
</h1>
""", unsafe_allow_html=True)

st.caption("Gaussian Naive Bayes Model | Behavioral Analysis")

# ----------- FEATURES -----------
features = [
'social_energy','alone_time_preference','talkativeness','deep_reflection',
'group_comfort','party_liking','listening_skill','empathy','creativity',
'organization','leadership','risk_taking','public_speaking_comfort',
'curiosity','routine_preference','excitement_seeking','friendliness',
'emotional_stability','planning','spontaneity','adventurousness',
'reading_habit','sports_interest','online_social_usage','travel_desire',
'gadget_usage','work_style_collaborative','decision_speed','stress_handling'
]

# ----------- SIDEBAR -----------
st.sidebar.title("📝 Self Rating")

inputs = {}
for col in features:
    nice = col.replace("_"," ").title()
    inputs[col] = st.sidebar.slider(nice, 1, 10, 5)

input_df = pd.DataFrame([inputs])

# ----------- PREDICT -----------
if st.button("✨ Predict Personality"):

    prediction = model.predict(input_df)[0]

    label_map = {
        0: "Introvert",
        1: "Extrovert",
        2: "Ambivert"
    }

    personality = label_map[prediction]

    # -------- RESULT CARD --------
    st.markdown(f"""
    <div style='background:#0E1117;
                padding:20px;
                border-radius:10px;
                border:1px solid #FF4B4B'>
    <h2 style='color:#00FFAA;text-align:center;'>
    {personality}
    </h2>
    </div>
    """, unsafe_allow_html=True)

    # -------- PROBABILITY CHART --------
    prob = model.predict_proba(input_df)[0]

    class_names = ["Introvert", "Extrovert", "Ambivert"]
    prob_df = pd.DataFrame({
        "Type": class_names,
        "Confidence": prob
    })

    st.subheader("Confidence Levels")

    for i in range(len(prob_df)):
        st.progress(float(prob_df["Confidence"][i]))
        st.write(prob_df["Type"][i],
                 round(prob_df["Confidence"][i]*100,2), "%")

    # -------- INTERPRETATION --------
    st.subheader("About You")

    if personality == "Introvert":
        st.info("""
        • Deep thinker  
        • Focused worker  
        • Prefers meaningful conversations  
        • Strong observation skills  
        """)
    
    elif personality == "Extrovert":
        st.info("""
        • Social and energetic  
        • Natural communicator  
        • Team oriented  
        • Leadership potential  
        """)
    
    else:
        st.info("""
        • Balanced personality  
        • Adaptable nature  
        • Both social & focused  
        • Flexible mindset  
        """)

# -------- FOOTER --------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit")
