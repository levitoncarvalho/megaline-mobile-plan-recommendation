# Streamlit app for Megaline plan recommendation

import streamlit as st
import pandas as pd
from src.predict import load_model, make_prediction

#Page Config

st.set_page_config(
    page_title = "Megaline Plan Recommender",
    page_icon = "📱",
    layout = "centered"
)

st.title("📱 Megaline – Plan Recommendation")
st.markdown("Enter the customer's usage data to receive a plan recommendation: **Smart** or **Ultra**.")

# Load Model
@st.cache_resource
def get_model():
    return load_model()

model = get_model()

#User input fields
col1, col2 = st.columns(2)
with col1:
    calls = st.number_input("Calls", min_value = 0.0, max_value = 250.00, value = 60.0, step = 1.0)
    messages = st.number_input("Text Messages", min_value = 0.0, max_value = 1500.0, value = 400.0, step = 1.0)
    minutes = st.number_input("Call Minutes", min_value = 0.0, max_value = 1500.0, value = 400.00, step = 1.0)
    mb_used = st.number_input("Data Used (MB)", min_value = 0.0, max_value = 50000.0, value = 15000.0, step = 100.0)

if st.button("Recommend Plan"):
    input_data = pd.DataFrame(
        [[calls, minutes, messages, mb_used]],
        columns=['calls', 'minutes', 'messages', 'mb_used']
    )
    prediction = make_prediction(model, input_data)
    plan = "Ultra" if prediction == 1 else "Smart"
    st.success(f"Recommended plan: **{plan}**")
    if plan == "Ultra":
        st.info("The customer benefits more from the Ultra plan, with more data and minutes to enjoy without no worries.")
    else:
        st.info("The smart plan is suitable for this customer's profile, making the smart economy with a fair price.")