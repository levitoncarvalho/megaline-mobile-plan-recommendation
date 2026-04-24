# Streamlit app for Megaline plan recommendation

import streamlit as st
import pandas as pd
from src.predict import load_model, make_prediction

# Page Config
st.set_page_config(
    page_title="Megaline Plan Recommender",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for UI enhancement
st.markdown("""
    <style>
    /* Main button styling */
    div.stButton > button:first-child {
        background-color: #0071e3;
        color: white;
        border-radius: 12px;
        border: none;
        padding: 10px 24px;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        background-color: #005bb5;
        box-shadow: 0 4px 12px rgba(0, 113, 227, 0.4);
        transform: translateY(-2px);
    }
    /* Smooth alert boxes */
    div.stAlert {
        border-radius: 12px;
    }
    /* Title styling */
    h1 {
        font-family: -apple-system, BlinkMacSystemFont, sans-serif;
        font-weight: 700;
        letter-spacing: -1px;
    }
    </style>
""", unsafe_allow_html=True)

# Load Model
@st.cache_resource
def get_model():
    return load_model()

model = get_model()

# User input fields (Sidebar)
with st.sidebar:
    st.image("https://img.icons8.com/ios-filled/100/0071e3/smartphone-tablet.png", width=60)
    st.header("User Profile")
    st.caption("Adjust the sliders to simulate customer behavior.")
    st.markdown("---")
    
    calls = st.slider("Calls", min_value=0.0, max_value=250.0, value=60.0, step=1.0)
    minutes = st.slider("Call Minutes", min_value=0.0, max_value=1500.0, value=400.0, step=1.0)
    messages = st.slider("Text Messages", min_value=0.0, max_value=1500.0, value=400.0, step=1.0)
    mb_used = st.number_input("Data Used (MB)", min_value=0.0, max_value=50000.0, value=15000.0, step=100.0)
    
    st.markdown("---")
    predict_button = st.button("Recommend Plan 🚀")

# Main page content
st.title("📱 Megaline – Plan Recommendation")
st.markdown("Enter the customer's usage data to receive a plan recommendation: **Smart** or **Ultra**.")

# Empty state before prediction
if not predict_button:
    st.info("👈 Enter the customer's usage metrics in the sidebar and click **Recommend Plan** to see the results.")

# Trigger prediction
if predict_button:
    input_data = pd.DataFrame(
        [[calls, minutes, messages, mb_used]],
        columns=['calls', 'minutes', 'messages', 'mb_used']
    )
    
    prediction = make_prediction(model, input_data)
    plan = "Ultra" if prediction == 1 else "Smart"
    
    st.markdown("### Analysis Result")
    
    # Results dashboard layout
    res_col1, res_col2 = st.columns([1, 2])
    
    with res_col1:
        if plan == "Ultra":
            st.metric(label="Recommended Plan", value="⚡ ULTRA")
        else:
            st.metric(label="Recommended Plan", value="💡 SMART")
            
    with res_col2:
        if plan == "Ultra":
            st.success("**Why Ultra?** The customer benefits more from the Ultra plan, with more data and minutes to enjoy without worries.")
        else:
            st.success("**Why Smart?** The smart plan is suitable for this customer's profile, making a smart economy with a fair price.")

# Footer
st.markdown("---")
st.caption("Developed by [Leviton Lima Carvalho](https://github.com/levitoncarvalho) as a portfolio project | Model trained on Megaline historical data")