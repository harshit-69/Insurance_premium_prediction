import streamlit as st
import requests 
import json

st.set_page_config(
    page_title="Insurance Premium Predictor",
    page_icon="💸",
    layout="centered",
    initial_sidebar_state="auto"
)

st.markdown("""
<style>
    /* Change the main page background to a slightly darker light gray for contrast */
    .main {
        font-family: 'Inter', sans-serif;
        background-color: #e6e9f0;
    }
    /* Keep the form background white for high contrast with dark text */
    .stForm {
        padding: 20px;
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        background-color: white; 
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.05);
        /* Fix: Force all text inside the form to be dark for visibility */
        color: #333; 
    }
    /* Fix: Explicitly target Streamlit elements (headers, labels, etc.) inside the form */
    .stForm h3, .stForm label, .stForm div[data-testid*="stText"] {
        color: #333 !important;
    }

    .st-emotion-cache-1ky897g {
        font-size: 1.5em; 
        font-weight: 600;
    }
    /* Button text remains white on teal background */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        background-color: #00A699;
        color: white;
        font-weight: bold;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #007a6d;
    }
</style>
""", unsafe_allow_html=True)


st.title("💸 AI Health Cost Predictor")
st.markdown("""
    <p style='font-size: 18px; color: #555;'>
    Enter policyholder details to estimate annual medical charges and determine the risk tier.
    </p>
""", unsafe_allow_html=True)

st.markdown("---") 

st.header("1. Enter Client Data")

with st.container():
    with st.form("prediction_form"):
        st.subheader("Personal & Health Details (Excluding Region)")
        
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.slider("Age (Years)", 18, 65, 30, key='age_slider')
            bmi = st.number_input("BMI (Body Mass Index)", min_value=15.0, max_value=50.0, value=25.0, step=0.1, key='bmi_input')
            
        with col2:
            sex = st.radio("Sex", ("male", "female"), key='sex_radio')
            children = st.slider("Number of Dependents", 0, 5, 1, key='children_slider')
            
        with col3:
            smoker = st.radio("Smoker Status", ("yes", "no"), key='smoker_radio')
            
        st.markdown("---") 
        predict_button = st.form_submit_button("💰 Calculate Premium & Risk")

if predict_button:
    input_data = {
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker
    }

    FASTAPI_ENDPOINT_URL = "http://127.0.0.1:8000/predict_insurance_charge" 
    
    st.markdown("## 2. Model Output")
    
    try:
        for attempt in range(3):
            try:
                response = requests.post(FASTAPI_ENDPOINT_URL, json=input_data, timeout=10)
                response.raise_for_status() 
                prediction_result = response.json()
                break
            except requests.exceptions.ConnectionError:
                if attempt < 2:
                    st.toast(f"Connection failed, retrying in 2 seconds... (Attempt {attempt + 1})")
                    import time; time.sleep(2)
                else:
                    raise

        if "predicted_charge" in prediction_result and "risk_category" in prediction_result:
            predicted_charge = prediction_result["predicted_charge"]
            risk_category = prediction_result["risk_category"]

            st.metric(label="Estimated Annual Medical Charge", value=f"${predicted_charge:,.2f}")
            
            color = "green" if "Tier 1" in risk_category else ("orange" if "Tier 2" in risk_category else "red")
            
            st.markdown(f"""
            <div style='background-color: #f0f0f0; padding: 10px; border-left: 5px solid {color}; border-radius: 5px;'>
                <p style='margin: 0; font-weight: bold; color: #333;'>
                    Risk Assessment: <span style='color: {color};'>{risk_category}</span>
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.balloons() 
            
        else:
            st.error("Prediction result format invalid. Missing 'predicted_charge' or 'risk_category'.")
            st.json(prediction_result)

    except requests.exceptions.HTTPError as e:
        st.error(f"⚠️ FastAPI Error {e.response.status_code}: Check your backend endpoint logic.")
        st.code(e.response.text)
        
    except Exception as e:
        st.error(f"❌ An unexpected error occurred: Could not connect to API or process response: {e}")

st.markdown("---")
st.caption("Powered by Streamlit and a predictive model trained on the Kaggle Insurance Dataset.")