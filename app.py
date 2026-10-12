import streamlit as st
import pickle
import pandas as pd
import numpy as np

st.title("⌒ Cutoff Bracket Predictor App")
st.write("Predict the target cutoff bracket for university programmes using our final optimized LightGBM model.")

# Load the model and scaler mappings
@st.cache_resource
def load_artifacts():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scalar.pkl', 'rb') as f:
        mappings = pickle.load(f)
    return model, mappings

try:
    model, mappings = load_artifacts()
    st.success("Model artifacts loaded successfully!")
except Exception as e:
    st.error(f"Error loading artifacts: {e}")

# Interface Inputs
year = st.number_input("Year", min_value=2020, max_value=2030, value=2025)

university = st.selectbox("University", list(mappings['university'].keys()))
programme = st.selectbox("Programme", list(mappings['programme'].keys()))
college = st.selectbox("College", list(mappings['college'].keys()))
category = st.selectbox("Category", list(mappings['category'].keys()))
seat_type = st.selectbox("Seat Type", list(mappings['seat_type'].keys()))

if st.button("Predict Cutoff Class"):
    # Target Encode the selected options
    global_mean = mappings['global_mean']
    uni_encoded = mappings['university'].get(university, global_mean)
    prog_encoded = mappings['programme'].get(programme, global_mean)
    col_encoded = mappings['college'].get(college, global_mean)
    cat_encoded = mappings['category'].get(category, global_mean)
    seat_encoded = mappings['seat_type'].get(seat_type, global_mean)
    
    # Prepare feature input vector
    features = np.array([[year, uni_encoded, prog_encoded, col_encoded, cat_encoded, seat_encoded]])
    
    # Predict
    prediction = model.predict(features)[0]
    
    st.metric(label="Predicted Class Decile Bracket (0-9)", value=str(prediction))
    st.info("Note: Decile 0 corresponds to the lowest cutoffs, and Decile 9 corresponds to the highest.")
