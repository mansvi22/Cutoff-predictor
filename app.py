
import streamlit as st
import pickle
import numpy as np
from pathlib import Path

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Cutoff Bracket Predictor",
    page_icon="🎓",
    layout="centered"
)

# --------------------------------------------------
# APP HEADER
# --------------------------------------------------
st.title("🎓 Cutoff Bracket Predictor App")

st.write(
    "Predict the target cutoff bracket for university "
    "programmes using your optimized LightGBM model."
)

st.divider()

# --------------------------------------------------
# ARTIFACT PATHS
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
MAPPINGS_PATH = BASE_DIR / "scalar.pkl"


# --------------------------------------------------
# LOAD MODEL AND MAPPINGS
# --------------------------------------------------
@st.cache_resource
def load_artifacts():
    # Required if model.pkl contains a LightGBM model
    import lightgbm  # noqa: F401

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH.name}"
        )

    if not MAPPINGS_PATH.exists():
        raise FileNotFoundError(
            f"Mappings file not found: {MAPPINGS_PATH.name}"
        )

    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    with open(MAPPINGS_PATH, "rb") as file:
        mappings = pickle.load(file)

    if not isinstance(mappings, dict):
        raise ValueError(
            "scalar.pkl must contain a dictionary of mappings."
        )

    return model, mappings


try:
    model, mappings = load_artifacts()
    st.success("Model artifacts loaded successfully!")

except Exception as error:
    st.error("Unable to load the model artifacts.")
    st.code(str(error))

    st.markdown(
        """
        **How to fix this:**
        1. Check that `model.pkl` and `scalar.pkl` are in your
           GitHub repository.
        2. Add `lightgbm` to `requirements.txt`.
        3. Check the Streamlit Cloud deployment logs.
        4. Ensure the model was saved using compatible library versions.
        """
    )

    st.stop()


# --------------------------------------------------
# VALIDATE MAPPINGS
# --------------------------------------------------
required_keys = [
    "university",
    "programme",
    "college",
    "category",
    "seat_type",
    "global_mean"
]

missing_keys = [
    key for key in required_keys
    if key not in mappings
]

if missing_keys:
    st.error(
        "The mappings file is missing these keys: "
        + ", ".join(missing_keys)
    )
    st.stop()

mapping_fields = [
    "university",
    "programme",
    "college",
    "category",
    "seat_type"
]

for field in mapping_fields:
    if not isinstance(mappings[field], dict):
        st.error(
            f"The '{field}' mapping must be a dictionary."
        )
        st.stop()

    if not mappings[field]:
        st.error(
            f"No options are available for {field}."
        )
        st.stop()


# --------------------------------------------------
# USER INPUTS
# --------------------------------------------------
st.subheader("Enter Admission Details")

year = st.number_input(
    "Admission Year",
    min_value=2020,
    max_value=2030,
    value=2025,
    step=1
)

university = st.selectbox(
    "University",
    options=list(mappings["university"].keys())
)

programme = st.selectbox(
    "Programme",
    options=list(mappings["programme"].keys())
)

college = st.selectbox(
    "College",
    options=list(mappings["college"].keys())
)

category = st.selectbox(
    "Category",
    options=list(mappings["category"].keys())
)

seat_type = st.selectbox(
    "Seat Type",
    options=list(mappings["seat_type"].keys())
)

st.divider()


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
if st.button(
    "Predict Cutoff Class",
    type="primary",
    use_container_width=True
):

    try:
        global_mean = mappings["global_mean"]

        # Encode categorical inputs using saved mappings
        uni_encoded = mappings["university"].get(
            university, global_mean
        )

        prog_encoded = mappings["programme"].get(
            programme, global_mean
        )

        col_encoded = mappings["college"].get(
            college, global_mean
        )

        cat_encoded = mappings["category"].get(
            category, global_mean
        )

        seat_encoded = mappings["seat_type"].get(
            seat_type, global_mean
        )

        # Prepare input features in the expected order
        features = np.array([[
            year,
            uni_encoded,
            prog_encoded,
            col_encoded,
            cat_encoded,
            seat_encoded
        ]], dtype=float)

        # Generate prediction
        prediction = model.predict(features)[0]

        # Display result
        st.subheader("Prediction Result")

        st.metric(
            label="Predicted Class Decile Bracket",
            value=str(prediction)
        )

        st.caption(
            "The model predicts a decile from 0 to 9, "
            "where 0 represents the lowest cutoff bracket "
            "and 9 represents the highest."
        )

        # Display selected inputs
        with st.expander("View Your Selected Details"):
            st.write(f"**Admission Year:** {year}")
            st.write(f"**University:** {university}")
            st.write(f"**Programme:** {programme}")
            st.write(f"**College:** {college}")
            st.write(f"**Category:** {category}")
            st.write(f"**Seat Type:** {seat_type}")

    except Exception as error:
        st.error("Prediction failed.")
        st.code(str(error))


# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.divider()

st.caption(
    "Cutoff Bracket Predictor | Machine Learning Powered"
)
