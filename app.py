from pathlib import Path

import mlflow
import pandas as pd
import streamlit as st

# ----- settings -----

# Run ID of the final model (run "xgb-final" in the MLflow UI)
RUN_ID = "76c771b2b95744a982eaf3bb82871102"

# absolute path to mlflow.db in the project folder (same as in the notebooks)
DB_PATH = (Path(__file__).parent / "mlflow.db").resolve()

# XGBoost predicts numbers, this is the order of the types (from the LabelEncoder)
CLASSES = ["Moderate", "Overcontroller", "Resilient", "Undercontroller"]

DESCRIPTIONS = {
    "Moderate": "Balanced, no extreme traits.",
    "Resilient": "Emotionally stable, calm under pressure.",
    "Overcontroller": "Anxious and introverted.",
    "Undercontroller": "Impulsive, less focused on rules.",
}

# the 19 questions from codebook.txt
QUESTIONS = {
    "N1": "I get stressed out easily.",
    "N2": "I am relaxed most of the time.",
    "N3": "I worry about things.",
    "N4": "I seldom feel blue.",
    "N5": "I am easily disturbed.",
    "N6": "I get upset easily.",
    "N7": "I change my mood a lot.",
    "N8": "I have frequent mood swings.",
    "N9": "I get irritated easily.",
    "N10": "I often feel blue.",
    "E1": "I am the life of the party.",
    "E3": "I feel comfortable around people.",
    "E4": "I keep in the background.",
    "E5": "I start conversations.",
    "E7": "I talk to a lot of different people at parties.",
    "E9": "I don't mind being the center of attention.",
    "E10": "I am quiet around strangers.",
    "C4": "I make a mess of things.",
    "A4": "I sympathize with others' feelings.",
}

# the model needs the columns in the same order as in data.csv
COLUMN_ORDER = [
    "N6", "N8", "N7", "N1", "N9", "N10", "N3", "N5", "E3", "N2",
    "E5", "N4", "E7", "E4", "age", "C4", "E1", "A4", "E10", "E9",
    "gender", "hand",
]


# ----- load the model -----

MODEL_DIR = Path(__file__).parent / "model"

@st.cache_resource
def load_model():
    if DB_PATH.exists():
        # local: load the model from the MLflow run (like in the course)
        mlflow.set_tracking_uri(f"sqlite:///{DB_PATH.as_posix()}")
        return mlflow.sklearn.load_model(f"runs:/{RUN_ID}/model")
    # online: load the model from the model folder in the repo
    return mlflow.sklearn.load_model(str(MODEL_DIR))

model = load_model()

# ----- page -----

st.title("Personality Type Prediction")
st.write("Answer the questions below. 1 = Disagree, 5 = Agree.")

answers = {}

st.header("About you")
answers["age"] = st.number_input("Age", min_value=13, max_value=100, value=25)
answers["gender"] = st.selectbox("Gender", ["Female", "Male", "Other"])
answers["hand"] = st.selectbox("Writing hand", ["Right", "Left", "Both"])

st.header("Questions")
for code, text in QUESTIONS.items():
    answers[code] = st.slider(text, 1, 5, 3)

if st.button("Predict"):
    # one row with all answers, in the right column order
    input_df = pd.DataFrame([answers])[COLUMN_ORDER]

    # probability for each of the 4 types (they add up to 1)
    probabilities = model.predict_proba(input_df)[0]

    # the type with the highest probability is the prediction
    best_index = int(probabilities.argmax())
    personality_type = CLASSES[best_index]
    confidence = probabilities[best_index]

    st.success(f"Your personality type: {personality_type}")
    st.write(DESCRIPTIONS[personality_type])
    st.metric("Confidence", f"{confidence:.0%}")
