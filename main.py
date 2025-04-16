import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

BC_MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
DIABETES_MODEL_PATH = os.path.join(MODEL_DIR, "diabetes.pkl")
TUMOR_MODEL_PATH = os.path.join(MODEL_DIR, "model.h5")
SYMPTOM_MODEL_PATH = os.path.join(MODEL_DIR, "svc.pkl")


# Load datasets
def load_csv(file_name):
    file_path = os.path.join(BASE_DIR, "datasets", file_name)
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        st.error(f"Error: {file_name} not found.")
        return pd.DataFrame()  # Return empty dataframe to prevent crashes


training_data = load_csv("Training.csv")
description = load_csv("description.csv")
precautions = load_csv("precautions_df.csv")
medications = load_csv("medications.csv")
diets = load_csv("diets.csv")
workout = load_csv("workout_df.csv")


# Load models
@st.cache_resource
def load_model_pickle(path):
    return joblib.load(path) if path.endswith('.pkl') else load_model(path)


diabetes_model = load_model_pickle(DIABETES_MODEL_PATH)
bc_model = load_model_pickle(BC_MODEL_PATH)
tumor_model = load_model_pickle(TUMOR_MODEL_PATH)

# Extract symptoms and disease labels from Training.csv
if not training_data.empty:
    symptoms_dict = {symptom: idx for idx, symptom in enumerate(training_data.columns[:-1])}
    disease_mapping = training_data.iloc[:, -1]  # Last column is Disease
else:
    symptoms_dict = {}
    disease_mapping = []


# Helper function to fetch data
def helper(dis):
    """Fetch description, precautions, medications, diet, and workout for a disease."""
    try:
        desc = description.loc[description['Disease'] == dis, 'Description'].values
        pre = precautions.loc[precautions['Disease'] == dis, ['Precaution_1', 'Precaution_2', 'Precaution_3',
                                                              'Precaution_4']].values.flatten()
        med = medications.loc[medications['Disease'] == dis, 'Medication'].values
        die = diets.loc[diets['Disease'] == dis, 'Diet'].values
        wrkout = workout.loc[workout['disease'] == dis, 'workout'].values

        # Convert all values to strings to avoid errors
        pre = [str(x) for x in pre if pd.notna(x)]
        med = [str(x) for x in med if pd.notna(x)]
        die = [str(x) for x in die if pd.notna(x)]
        wrkout = [str(x) for x in wrkout if pd.notna(x)]

        return desc[0] if len(desc) > 0 else "No data", pre, med, die, wrkout
    except Exception:
        return "No Data Available", [], [], [], []


# ✅ *Improved Prediction Logic (Cosine Similarity)*
def predict_disease(user_symptoms):
    """Predict disease using cosine similarity with Training.csv"""
    if training_data.empty:
        return "Unknown Disease"

    # Convert dataset symptoms into binary format
    symptom_columns = training_data.columns[:-1]
    symptom_data = training_data[symptom_columns].values

    # Convert user input into binary format
    input_vector = np.zeros(len(symptom_columns))
    for symptom in user_symptoms:
        if symptom in symptom_columns:
            input_vector[list(symptom_columns).index(symptom)] = 1

    # Compute cosine similarity
    similarities = cosine_similarity([input_vector], symptom_data)
    best_match_index = np.argmax(similarities)
    predicted_disease = training_data.iloc[best_match_index, -1]  # Last column is the disease

    return predicted_disease


# Sidebar Navigation
st.sidebar.title("Health Care Center - Disease Prediction")
app_type = st.sidebar.selectbox("Select Prediction Model", [
    "Symptom-Based Disease Prediction", "Diabetes Prediction", "Breast Cancer Prediction", "MRI Tumor Detection"
])

if app_type == "Symptom-Based Disease Prediction":
    st.title("Disease Prediction Based on Symptoms")

    # Use multiselect instead of text input for better accuracy
    selected_symptoms = st.multiselect("Select Symptoms", list(symptoms_dict.keys()))

    if st.button("Predict"):
        if selected_symptoms:
            predicted_disease = predict_disease(selected_symptoms)
            dis_des, precautions, medications, rec_diet, workout = helper(predicted_disease)

            st.success(f"Predicted Disease: {predicted_disease}")

            with st.expander("Disease Description", expanded=True):
                st.write(dis_des)

            with st.expander("Precautions", expanded=True):
                st.write("\n".join(precautions) if precautions else "No Precautions Available")

            with st.expander("Medications", expanded=True):
                st.write(", ".join(medications) if medications else "No Medications Found")

            with st.expander("Recommended Diet", expanded=True):
                st.write(", ".join(rec_diet) if rec_diet else "No Diet Recommendations")

            with st.expander("Suggested Workouts", expanded=True):
                st.write(", ".join(workout) if workout else "No Workout Recommendations")

        else:
            st.warning("Please select at least one symptom.")

elif app_type == "Diabetes Prediction":
    st.title("Diabetes Prediction")
    fields = {
        "Pregnancies": (0, 20),
        "Glucose": (70, 200),
        "Blood Pressure": (50, 180),
        "Skin Thickness": (0, 99),
        "Insulin": (0, 500),
        "BMI": (10.0, 50.0),
        "Diabetes Pedigree Function": (0.1, 2.5),
        "Age": (1, 120)
    }

    input_data = []
    for field, (min_val, max_val) in fields.items():
        value = st.number_input(
            f"{field} (Range: {min_val} - {max_val})",
            min_value=min_val,
            max_value=max_val,
        )
        input_data.append(value)

    if st.button("Predict Diabetes"):
        prediction = diabetes_model.predict([input_data])[0]
        st.success("Diabetic" if prediction == 1 else "Non-Diabetic")

elif app_type == "Breast Cancer Prediction":
    st.title("Breast Cancer Prediction")
    feature_ranges = {
        "id": (1, 99999), "radius_mean": (6, 30), "texture_mean": (9, 40), "perimeter_mean": (40, 200),
        "area_mean": (150, 2500), "smoothness_mean": (0.05, 0.15), "compactness_mean": (0.02, 0.3),
        "concavity_mean": (0.01, 0.4), "concave points_mean": (0.01, 0.2), "symmetry_mean": (0.1, 0.3),
        "fractal_dimension_mean": (0.05, 0.1), "radius_se": (0.1, 2.5), "texture_se": (0.5, 5.0),
        "perimeter_se": (0.5, 15.0), "area_se": (5, 500), "smoothness_se": (0.001, 0.03),
        "compactness_se": (0.002, 0.13), "concavity_se": (0.002, 0.4), "concave points_se": (0.001, 0.05),
        "symmetry_se": (0.008, 0.08), "fractal_dimension_se": (0.001, 0.03), "radius_worst": (7, 40),
        "texture_worst": (10, 50), "perimeter_worst": (50, 250), "area_worst": (200, 4000),
        "smoothness_worst": (0.07, 0.25), "compactness_worst": (0.03, 1.5), "concavity_worst": (0.03, 1.5),
        "concave points_worst": (0.02, 0.4), "symmetry_worst": (0.15, 0.7),
        "fractal_dimension_worst": (0.05, 0.2)

    }
    input_data = [st.number_input(feature, min_value=min_val, max_value=max_val) for feature, (min_val, max_val) in
                  feature_ranges.items()]

    if st.button("Predict Breast Cancer"):
        prediction = bc_model.predict([input_data])[0]
        st.success("Malignant (Cancer)" if prediction == 1 else "Benign (No Cancer)")

elif app_type == "MRI Tumor Detection":
    st.title("MRI Tumor Detection")
    uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded MRI Image")

        img = image.resize((128, 128)).convert('RGB')
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = tumor_model.predict(img_array)
        predicted_class_index = np.argmax(predictions)
        confidence_score = np.max(predictions)

        class_labels = ['pituitary', 'glioma', 'notumor', 'meningioma']
        result = "No Tumor" if class_labels[
                                   predicted_class_index] == 'notumor' else f"Tumor: {class_labels[predicted_class_index]}"

        st.success(f"Prediction: {result}")
        st.write(f"Confidence: {confidence_score * 100:.2f}%")
