# streamlit_app.py

# Libraries --------------------------------------------

import streamlit as st
import pandas as pd
import pickle
import requests
import joblib
from io import BytesIO
from scipy.sparse import hstack



# 0. Messages ------------------------------------------

st.title('Detecting the Difficulty Level of French Texts with LingoRank Innovative App')

st.markdown("""
Welcome to the LingoRank App, an innovative platform designed for both learners and educators to gauge the complexity of French texts. Our application harnesses the power of machine learning to evaluate text difficulty, providing instant feedback to help users select reading materials that align with their proficiency levels.
""")

st.markdown("""
Whether you're a language instructor seeking to curate tailored educational content, or a self-learner endeavoring to improve your French, LingoRank is your go-to companion for a seamless and adaptive learning experience.
""")

st.markdown("""
Discover how we can elevate your French learning journey!
""")



# 1. Read Data from GitHub -----------------------------

# @st.cache(allow_output_mutation=True)
def load_component(url):
    response = requests.get(url)
    component_file = BytesIO(response.content)
    component = joblib.load(component_file)
    return component
tfidf_vectorizer_url = 'https://raw.githubusercontent.com/takakishi/HEC_DS_ML_project/main/model/tfidf_vectorizer.joblib'
length_scaler_url = 'https://raw.githubusercontent.com/takakishi/HEC_DS_ML_project/main/model/length_scaler.joblib'
model_url = 'https://raw.githubusercontent.com/takakishi/HEC_DS_ML_project/main/model/log_reg_length_basic.joblib'
model = load_component(model_url)
tfidf_vectorizer = load_component(tfidf_vectorizer_url)
length_scaler = load_component(length_scaler_url)



# 2. URLs to raw CSV files -----------------------------

training_data_url = 'https://raw.githubusercontent.com/takakishi/HEC_DS_ML_project/main/data/data_raw/training_data.csv'
# unlabelled_test_data_url = pd.read_csv('https://raw.githubusercontent.com/takakishi/HEC_DS_ML_project/main/data/data_raw/unlabelled_test_data.csv')
# sample_submission_url = pd.read_csv('https://raw.githubusercontent.com/takakishi/HEC_DS_ML_project/main/data/data_raw/sample_submission.csv')

def load_data(url):
    data = pd.read_csv(url)
    return data

training_data = load_data(training_data_url)

# Display data
if st.checkbox('Show our training data sample'):
    st.write(training_data.head())



# 3. User Input for Model Prediction ------------------

user_input = st.text_area("Enter French text here")

if st.button('Predict Difficulty'):
    # Preprocess the user input
    user_input_tfidf = tfidf_vectorizer.transform([user_input])  # Vectorize text input
    user_input_length = [[len(user_input.split())]]  # Calculate length
    processed_input_length = length_scaler.transform(user_input_length)  # Scale length

    # Combine TF-IDF and length features
    final_input = hstack([user_input_tfidf, processed_input_length])

    # Predict and display the difficulty level
    prediction = model.predict(final_input)
    st.write(f"The predicted difficulty level is: {prediction[0]}")



# 4. Interact with Results ----------------------------

# E.g.,
# model_accuracy = {'Model A': 0.90, 'Model B': 0.85}
# st.bar_chart(model_accuracy)
