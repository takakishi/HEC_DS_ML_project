# streamlit_app.py

# Libraries --------------------------------------------

import streamlit as st
import pandas as pd
import pickle
import requests
from io import BytesIO

import pkg_resources  # <- New import for checking packages
# Display installed packages and versions (For debugging purposes)
installed_packages = {d.project_name: d.version for d in pkg_resources.working_set}
st.text("Installed packages and versions:")
st.text(installed_packages)  # <- This will display the packages in the Streamlit app



# 0. Messages ------------------------------------------

st.title('Detecting the Difficulty Level of French Texts with LingoRank Innovative App')

# Description
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

# Function to load the trained model
@st.cache(allow_output_mutation=True)
def load_component(url):
    response = requests.get(url)
    component_file = BytesIO(response.content)
    component = pickle.load(component_file)
    return component

# Load the trained models
model_feature_url = 'https://raw.githubusercontent.com/takakishi/HEC_DS_ML_project/main/model/model_feature.pkl'
tfidf_vectorizer_url = 'https://raw.githubusercontent.com/takakishi/HEC_DS_ML_project/main/model/tfidf_vectorizer.pkl'
length_scaler_url = 'https://raw.githubusercontent.com/takakishi/HEC_DS_ML_project/main/model/length_scaler.pkl'

# model = load_component(model_feature_url)
# tfidf_vectorizer = load_component(tfidf_vectorizer_url)
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
    # Preprocessing the user input
    user_input_length = [[len(user_input.split())]]  # Calculate length as a list of lists
    processed_input_length = length_scaler.transform(user_input_length)

    # Display the processed length (You can replace this with your model prediction later)
    st.write(f"Processed sentence length: {processed_input_length[0][0]}")



# 4. Interact with Results ----------------------------

# E.g.,
model_accuracy = {'Model A': 0.90, 'Model B': 0.85}
st.bar_chart(model_accuracy)



# bash ------------------------------------------
# pip install streamlit
# streamlit run streamlit_app.py
