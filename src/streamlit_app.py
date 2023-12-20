# streamlit_app.py

# Libraries --------------------------------------------

import streamlit as st
import pandas as pd
import pickle
import requests
import joblib
from io import BytesIO
from scipy.sparse import hstack

# import pkg_resources  # for checking packages
# installed_packages = {d.project_name: d.version for d in pkg_resources.working_set}
# st.text("Installed packages and versions:")
# st.text(installed_packages)



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
# @st.cache(allow_output_mutation=True)
# def load_component(url):
#     response = requests.get(url)
#     if response.status_code == 200:  # Check if the request was successful
#         component_file = BytesIO(response.content)
#         component = joblib.load(component_file)
#         return component
#     else:
#         response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
# Load the trained models (Not working on Dec 17)
# model_feature_url = 'https://raw.githubusercontent.com/takakishi/HEC_DS_ML_project/main/model/model_feature.pkl'
# tfidf_vectorizer_url = 'https://raw.githubusercontent.com/takakishi/HEC_DS_ML_project/main/model/tfidf_vectorizer.pkl'
# length_scaler_url = 'https://raw.githubusercontent.com/takakishi/HEC_DS_ML_project/main/model/length_scaler.pkl'
# model = load_component(model_feature_url)
# tfidf_vectorizer = load_component(tfidf_vectorizer_url)
# length_scaler = load_component(length_scaler_url)

# Work well on Dec 17
# @st.cache(allow_output_mutation=True)
def load_component(url):
    response = requests.get(url)
    component_file = BytesIO(response.content)
    component = joblib.load(component_file)
    return component
# tfidf_vectorizer_url = 'https://raw.githubusercontent.com/takakishi/HEC_DS_ML_project/main/model/tfidf_vectorizer.joblib'
# length_scaler_url = 'https://raw.githubusercontent.com/takakishi/HEC_DS_ML_project/main/model/length_scaler.joblib'
# model_url = 'https://raw.githubusercontent.com/takakishi/HEC_DS_ML_project/main/model/log_reg_length_basic.joblib'
# model = load_component(model_url)
# tfidf_vectorizer = load_component(tfidf_vectorizer_url)
# length_scaler = load_component(length_scaler_url)

# URLs of the joblib files
bert_vectorizer_url = 'https://raw.githubusercontent.com/takakishi/HEC_DS_ML_project/main/model/bert_vectorizer.joblib'
classifier_concat_model_url = 'https://raw.githubusercontent.com/takakishi/HEC_DS_ML_project/main/model/classifier_concat_model.joblib'
classifier_feature_model_url = 'https://raw.githubusercontent.com/takakishi/HEC_DS_ML_project/main/model/classifier_feature_model.joblib'
classifier_tfidf_model_url = 'https://raw.githubusercontent.com/takakishi/HEC_DS_ML_project/main/model/classifier_tfidf_model.joblib'
tfidf_vectorizer_url = 'https://raw.githubusercontent.com/takakishi/HEC_DS_ML_project/main/model/tfidf_vectorizer.joblib'
final_NN_url = 'https://drive.google.com/file/d/1Ua4jAG8XuVqM_ziTabBTvzM_Re0v1PQK/view?usp=drive_link'

# Load the models
bert_vectorizer = load_component(bert_vectorizer_url)
classifier_concat_model = load_component(classifier_concat_model_url)
classifier_feature_model = load_component(classifier_feature_model_url)
classifier_tfidf_model = load_component(classifier_tfidf_model_url)
tfidf_vectorizer = load_component(tfidf_vectorizer_url)
final_NN = load_component(final_NN_url)





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

# Complete version for Logistic Regression
# if st.button('Predict Difficulty'):
#     # Preprocess the user input
#     user_input_tfidf = tfidf_vectorizer.transform([user_input])  # Vectorize text input
#     user_input_length = [[len(user_input.split())]]  # Calculate length
#     processed_input_length = length_scaler.transform(user_input_length)  # Scale length

#     # Combine TF-IDF and length features
#     final_input = hstack([user_input_tfidf, processed_input_length])

#     # Predict and display the difficulty level
#     prediction = model.predict(final_input)
#     st.write(f"The predicted difficulty level is: {prediction[0]}")

# When the user presses the 'Predict Difficulty' button, process the input
if st.button('Predict Difficulty'):
    # Preprocess the user input
    user_input_tfidf = tfidf_vectorizer.transform([user_input])  # Vectorize text input using TF-IDF
    # ... Add any additional preprocessing steps for your models here ...

    # Combine features from different models as needed for your final model
    # This is a placeholder; you'll need to modify it according to your actual model's needs
    combined_features = hstack([user_input_tfidf, ...])

    # Predict and display the difficulty level using your final model
    prediction = final_NN.predict(combined_features)
    st.write(f"The predicted difficulty level is: {prediction[0]}")



# 4. Interact with Results ----------------------------

# E.g.,
model_accuracy = {'Model A': 0.90, 'Model B': 0.85}
st.bar_chart(model_accuracy)
