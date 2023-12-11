# streamlit_app.py

# Libraries --------------------------------------------

import streamlit as st
import pandas as pd

st.title('Detecting the Difficulty Level of French Texts with LingoRank Innovative App')



# 1. Read Data from GitHub -----------------------------

@st.cache
def load_data(url):
    data = pd.read_csv(url)
    return data



# 2. URLs to raw CSV files -----------------------------

training_data_url = 'https://raw.githubusercontent.com/takakishi/HEC_DS_ML_project/main/data/data_raw/training_data.csv'
# add more

training_data = load_data(training_data_url)

# Display data
if st.checkbox('Show training data sample'):
    st.write(training_data.head())


# 3. User Input for Model Prediction ------------------

user_input = st.text_area("Enter French text here")

# if st.button('Predict Difficulty'):
#    prediction = your_model.predict([user_input])
#    st.write(f"The predicted difficulty level is: {prediction}")



# 4. Interact with Results ----------------------------

# E.g.,
model_accuracy = {'Model A': 0.90, 'Model B': 0.85}
st.bar_chart(model_accuracy)



# bash ------------------------------------------
# pip install streamlit
# streamlit run streamlit_app.py
