# DS and ML Final Project:<br> Detecting the Difficulty Level of French Texts

## Members of UNIL_Zurich
- Matteo Frison
- Takaaki Kishida


## Overview
In our startup *LingoRank*, we aim to revolutionize the way English speakers learn French by predicting the difficulty level of French texts. Our model assists learners by recommending texts that match their current language proficiency, ensuring an effective and gradual learning curve.


## Data
We utilized a dataset provided by the professor, which includes French texts with associated difficulty levels. 


## File Descriptions
- `data/data_raw/`: The folder stores the datasets provided at the Kaggle page.
  - `training_data.csv`: The dataset used for training the models.
  - `unlabelled_test_data.csv`: The test dataset for which the model predictions are generated.

- `src/`: This directory contains all the source code used in this project.
  - `BERT4.ipynb`: The main script for training the model and running predictions.
  - `logistic_regression.ipynb`: The script for logistic regression. It does not produce the best predictions but it was our initial try. The results are presented in our table below.

- `model/`
  - This folder stores our models exported from our code in `src/`.


## Deliverables
This repository documents the entire process of model development and provides the following components.

### (1) Model Comparison Table (NN = Neural Network)
| Metrics     | [Logistic Regression](https://github.com/takakishi/HEC_DS_ML_project/blob/main/src/logistic_regression.ipynb) | Readiblity features NN | CamemBert NN | Word Frequency NN | Final NN |
|-------------|---------|---------|---------|---------|---------|
| Precision   | 0.470   | 0.xxx   | 0.xxx   | 0.xxx   | 0.xxx   |
| Recall      | 0.472   | 0.xxx   | 0.xxx   | 0.xxx   | 0.xxx   |
| F1-Score    | 0.464   | 0.xxx   | 0.xxx   | 0.xxx   | 0.xxx   |
| Accuracy    | 0.475   | 0.xxx   | 0.xxx   | 0.xxx   | 0.xxx   |

### (2) Presentation Video
Please view our [YouTube presentation video](#) for a detailed explanation of our methodology and a demo of our application.

### (3) User Interface/Application
We also developed a user-friendly interface with [Streamlit](https://streamlit.io/), allowing users to input French text and receive a difficulty assessment instantly. You can open and use our app on your PC by running the following command on your terminal:
```bash
# Clone the repository by
git clone https://github.com/takakishi/HEC_DS_ML_project.git

# Install streamlit if you use it for the first time
pip install streamlit
cd repository_name # specify this repository
streamlit run .\src\streamlit_app.py
```


## Guide to Replication
- You can reproduce the results shown in the above table simply by running each notebook without using any other source files.
- If you want to export the models, you need to uncomment the last code chunks in the notebooks as necessary.
