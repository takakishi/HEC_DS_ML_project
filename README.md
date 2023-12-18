# DS and ML Final Project:<br> Detecting the Difficulty Level of French Texts

## Members
- Matteo Frison
- Takaaki Kishida


## Overview
In our startup *LingoRank*, we aim to revolutionize the way English speakers learn French by predicting the difficulty level of French texts. Our model assists learners by recommending texts that match their current language proficiency, ensuring an effective and gradual learning curve.


## Data
We utilized a dataset provided by the professor, which includes French texts with associated difficulty levels. 


## File Descriptions
- `src/`: This directory contains all the source code used in this project.
  - `XXX.ipynb`: The main script for training the model and running predictions.
  - `YYY.ipynb`: other code, not the best predictions but used in our table.

- `data/data_raw/`: The folder stores the datasets provided at the Kaggle page
  - `training_data.csv`: The dataset used for training the models.
  - `unlabelled_test_data.csv`: The test dataset for which the model predictions are generated.

- `note/`
  - `EDA.ipynb`: An exploratory data analysis visualize and understand the dataset.


## Deliverables
This repository documents the entire process of model development and provides the following components.

### (1) Model Comparison Table
| Metrics     | Logistic Regression | Neural Network |
|-------------|---------------------|----------------|
| Precision   | ...                 | ...            |
| Recall      | ...                 | ...            |
| F1-Score    | ...                 | ...            |
| Accuracy    | ...                 | ...            |

### (2) Presentation Video
Please view our [YouTube presentation video](#) for a detailed explanation of our methodology and a demo of our application.

### (3) User Interface/Application
We also developed a user-friendly interface with [Streamlit](https://streamlit.io/), allowing users to input French text and receive a difficulty assessment instantly. You can open and use our app on your PC by running the following command on your terminal:
```bash
cd set_this_repository
streamlit run .\src\streamlit_app.py
```


## Guide to Replication (TBC)
