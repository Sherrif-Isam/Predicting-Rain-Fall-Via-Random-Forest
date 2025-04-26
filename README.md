# Predicting-Rain-Fall-Via-Random-Forest

🌧️ Predicting Rainfall Using Random Forests
This project leverages a Random Forest Classifier to predict whether it will rain tomorrow in Australia based on weather observations. As part of a broader exploration into machine learning models, this project demonstrates the use of ensemble methods to improve predictive accuracy and robustness over single decision trees.

📦 Dataset
Source: Kaggle – Rain in Australia Dataset

Records: Approximately 142,000 samples

Target: RainTomorrow (Yes/No)

Features: Includes temperature, rainfall, humidity, wind direction, pressure, and more

🧠 What This Project Covers
Exploratory Data Analysis (EDA) and feature selection

Handling missing values and categorical encoding

Splitting the dataset into training and testing subsets

Building a Random Forest Classifier using scikit-learn

Hyperparameter tuning basics (number of estimators, max depth)

Feature importance visualization

Model evaluation on unseen test data

🌲 Why Random Forests?
Combines multiple decision trees to reduce overfitting

Increases model accuracy and stability

Handles missing values and outliers better than many single models

Useful for feature ranking and interpretability

⚙️ Tools & Libraries
Python 3.x

Pandas

NumPy

Matplotlib & Seaborn

Scikit-learn

Google Colab

📈 Model Evaluation
The Random Forest model was assessed using:

Accuracy Score

Confusion Matrix

Precision, Recall, and F1-Score

Feature importance analysis

Performance was compared against baseline logistic regression and decision tree models to understand improvements.

🧠 Key Learnings
How ensemble methods improve model generalization

Importance of feature importance interpretation in Random Forests

How hyperparameters like n_estimators and max_depth influence performance

Differences between simple trees and forests in bias-variance trade-offs

🚀 How to Run
Clone the repository:

bash
Copy
Edit
git clone https://github.com/Sherrif-Isam/Predicting-Rain-Fall-Via-Random-Forest.git
Open the notebook: Australia_Rain_RandomForest.ipynb

Execute the cells step-by-step using Google Colab or a local Jupyter Notebook setup.

📌 Author
Sherrif Isam
GitHub: @Sherrif-Isam
