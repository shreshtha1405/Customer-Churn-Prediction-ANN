# Customer Churn Prediction — ML Model Comparison

## 📌 Project Overview

This project predicts whether a telecom customer is likely to **churn (leave the service)** and compares multiple machine learning models to identify the best-performing approach.

The project focuses on data preprocessing, exploratory data analysis, class-imbalance handling, feature engineering, model training, and evaluation. Six models were compared, including an **Artificial Neural Network (ANN)**.

---

## 📊 Dataset

* **Dataset:** Telco Customer Churn
* **Records:** 7,043 customers
* **Features:** 21 columns
* **Target:** `Churn`

  * `Yes` → Customer churned
  * `No` → Customer stayed

### Key Features

* Demographics: Gender, Senior Citizen, Partner, Dependents
* Customer information: Tenure, Phone Service, Multiple Lines
* Internet services: Internet Service, Online Security, Online Backup, Device Protection, Tech Support
* Entertainment: Streaming TV, Streaming Movies
* Billing: Contract, Paperless Billing, Payment Method
* Financial: Monthly Charges, Total Charges

---

## 🛠️ Technologies Used

* **Python**
* **Pandas & NumPy**
* **Scikit-learn**
* **XGBoost**
* **TensorFlow / Keras**
* **imbalanced-learn (SMOTE)**
* **Matplotlib & Seaborn**
* **Jupyter Notebook / Google Colab**

---

## 🔍 Project Workflow

### 1. Data Cleaning

* Converted `TotalCharges` to numeric format
* Removed **11 rows** containing blank values
* Dropped the `customerID` column

### 2. Exploratory Data Analysis

Analyzed:

* Overall churn distribution
* Churn by gender and senior citizen status
* Churn across contract types
* Tenure vs. churn
* Feature correlations using a correlation heatmap

### 3. Data Encoding

* Converted binary Yes/No features into **0/1**
* Applied one-hot encoding to categorical features such as Contract, Internet Service, and Payment Method

### 4. Handling Class Imbalance

The original dataset contained:

* **5,163 non-churned customers**
* **1,869 churned customers**

Applied **SMOTE (Synthetic Minority Over-sampling Technique)** to address class imbalance.

### 5. Feature Scaling

Applied **MinMaxScaler** to numerical features:

* Tenure
* Monthly Charges
* Total Charges

### 6. Train-Test Split

Used an **80/20 stratified train-test split** to maintain the target-class distribution.

### 7. Model Training & Evaluation

Trained and compared six machine learning models using:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC

---

## 🧠 Models Compared

| Model               |  Accuracy | Precision |   Recall |  ROC-AUC |
| ------------------- | --------: | --------: | -------: | -------: |
| Logistic Regression |     83.8% |      0.85 |     0.83 |     0.93 |
| Decision Tree       |     81.3% |      0.81 |     0.82 |     0.81 |
| Random Forest       |     84.8% |      0.84 |     0.86 |     0.93 |
| KNN                 |     76.4% |      0.74 |     0.82 |     0.83 |
| **XGBoost**         | **84.9%** |  **0.85** | **0.85** | **0.94** |
| ANN (Keras)         |     74.0% |      0.87 |     0.56 |     0.85 |

### ANN Architecture

```text
Input (26 features)
        ↓
Dense(20, ReLU)
        ↓
Dense(15, ReLU)
        ↓
Dense(1, Sigmoid)
```

* **Loss Function:** Binary Crossentropy
* **Optimizer:** Adam
* **Epochs:** 20

---

## 📈 Key Insights

* **XGBoost achieved the best overall performance**, with 84.9% accuracy and 0.94 ROC-AUC.
* Random Forest also performed strongly with 84.8% accuracy and 0.86 recall.
* Tree-based models outperformed the ANN on this structured tabular dataset.
* **Contract Type and Tenure** were identified as important predictors of customer churn.
* The results demonstrate that model selection should depend on **data characteristics and business requirements**, rather than automatically choosing deep learning.

---

## 🚀 Conclusion

The model comparison shows that **XGBoost was the most suitable model** for this dataset based on its overall performance and ROC-AUC score.

For structured tabular data with approximately 7,000 records, traditional machine learning models such as XGBoost and Random Forest can outperform neural networks while remaining relatively easier to interpret.

---

## 🔮 Future Improvements

* Hyperparameter tuning using **GridSearchCV / Optuna**
* Experiment with **class weights and ADASYN**
* Perform additional feature engineering
* Compare additional ensemble models
* Deploy the best-performing model using a **REST API**
* Build a simple interface for real-time churn prediction

---

## 📁 Project Structure

```text
Customer-Churn-Prediction/
│
├── Customer_Churn_Prediction.ipynb
├── customer_churn.csv
└── README.md
```

## 👩‍💻 Author

**Shreshtha Sharma**

B.Tech CSE | Machine Learning & Data Analytics
