Customer Churn Prediction — ML Model Comparison
📌 Project Overview

Customer churn prediction identifies customers who are likely to stop using a service, so a business can take proactive retention action. This project builds and compares multiple machine learning models — including an Artificial Neural Network (ANN) — to predict whether a telecom customer will churn.

📊 Dataset Information
Dataset: Telco Customer Churn dataset (customer_churn.csv)
Records: 7,043 customers, 21 columns
Target Variable: Churn
Yes → Customer churned
No → Customer stayed
Features include:
Gender, Senior Citizen, Partner, Dependents
Tenure (months with company)
Phone Service, Multiple Lines, Internet Service
Online Security, Online Backup, Device Protection, Tech Support
Streaming TV, Streaming Movies
Contract Type, Paperless Billing, Payment Method
Monthly Charges, Total Charges
⚙️ Technologies Used
Python
Pandas, NumPy
Scikit-learn, XGBoost, imbalanced-learn (SMOTE)
TensorFlow / Keras
Matplotlib, Seaborn
Jupyter / Google Colab
🔍 Project Workflow
Data Cleaning — converted TotalCharges to numeric, removed 11 blank rows, dropped customerID
EDA — analyzed churn distribution, churn by gender/senior citizen/contract type, tenure vs churn, correlation heatmap
Encoding — mapped Yes/No columns to 1/0; one-hot encoded multi-category columns (Internet Service, Contract, Payment Method)
Handling Class Imbalance — dataset was imbalanced (5,163 "No" vs 1,869 "Yes"); applied SMOTE to balance both classes
Feature Scaling — applied MinMaxScaler to tenure, MonthlyCharges, TotalCharges
Train-Test Split — 80/20 split, stratified on target
Model Training & Comparison — trained 6 models and evaluated on Accuracy, Precision, Recall, F1-score, ROC-AUC
🧠 Models Compared
Model	Accuracy	Precision	Recall	ROC-AUC
Logistic Regression	83.8%	0.85	0.83	0.93
Decision Tree	81.3%	0.81	0.82	0.81
Random Forest	84.8%	0.84	0.86	0.93
KNN	76.4%	0.74	0.82	0.83
XGBoost	84.9%	0.85	0.85	0.94
ANN (Keras)	74.0%	0.87	0.56	0.85

ANN Architecture: Input (26 features) → Dense(20, ReLU) → Dense(15, ReLU) → Dense(1, Sigmoid) Loss: Binary Crossentropy | Optimizer: Adam | Epochs: 20

📈 Key Insight

XGBoost and Random Forest outperformed the ANN on this dataset. Since the data is structured/tabular with a moderate size (~7,000 rows), tree-based models captured feature interactions more effectively than the neural network, which typically needs larger and/or unstructured data (images, text) to show its advantage.

Feature importance and point-biserial correlation analysis identified Contract Type and Tenure as the strongest predictors of churn.

🚀 Conclusion

The comparison shows that model choice should be driven by data characteristics, not by defaulting to deep learning. XGBoost was selected as the best-performing model for this problem, balancing accuracy and interpretability.

📌 Future Improvements
Hyperparameter tuning (GridSearchCV / Optuna) for XGBoost and ANN
Try additional class-imbalance techniques (class weights, ADASYN)
Deploy the best model via a simple API for real-time predictions
