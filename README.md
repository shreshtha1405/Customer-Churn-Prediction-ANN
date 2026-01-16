# Customer Churn Prediction using Artificial Neural Network (ANN)

## 📌 Project Overview
Customer churn prediction is the process of identifying customers who are likely to leave a service or stop using a product.  
In this project, an **Artificial Neural Network (ANN)** model is built to predict whether a customer will churn or not based on historical data.

---

## 📊 Dataset Information
- Dataset Name: `churn.csv`
- Type: Customer Churn Dataset
- Target Variable: `Exited`
  - 1 → Customer Churned
  - 0 → Customer Stayed
- Features include:
  - Credit Score
  - Geography
  - Gender
  - Age
  - Tenure
  - Balance
  - Number of Products
  - Has Credit Card
  - Is Active Member
  - Estimated Salary

---

## ⚙️ Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow / Keras
- Matplotlib / Seaborn
- Jupyter Notebook

---

## 🧠 Model Used
- Artificial Neural Network (ANN)
- Architecture:
  - Input Layer
  - Hidden Layers with ReLU activation
  - Output Layer with Sigmoid activation
- Loss Function: Binary Crossentropy
- Optimizer: Adam

---

## 🔍 Project Workflow
1. Data Loading and Exploration
2. Data Preprocessing
   - Encoding categorical variables
   - Feature Scaling
3. Train-Test Split
4. ANN Model Building
5. Model Training
6. Prediction and Evaluation

---

## 📈 Model Performance
- The model predicts whether a customer will churn or not with good accuracy.
- This project demonstrates the application of deep learning for binary classification.

---

## 🚀 Conclusion
The ANN-based model helps in identifying customers who are at risk of churning, allowing businesses to take proactive retention measures.

---

## 📌 Future Improvements
- Add more visualizations for better EDA
- Include confusion matrix and classification report
- Perform hyperparameter tuning
- Try other ML/DL models for comparison
