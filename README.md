# 💳 Bank Churn Prediction ML Pipeline

## 📌 Project Overview
This project builds an end-to-end Machine Learning pipeline to predict customer churn in a bank. The goal is to identify customers who are likely to leave the bank using classification models.

---

## 📊 Dataset
- Dataset used: Customer Churn Records  
- Contains customer details such as:
  - Credit Score
  - Geography
  - Gender
  - Age
  - Balance
  - Number of Products
  - Estimated Salary
  - Satisfaction Score
  - Complaint status

📁 Dataset file: [Customer-Churn-Records.csv](./Customer-Churn-Records.csv)

---

## ⚙️ Technologies Used
- Python  
- Pandas, NumPy  
- Scikit-learn  
- XGBoost  
- Matplotlib, Seaborn  

---

## 🔄 Workflow
1. Data Loading and Exploration  
2. Data Cleaning (removed irrelevant columns like CustomerId, Surname)  
3. Feature Engineering  
4. Train-Test Split (80-20)  
5. Preprocessing:
   - StandardScaler for numerical features  
   - OneHotEncoder for categorical features  
6. Model Training using Pipeline  
7. Model Evaluation  

---

## 🤖 Models Used
- Logistic Regression  
- Random Forest Classifier  
- K-Nearest Neighbors (KNN)  
- XGBoost Classifier  

---

## 📈 Evaluation Metrics
- Accuracy Score  
- Confusion Matrix  
- Classification Report (Precision, Recall, F1-score)  

---

## 🚀 Key Features
- Used **ColumnTransformer** for handling mixed data types  
- Built **ML pipelines** for clean and scalable workflow  
- Compared multiple models for performance  
- Implemented proper preprocessing and encoding  

---


---


## ▶️ How to Run

1. Clone the repository:
```bash
git clone https://github.com/iddesh/bank-churn-ml-pipeline.git
cd bank-churn-ml-pipeline
