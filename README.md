# Customer Churn Prediction Project

This project predicts customer churn for an e-commerce platform using machine learning. By identifying customers at risk of churning, the business can take proactive steps to improve retention and increase customer satisfaction.

---

## Table of Contents

- [Overview](#overview)
- [Data Description](#data-description)
- [Data Preprocessing](#data-preprocessing)
- [Model Tuning](#model-tuning)
- [Evaluation Metrics](#evaluation-metrics)
- [Business Insights](#business-insights)
- [Usage](#usage)
- [Future Work](#future-work)
- [Conclusion](#conclusion)

---

## Overview

- **Goal:** Predict customer churn to help the business take proactive retention measures.
- **Model:** RandomForestClassifier
- **Techniques:**  
  - Data cleaning and feature engineering  
  - Label encoding and median imputation  
  - SMOTE for class balancing  
  - GridSearchCV for hyperparameter tuning  
  - Decision threshold adjustment to improve recall  
- **Dashboard:** An interactive dashboard built with Streamlit for real-time predictions and model monitoring.

---

## Data Description

The dataset contains customer-related features from an e-commerce platform. Key variables include:

- **Churn:** Target variable (0 = retained, 1 = churned).
- **Customer Features:** Tenure, order count, coupon usage, satisfaction score, etc.
- **Additional Features:** Preferred login device, preferred payment mode, city tier, etc.

*Challenges addressed:*
- **Missing Values:** Imputed using the median.
- **Class Imbalance:** Addressed using SMOTE.

---

## Data Preprocessing

1. **Data Cleaning & Feature Selection:**
   - Dropped non-predictive features like `CustomerID`.
   - Handled missing values via median imputation.
   - Converted categorical features (e.g., `PreferredLoginDevice`, `Gender`) to numeric using Label Encoding.

2. **Class Balancing:**
   - Applied SMOTE to balance the dataset by generating synthetic samples for the minority (churn) class.

3. **Feature Consistency:**
   - Ensured that the prediction input features (names and order) match exactly those used during model training, using the model's `feature_names_in_` attribute.

---

## Model Tuning

- **Algorithm:** RandomForestClassifier
- **Hyperparameter Tuning:**  
  - Used GridSearchCV with 3-fold cross-validation.
  - Tuned parameters including `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, and `class_weight` (set to `"balanced"`).

- **Decision Threshold Adjustment:**  
  - Lowered the classification threshold from 0.5 to 0.4 to improve recall on the churn class, ensuring high detection of potential churners.

---

## Evaluation Metrics

- **Accuracy:** ~98% overall
- **Precision & Recall:**  
  - **Non-Churn:** High precision (≈98–99%) and near-perfect recall.
  - **Churn:** Adjusted threshold leads to a high recall (~99%) while maintaining good precision.
- **F1-Score:** High for both classes, indicating a well-balanced model.
- **Confusion Matrix:**  
  - Very few false negatives for churn, crucial for timely retention actions.

---

## Business Insights

- **Key Predictors:**  
  - Shorter tenure, lower order count, and lower satisfaction scores are strong indicators of potential churn.
  
- **Actionable Recommendations:**  
  - **Targeted Retention:** Use personalized promotions, loyalty rewards, and improved onboarding for high-risk customers.
  - **Service Enhancements:** Focus on improving areas where churn risk is highest based on model insights.
  - **Continuous Monitoring:** Implement a dashboard for real-time monitoring and periodic model retraining.

---

## Usage

### Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
