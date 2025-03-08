# Churn-Analysis-E-Commerce

Customer Churn Prediction Project Report
1. Introduction
The goal of this project is to predict customer churn for an e-commerce platform. Churn is defined as the event where a customer stops using the service, and predicting this behavior early can help the company take proactive retention measures. In this project, we developed a machine learning model using a RandomForestClassifier, applied oversampling to balance the classes, and tuned the model’s hyperparameters to optimize performance.

2. Data Description
The dataset used in this project was extracted from an Excel file containing two sheets:

Data Dictionary: Provides descriptions for each column.
E Comm: The main dataset with features such as:
CustomerID: Unique identifier (dropped during preprocessing).
Churn: Target variable indicating if a customer churned (binary: 0 = retained, 1 = churned).
Customer Features: Includes tenure, preferred login device, preferred payment mode, gender, order count, coupon usage, satisfaction score, and several other variables.
The dataset presented challenges such as missing values and class imbalance (fewer churned customers compared to non-churners).

3. Data Preprocessing
3.1 Data Cleaning & Feature Selection
Dropping Non-Predictive Columns:
The CustomerID was dropped since it does not contribute to predicting churn.

Handling Missing Values:
Missing values were imputed using the median value of each feature, ensuring that numerical distributions are preserved.

Categorical Encoding:
Categorical features (e.g., PreferredLoginDevice, PreferredPaymentMode, Gender, PreferedOrderCat, MaritalStatus) were converted to numeric using Label Encoding. This step is crucial because machine learning models require numerical input.

3.2 Addressing Class Imbalance
SMOTE (Synthetic Minority Oversampling Technique):
Since the number of churned customers (minority class) was significantly lower than the number of retained customers, SMOTE was applied. This technique synthetically generates new samples for the minority class, resulting in a more balanced dataset. A balanced dataset improves model performance, especially in recall for the minority class.
3.3 Feature Consistency for Prediction
When deploying the model, it is essential that the input features during prediction match exactly (in name and order) the features used during training. We achieved this by reordering the input DataFrame according to the model’s feature_names_in_ attribute.
4. Model Tuning
4.1 Model Selection
RandomForestClassifier:
A Random Forest model was chosen for its robustness, ability to capture nonlinear relationships, and built-in feature importance metrics.
4.2 Hyperparameter Tuning
GridSearchCV:
To optimize model performance, a grid search was performed over multiple hyperparameters:
n_estimators: Number of trees in the forest.
max_depth: Maximum depth of each tree.
min_samples_split & min_samples_leaf: Minimum number of samples required to split a node and be at a leaf node, respectively.
class_weight: Set to “balanced” to further address class imbalance.
A 3-fold cross-validation was used during the grid search to ensure that the selected parameters generalized well to unseen data.
4.3 Decision Threshold Adjustment
Threshold Modification:
Instead of the default 0.5 threshold for classifying churn, the decision threshold was lowered (to 0.4) to improve recall for the churn class. This adjustment helps capture more true churners at the expense of a slight decrease in precision, which is often acceptable for retention strategies.
5. Evaluation Metrics
5.1 Key Metrics Reported
Accuracy:
The overall accuracy of the tuned model was approximately 98%, indicating that the model correctly classified 98% of the test cases.

Precision & Recall:

Non-Churn (0): High precision (around 98–99%) and near-perfect recall, meaning almost all non-churners were correctly identified.
Churn (1): Recall was significantly improved (up to 99%) after adjusting the decision threshold, ensuring that nearly all churn cases were detected. Precision remained high, indicating that when the model predicts churn, it is very likely to be correct.
F1-Score:
The F1-score for both classes was high, reflecting a good balance between precision and recall.

5.2 Confusion Matrix Insights
The confusion matrix (not detailed here) showed very few false negatives for churn. This is crucial because missing a churn prediction could mean a lost opportunity for customer retention.
6. Business Insights & Recommendations
6.1 Key Insights
Critical Factors:
Feature importance analysis revealed that variables such as tenure, order count, and satisfaction score are significant predictors of churn. Customers with shorter tenure, lower order count, and lower satisfaction scores are at a higher risk of churning.

Customer Segmentation:
By analyzing features like preferred order category and payment mode, the model can help identify customer segments that are more vulnerable to churn. This segmentation allows for targeted interventions.

6.2 Actionable Recommendations
Targeted Retention Strategies:
For customers identified as high risk:

Personalized Engagement: Reach out with personalized promotions, discounts, or loyalty rewards.
Improved Onboarding: Enhance the onboarding experience for new customers to increase early engagement and satisfaction.
Service Improvements: Address common pain points identified in the churn drivers (e.g., enhance app experience, streamline order processes).
Monitoring & Continuous Improvement:

Dashboard Implementation: Deploy the model via an interactive dashboard (e.g., using Streamlit) for real-time predictions and monitoring.
Periodic Retraining: Regularly update the model with new data to maintain accuracy and adapt to changing customer behaviors.
7. Conclusion
The project successfully developed a robust churn prediction model using RandomForestClassifier, with extensive data preprocessing and hyperparameter tuning. By addressing class imbalance with SMOTE and fine-tuning the decision threshold, the model achieved high recall for churn detection—a critical requirement for proactive customer retention strategies. The insights generated not only help predict churn but also provide actionable recommendations for business improvements.
