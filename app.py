import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# -------------------------------
# Dashboard Title & Navigation
# -------------------------------
st.title("Customer Churn Prediction Dashboard")
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Predict New Customer", "Model Performance", "Feature Importance"])

# -------------------------------
# Load the saved model
# -------------------------------
@st.cache_data
def load_model():
    return joblib.load("best_churn_model.pkl")

model = load_model()

# -------------------------------
# 1. Predict New Customer Page
# -------------------------------
if page == "Predict New Customer":
    st.header("Predict New Customer Churn")
    st.write("Enter the customer details below to predict the probability of churn.")
    
    with st.form("prediction_form"):
        # Primary features collected from user input
        PreferredLoginDevice = st.selectbox("Preferred Login Device", ["0", "1"])  
        PreferredPaymentMode = st.selectbox("Preferred Payment Mode", ["0", "1", "2", "3"])
        Gender = st.selectbox("Gender", ["0", "1"])
        PreferedOrderCat = st.selectbox("Preferred Order Category", ["0", "1", "2", "3"])
        MaritalStatus = st.selectbox("Marital Status", ["0", "1"])
        Tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
        # IMPORTANT: Adjusted the key name to match the model's expected name.
        OrderAmountHike = st.number_input("Order Amount Hike (%)", min_value=0, max_value=100, value=10)
        CouponUsed = st.number_input("Coupons Used", min_value=0, max_value=50, value=5)
        OrderCount = st.number_input("Order Count", min_value=0, max_value=100, value=50)
        DaySinceLastOrder = st.number_input("Days Since Last Order", min_value=0, max_value=365, value=30)
        CashbackAmount = st.number_input("Cashback Amount", min_value=0.0, value=100.0)
        
        # Additional features expected by the model with default values
        CityTier = st.number_input("City Tier", min_value=1, max_value=3, value=1)
        WarehouseToHome = st.number_input("Warehouse To Home (Distance)", min_value=0.0, value=10.0)
        HourSpendOnApp = st.number_input("Hour Spend On App", min_value=0.0, value=2.0)
        NumberOfDeviceRegistered = st.number_input("Number Of Devices Registered", min_value=0, value=2)
        SatisfactionScore = st.number_input("Satisfaction Score", min_value=0, max_value=10, value=5)
        NumberOfAddress = st.number_input("Number Of Addresses", min_value=0, value=1)
        Complain = st.number_input("Complain (0 or 1)", min_value=0, max_value=1, value=0)
        
        submitted = st.form_submit_button("Predict")
        
        if submitted:
            # Create a DataFrame with all the collected inputs
            input_data = pd.DataFrame({
                "PreferredLoginDevice": [int(PreferredLoginDevice)],
                "PreferredPaymentMode": [int(PreferredPaymentMode)],
                "Gender": [int(Gender)],
                "PreferedOrderCat": [int(PreferedOrderCat)],
                "MaritalStatus": [int(MaritalStatus)],
                "Tenure": [Tenure],
                # Use the expected column name exactly:
                "OrderAmountHikeFromlastYear": [OrderAmountHike],
                "CouponUsed": [CouponUsed],
                "OrderCount": [OrderCount],
                "DaySinceLastOrder": [DaySinceLastOrder],
                "CashbackAmount": [CashbackAmount],
                "CityTier": [int(CityTier)],
                "WarehouseToHome": [WarehouseToHome],
                "HourSpendOnApp": [HourSpendOnApp],
                "NumberOfDeviceRegistered": [int(NumberOfDeviceRegistered)],
                "SatisfactionScore": [int(SatisfactionScore)],
                "NumberOfAddress": [int(NumberOfAddress)],
                "Complain": [int(Complain)]
            })
            
            # Retrieve the expected feature names from the trained model
            # This requires that your model has the attribute feature_names_in_
            expected_columns = model.feature_names_in_
            input_data = input_data[expected_columns]
            
            # Make prediction: first get churn probability
            churn_prob = model.predict_proba(input_data)[0][1]
            prediction = model.predict(input_data)[0]
            
            st.write(f"**Churn Probability:** {churn_prob:.2f}")
            if prediction == 1:
                st.error("High risk of churn!")
            else:
                st.success("Customer likely to be retained!")

# -------------------------------
# 2. Model Performance Page
# -------------------------------
elif page == "Model Performance":
    st.header("Model Performance Over Time")
    
    # For demonstration purposes, we simulate historical performance metrics.
    performance_data = pd.DataFrame({
        "Date": pd.date_range(start="2023-01-01", periods=12, freq="M"),
        "Accuracy": np.random.uniform(0.95, 0.99, 12),
        "Recall": np.random.uniform(0.90, 1.0, 12),
        "Precision": np.random.uniform(0.95, 1.0, 12)
    })
    
    st.write("### Historical Performance Metrics")
    performance_data.set_index("Date", inplace=True)
    st.line_chart(performance_data)
    
    st.write("### Current Model Performance")
    st.info("Refer to the evaluation metrics printed during model tuning for current performance.")

# -------------------------------
# 3. Feature Importance Page
# -------------------------------
elif page == "Feature Importance":
    st.header("Feature Importance")
    
    # Retrieve the feature names from the model to ensure consistency.
    feature_names = model.feature_names_in_
    
    # Get feature importances from the model
    importance = model.feature_importances_
    importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importance})
    importance_df = importance_df.sort_values(by="Importance", ascending=False)
    
    # Plot feature importance using Plotly Express
    fig = px.bar(importance_df, x="Importance", y="Feature", orientation="h",
                 title="Feature Importance", text="Importance")
    fig.update_layout(yaxis=dict(autorange="reversed"))
    
    st.plotly_chart(fig, use_container_width=True)
