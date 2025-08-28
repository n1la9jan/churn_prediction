import streamlit as st
import joblib
import numpy as np
model = joblib.load('data/model.pkl')
scaler = joblib.load('data/scaler.pkl')


st.title('Churn Prediction')
st.divider()
st.write('Enter values and click on Predict button')
st.divider()
tenure = st.number_input('Tenure', min_value=0, max_value=72, value=10)
MonthlyCharges = st.number_input('Monthly Charges', min_value=50, max_value=200)
gender = st.selectbox('Enter Gender', ['Male', 'Female'])
st.divider()

prediction = st.button('Predict')
if prediction:
    selected_gender = 1 if gender == 'Female' else 0
    x = [tenure, MonthlyCharges, selected_gender]
    x1 = np.array(x)
    x_arr = scaler.transform([x1])
    predict = model.predict(x_arr)[0]
    predicted = 'The customer is likely to churn' if predict == 1 else 'The customer is likely to stay'
    st.write(predicted)
else:
    st.write('Click on Predict button to get the result')
