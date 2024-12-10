import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("dataset/data.csv")

# Preprocess the dataset
df['bmi'] = df['bmi'].fillna(df['bmi'].median())  # Fill missing BMI values with the median
df['gender_Male'] = df['gender'].apply(lambda x: 1 if x == 'Male' else 0)
df['gender_Female'] = df['gender'].apply(lambda x: 1 if x == 'Female' else 0)
df['ever_married'] = df['ever_married'].apply(lambda x: 1 if x == 'Yes' else 0)
df['work_type_Private'] = df['work_type'].apply(lambda x: 1 if x == 'Private' else 0)
df['work_type_Self_employed'] = df['work_type'].apply(lambda x: 1 if x == 'Self-employed' else 0)
df['work_type_Govt_job'] = df['work_type'].apply(lambda x: 1 if x == 'Govt_job' else 0)
df['work_type_children'] = df['work_type'].apply(lambda x: 1 if x == 'children' else 0)
df['work_type_Never_worked'] = df['work_type'].apply(lambda x: 1 if x == 'Never_worked' else 0)
df['Residence_type'] = df['Residence_type'].apply(lambda x: 1 if x == 'Urban' else 0)
df['smoking_status_formerly_smoked'] = df['smoking_status'].apply(lambda x: 1 if x == 'formerly smoked' else 0)
df['smoking_status_never_smoked'] = df['smoking_status'].apply(lambda x: 1 if x == 'never smoked' else 0)
df['smoking_status_smokes'] = df['smoking_status'].apply(lambda x: 1 if x == 'smokes' else 0)
df['smoking_status_Unknown'] = df['smoking_status'].apply(lambda x: 1 if x == 'Unknown' else 0)

# Drop non-numeric columns for model training
df_model = df.copy()
df_model.drop(['Residence_type', 'work_type', 'smoking_status', 'gender', 'ever_married'], axis=1, inplace=True)

X = df_model.drop('stroke', axis=1)
y = df_model['stroke']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression Class
class LogisticRegression:
    def __init__(self, learning_rate=0.001, n_iters=1000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)  # Clipping to avoid overflow
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        n_samples, n_features = X.shape
        self.weights = np.random.randn(n_features) * 0.01  # Random initialization of weights
        self.bias = 0

        for i in range(self.n_iters):
            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(linear_pred)

            dw = (1 / n_samples) * np.dot(X.T, (predictions - y))
            db = (1 / n_samples) * np.sum(predictions - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if i % 100 == 0:  # Print loss every 100 iterations to debug
                loss = -np.mean(y * np.log(predictions + 1e-7) + (1 - y) * np.log(1 - predictions + 1e-7))
                print(f"Iteration {i}: Loss = {loss:.4f}")

    def predict(self, X):
        X = np.array(X)
        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear_pred)
        return np.array([1 if i > 0.5 else 0 for i in y_pred])

    def predict_proba(self, X):
        X = np.array(X)
        linear_pred = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_pred)

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y == y_pred)

# Instantiate and train logistic regression
logistic_reg = LogisticRegression(learning_rate=0.001)
logistic_reg.fit(X_train_scaled, y_train)

# Sidebar for User Input
st.title("Stroke Risk Prediction")
st.sidebar.header("Enter Your Details")

# Collect user input details
age = st.sidebar.slider('Age', 18, 100, 30)
gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
hypertension = st.sidebar.selectbox('Hypertension', ['Yes', 'No'])
heart_disease = st.sidebar.selectbox('Heart Disease', ['Yes', 'No'])
ever_married = st.sidebar.selectbox('Ever Married', ['Yes', 'No'])
work_type = st.sidebar.selectbox('Work Type', ['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'])
residence_type = st.sidebar.selectbox('Residence Type', ['Urban', 'Rural'])
avg_glucose_level = st.sidebar.number_input('Average Glucose Level', 50, 300, 100)
bmi = st.sidebar.number_input('BMI', 10, 50, 20)
smoking_status = st.sidebar.selectbox('Smoking Status', ['formerly smoked', 'never smoked', 'smokes', 'Unknown'])

predict_button = st.sidebar.button('Predict Stroke Risk')

# Initialize user data dictionary with default values
user_data = {
    'age': age,
    'gender_Male': 1 if gender == 'Male' else 0,
    'gender_Female': 1 if gender == 'Female' else 0,
    'hypertension': 1 if hypertension == 'Yes' else 0,
    'heart_disease': 1 if heart_disease == 'Yes' else 0,
    'ever_married': 1 if ever_married == 'Yes' else 0,
    'work_type_Private': 1 if work_type == 'Private' else 0,
    'work_type_Self_employed': 1 if work_type == 'Self-employed' else 0,
    'work_type_Govt_job': 1 if work_type == 'Govt_job' else 0,
    'work_type_children': 1 if work_type == 'children' else 0,
    'work_type_Never_worked': 1 if work_type == 'Never_worked' else 0,
    'Residence_type': 1 if residence_type == 'Urban' else 0,
    'smoking_status_formerly_smoked': 1 if smoking_status == 'formerly smoked' else 0,
    'smoking_status_never_smoked': 1 if smoking_status == 'never smoked' else 0,
    'smoking_status_smokes': 1 if smoking_status == 'smokes' else 0,
    'smoking_status_Unknown': 1 if smoking_status == 'Unknown' else 0,
    'avg_glucose_level': avg_glucose_level,
    'bmi': bmi
}

# Convert user data to DataFrame
user_input_df = pd.DataFrame([user_data])

# Ensure the input DataFrame has the same columns as the training data
missing_cols = set(X.columns) - set(user_input_df.columns)
for col in missing_cols:
    user_input_df[col] = 0  # Add missing columns with default value 0

# Reorder the columns to match the order of the training data
user_input_df = user_input_df[X.columns]

# Only check for missing values after user presses the button
if predict_button:
    if user_input_df.isnull().values.any():
        st.error("Invalid input! Please ensure all fields are filled correctly.")
    else:
        # Scale the user input data to match the trained model features
        user_input_scaled = scaler.transform(user_input_df)

        stroke_prediction = logistic_reg.predict(user_input_scaled)
        probability = logistic_reg.predict_proba(user_input_scaled)

        # Extract probabilities for both classes (stroke risk = 1 and stroke risk = 0)
        stroke_risk_probability = probability[0] * 100
        no_stroke_risk_probability = (1 - probability[0]) * 100

        if stroke_prediction[0] == 1:
            st.subheader("You are at risk of a stroke.")
        else:
            st.subheader("You are not at risk of a stroke.")

        # Display both probabilities
        st.write(f"Probability of Stroke Risk: {stroke_risk_probability:.2f}%")
        st.write(f"Probability of No Stroke Risk: {no_stroke_risk_probability:.2f}%")
