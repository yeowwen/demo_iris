import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the Iris dataset
df = pd.read_csv('iris.csv')

# Sidebar to select features and model parameters
st.sidebar.header('Select Features and Model Parameters')

# Select features for prediction
selected_features = st.sidebar.multiselect('Select Features for Prediction', df.columns[:-1])

# Split the data into features (X) and target variable (y)
X = df[selected_features]
y = df['Species']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = SVC()
model.fit(X_train, y_train)

# Model prediction
y_pred = model.predict(X_test)

# Display accuracy
accuracy = accuracy_score(y_test, y_pred)
st.sidebar.text(f'Model Accuracy: {accuracy:.2f}')

# Main content
st.title('Iris Species Prediction App')

# Display the DataFrame
st.subheader('DataFrame')
st.write(df)

# Prediction
st.subheader('Prediction')
user_input = {}
for feature in selected_features:
    user_input[feature] = st.number_input(f'Enter {feature}', min_value=df[feature].min(), max_value=df[feature].max())

prediction = model.predict(pd.DataFrame([user_input]))
st.write(f'Predicted Species: {prediction[0]}')