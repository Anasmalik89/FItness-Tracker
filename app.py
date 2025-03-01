import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import time
import warnings

warnings.filterwarnings('ignore')

# Set Page Configurations
st.set_page_config(page_title="Personal Fitness Tracker", layout="wide")

# Custom CSS Styling
st.markdown(
    """
    <style>
        .stApp {
            background-color: #1E1E1E;
        }
        .title {
            color: #00D4FF;
            text-align: center;
        }
        .sub-title {
            color: #FFD700;
            text-align: center;
        }
        .metric-box {
            background-color: #262730;
            padding: 10px;
            border-radius: 10px;
            text-align: center;
            color: white;
            font-size: 18px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# App Title
st.markdown("<h1 class='title'>Personal Fitness Tracker</h1>", unsafe_allow_html=True)
st.markdown("<h3 class='sub-title'>Predict Calories Burned Based on Your Inputs</h3>", unsafe_allow_html=True)
st.write("---")

# Sidebar Input Parameters
st.sidebar.header("Enter Your Details")

def user_input_features():
    age = st.sidebar.slider("Age", 10, 100, 30)
    bmi = st.sidebar.slider("BMI", 15, 40, 22)
    duration = st.sidebar.slider("Exercise Duration (min)", 0, 60, 20)
    heart_rate = st.sidebar.slider("Heart Rate", 60, 160, 80)
    body_temp = st.sidebar.slider("Body Temperature (°C)", 35, 42, 37)
    gender = st.sidebar.radio("Gender", ["Male", "Female"]) == "Male"
    return pd.DataFrame({"Age": [age], "BMI": [bmi], "Duration": [duration], "Heart_Rate": [heart_rate], "Body_Temp": [body_temp], "Gender_male": [int(gender)]})

user_data = user_input_features()

st.subheader("Your Entered Data")
st.dataframe(user_data, use_container_width=True)

# Load Dataset
@st.cache_data
def load_data():
    calories = pd.read_csv("calories.csv")
    exercise = pd.read_csv("exercise.csv")
    df = exercise.merge(calories, on="User_ID").drop(columns=["User_ID"])
    df["BMI"] = df["Weight"] / ((df["Height"] / 100) ** 2)
    df = df.round(2)
    return df

df = load_data()

# Train Model
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
X_train = pd.get_dummies(train_data.drop("Calories", axis=1), drop_first=True)
y_train = train_data["Calories"]

X_test = pd.get_dummies(test_data.drop("Calories", axis=1), drop_first=True)

model = RandomForestRegressor(n_estimators=500, max_depth=6, random_state=42)
model.fit(X_train, y_train)

# Align user input with model columns
user_data = user_data.reindex(columns=X_train.columns, fill_value=0)

# Predict Calories Burned
prediction = model.predict(user_data)[0]

st.write("---")
st.markdown("<h2 class='title'>Prediction</h2>", unsafe_allow_html=True)
st.markdown(f"<div class='metric-box'><b>{round(prediction, 2)}</b> kilocalories burned</div>", unsafe_allow_html=True)

# Similar Data Insights
st.write("---")
st.markdown("<h2 class='sub-title'>Similar Data Insights</h2>", unsafe_allow_html=True)
similar = df[(df["Calories"] >= prediction - 10) & (df["Calories"] <= prediction + 10)]
st.dataframe(similar.sample(5), use_container_width=True)

# Summary Statistics
st.write("---")
st.markdown("<h2 class='title'>General Statistics</h2>", unsafe_allow_html=True)
st.write(f"Your Age is higher than **{round((df['Age'] < user_data['Age'].values[0]).mean() * 100, 2)}%** of users.")
st.write(f"Your BMI is higher than **{round((df['BMI'] < user_data['BMI'].values[0]).mean() * 100, 2)}%** of users.")
st.write(f"Your Exercise Duration is longer than **{round((df['Duration'] < user_data['Duration'].values[0]).mean() * 100, 2)}%** of users.")
st.write(f"Your Heart Rate is higher than **{round((df['Heart_Rate'] < user_data['Heart_Rate'].values[0]).mean() * 100, 2)}%** of users.")
st.write(f"Your Body Temperature is higher than **{round((df['Body_Temp'] < user_data['Body_Temp'].values[0]).mean() * 100, 2)}%** of users.")

# Visualization
st.write("---")
st.markdown("<h2 class='sub-title'>Data Visualization</h2>", unsafe_allow_html=True)
fig, ax = plt.subplots(figsize=(8, 4))
sns.histplot(df["Calories"], bins=20, kde=True, color="cyan", ax=ax)
ax.set_title("Distribution of Calories Burned")
st.pyplot(fig)

st.write("---")
st.markdown("### Developed by Malik")