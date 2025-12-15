import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# ================= PAGE CONFIG =================
st.set_page_config(page_title="Salary Prediction Dashboard",
                   page_icon="💼", layout="centered")

st.title("💼 Salary Prediction Dashboard")
st.markdown("Predict salary based on your experience and profile using our dataset.")

# ================= ADMIN PANEL: UPLOAD DATA =================
st.sidebar.header("⚙️ Admin Panel")
uploaded_file = st.sidebar.file_uploader("Upload CSV to retrain model", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    # Default dataset path
    df = pd.read_csv("Salary Data.csv")


st.subheader("Dataset Preview")
st.dataframe(df.head())

# Drop missing values
df = df.dropna(subset=["Age","Gender","Education Level","Job Title","Years of Experience","Salary"])

# ================= TRAIN MODEL =================
df_encoded = pd.get_dummies(df, columns=["Gender","Education Level","Job Title"], drop_first=True)

X = df_encoded.drop("Salary", axis=1)
y = df_encoded["Salary"]

# Train/test split for model confidence
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# ================= MODEL CONFIDENCE =================
y_pred_test = model.predict(X_test)
r2 = r2_score(y_test, y_pred_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

st.sidebar.markdown("### 📊 Model Confidence")
st.sidebar.info(f"R² Score: {r2:.2f}\nRMSE: ₹ {rmse:,.0f}")

# ================= USER INPUTS =================
st.subheader("Enter Your Details")

age = st.selectbox("Age", sorted(df["Age"].unique()))
gender = st.selectbox("Gender", df["Gender"].unique())
education = st.selectbox("Education Level", df["Education Level"].unique())
job = st.selectbox("Job Title", df["Job Title"].unique())
experience = st.selectbox("Years of Experience", sorted(df["Years of Experience"].unique()))

# ================= PREPARE USER INPUT =================
input_dict = {"Age": age, "Years of Experience": experience}

for col in X.columns:
    if col.startswith("Gender_"):
        input_dict[col] = 1 if col == f"Gender_{gender}" else 0
    elif col.startswith("Education Level_"):
        input_dict[col] = 1 if col == f"Education Level_{education}" else 0
    elif col.startswith("Job Title_"):
        input_dict[col] = 1 if col == f"Job Title_{job}" else 0

input_df = pd.DataFrame([input_dict])

# ================= PREDICTION =================
if st.button("📈 Predict Salary"):
    salary = model.predict(input_df)[0]
    st.success(f"💰 Estimated Salary: ₹ {salary:,.0f}")
    st.info(f"Salary Range: ₹ {salary*0.9:,.0f} - ₹ {salary*1.1:,.0f}")

    # ================= SCENARIO ANALYSIS =================
    st.subheader("🔮 Scenario: +2 Years Experience")
    input_df_scenario = input_df.copy()
    input_df_scenario["Years of Experience"] += 2
    future_salary = model.predict(input_df_scenario)[0]
    st.info(f"Future Salary Estimate: ₹ {future_salary:,.0f}")

    # ================= ROLE-BASED COMPARISON =================
    st.subheader("💡 Role-Based Salary Comparison")
    role_avg = df.groupby("Job Title")["Salary"].mean().sort_values(ascending=False)
    st.bar_chart(role_avg)

    # ================= EXPERIENCE VS SALARY CHART =================
    st.subheader("📊 Experience vs Salary Trend")
    exp_salary = df.groupby("Years of Experience")["Salary"].mean()
    fig, ax = plt.subplots()
    ax.plot(exp_salary.index, exp_salary.values, marker='o', color='blue')
    ax.set_xlabel("Years of Experience")
    ax.set_ylabel("Average Salary")
    ax.set_title("Salary Growth by Experience")
    st.pyplot(fig)

    # ================= HIRING COST ESTIMATOR =================
    st.subheader("💼 Hiring Cost Estimator")
    team_size = st.number_input("Enter team size", min_value=1, value=1)
    monthly_cost = salary / 12
    st.write(f"💵 Monthly Cost per Employee: ₹ {monthly_cost:,.0f}")
    st.write(f"💰 Annual Cost per Employee: ₹ {salary:,.0f}")
    st.write(f"💼 Total Annual Cost for {team_size} Employees: ₹ {salary*team_size:,.0f}")

    # ================= EXPORT REPORT =================
    st.subheader("📥 Export Prediction Report")
    report = pd.DataFrame({
        "Feature": ["Age", "Gender", "Education Level", "Job Title", "Years of Experience", "Predicted Salary"],
        "Value": [age, gender, education, job, experience, salary]
    })
    st.download_button("⬇️ Download Report as CSV", report.to_csv(index=False), "salary_report.csv", "text/csv")
