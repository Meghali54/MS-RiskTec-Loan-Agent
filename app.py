import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Seed
np.random.seed(1234)

# Generate synthetic customer data
def generate_customer_data(n=1000):
    df = pd.DataFrame({
        "Age": np.random.randint(25, 66, n),
        "Income": np.random.randint(20000, 150001, n),
        "Occupation": np.random.choice(["Engineer", "Teacher", "Doctor", "Lawyer", "Others"], n),
        "LoanAmount": np.random.randint(5000, 50001, n),
        "LoanTerm": np.random.randint(12, 61, n),
        "RepaymentHistory": np.random.choice(["On Time", "Delayed", "Defaulted"], n, p=[0.7, 0.2, 0.1]),
        "CreditScore": np.random.randint(300, 851, n),
        "SavingsBalance": np.round(np.random.uniform(500, 10000, n), 2),
        "UnusualTransactions": np.random.randint(0, 7, n),
        "ResidentialChanges": np.random.randint(0, 4, n),
        "EmploymentStability": np.random.randint(1, 6, n),
        "IncomeStability": np.random.randint(1, 6, n),
        "ExpenseToIncomeRatio": np.round(np.random.uniform(0.3, 1, n), 2),
        "TransactionData": np.random.randint(1000, 50001, n)
    })

    def compute_default(row):
        if row['CreditScore'] > 700 and row['RepaymentHistory'] == "On Time":
            return 0
        elif row['CreditScore'] < 600 or row['RepaymentHistory'] == "Defaulted":
            return 1
        elif row['LoanAmount'] > 30000 and row['SavingsBalance'] < 1000:
            return 1
        elif row['UnusualTransactions'] > 2 or row['EmploymentStability'] < 3:
            return 1
        elif row['ExpenseToIncomeRatio'] > 0.5 and row['SavingsBalance'] < 1000:
            return 1
        elif row['Occupation'] == "Others" and row['Income'] < 25000:
            return 1
        elif row['LoanAmount'] > 0.6 * row['Income'] and row['RepaymentHistory'] == "Delayed":
            return 1
        else:
            return 0

    df["Default"] = df.apply(compute_default, axis=1)
    return df

# Generate data
df = generate_customer_data()

# Encode categorical variables
df_encoded = pd.get_dummies(df, columns=["Occupation", "RepaymentHistory"], drop_first=False)

# Split data
X = df_encoded.drop("Default", axis=1)
y = df_encoded["Default"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
X_train = X_train.astype(float)
X_test = X_test.astype(float)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=123)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

st.title("Customer Loan Default Prediction and Analysis")

# Display Classification Report and Confusion Matrix
st.write("### Model Evaluation")
report = classification_report(y_test, y_pred)
st.text("Classification Report:\n" + report)
st.text("Confusion Matrix:\n" + str(confusion_matrix(y_test, y_pred)))


# Score all data
df["Default_Prob"] = model.predict_proba(X)[:,1]
df["RiskSegment"] = pd.cut(df["Default_Prob"],
                           bins=[-0.01, 0.25, 0.65, 1],
                           labels=["Low Risk", "Medium Risk", "High Risk"])

# AI Recommendations
st.write("### AI Recommendations")
segment_summary = df.groupby("RiskSegment").agg({
    "Income": "mean",
    "LoanAmount": "mean",
    "CreditScore": "mean",
    "Default_Prob": "mean",
    "Default": "mean"
}).round(2)

st.dataframe(segment_summary)

st.write("\n Suggested Focus:")
if segment_summary.loc["High Risk"]["Default"] > 0.5:
    st.write(" High-Risk customers require better screening and stricter loan terms.")
if segment_summary.loc["Low Risk"]["Income"] > 50000:
    st.write(" Focus marketing on Low-Risk segment with higher income potential.")
if segment_summary.loc["Medium Risk"]["CreditScore"] >= 650:
    st.write(" Medium-Risk segment can be targeted for financial literacy and cross-sell offers.")

# Plot Feature Correlation Heatmap
st.write("### Feature Correlation Heatmap")
corr = X_test.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
st.pyplot(plt)

# Plot Feature Correlation with Target
st.write("### Feature Correlation with Target")
correlations = X_test.copy()
correlations['target'] = y_test.values
corr_with_target = correlations.corr()['target'].drop('target')

plt.figure(figsize=(10, 6))
corr_with_target.sort_values(ascending=False).plot(kind='bar', color='skyblue')
plt.title('Feature Correlation with Target')
plt.ylabel('Correlation Coefficient')
st.pyplot(plt)

# Plot Customer Risk Segments
st.write("### Customer Risk Segments")
fig = px.scatter(df, x="LoanAmount", y="CreditScore", color="RiskSegment",
                 title="Customer Risk Segments", opacity=0.7)
st.plotly_chart(fig)
