import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

st.set_page_config(page_title="AI Cancer Intelligence", layout="wide")

st.title(" AI Cancer Intelligence Dashboard")
st.markdown("Advanced Machine Learning Analytics System")
@st.cache_data
def load_data():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    return X, y

X, y = load_data()
st.sidebar.header("Model Configuration")
test_size = st.sidebar.slider("Test Size (%)", 10, 40, 20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size/100, random_state=42
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
models = {
    "Logistic Regression": LogisticRegression(max_iter=5000),
    "Random Forest": RandomForestClassifier(),
    "Support Vector Machine": SVC(probability=True)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = {
        "model": model,
        "accuracy": acc
    }
st.subheader(" Model Accuracy Comparison")

acc_df = pd.DataFrame({
    "Model": results.keys(),
    "Accuracy": [results[m]["accuracy"] for m in results]
})

st.bar_chart(acc_df.set_index("Model")) 
best_model_name = max(results, key=lambda x: results[x]["accuracy"])
best_model = results[best_model_name]["model"]

st.success(f" Best Performing Model: {best_model_name}")

st.subheader(" Confusion Matrix (Best Model)")
y_pred_best = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)
fig1, ax1 = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm", ax=ax1)
st.pyplot(fig1)
st.subheader("ROC Curve")
y_prob = best_model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
fig2, ax2 = plt.subplots()
ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
ax2.plot([0,1],[0,1],'--')
ax2.set_xlabel("False Positive Rate")
ax2.set_ylabel("True Positive Rate")
ax2.legend()
st.pyplot(fig2)
if best_model_name == "Random Forest":
    st.subheader("📌 Feature Importance")

    importance = best_model.feature_importances_
    imp_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": importance
    }).sort_values(by="Importance", ascending=False).head(10)
    st.bar_chart(imp_df.set_index("Feature"))
st.subheader("🔍 Patient Risk Prediction")
input_data = []
for feature in X.columns[:5]:
    value = st.number_input(
        f"{feature}",
        float(X[feature].min()),
        float(X[feature].max()),
        float(X[feature].mean())
    )
    input_data.append(value)

if st.button("Predict Risk"):
    full_input = np.zeros(X.shape[1])
    full_input[:5] = input_data
    full_input = scaler.transform([full_input])
    prediction = best_model.predict(full_input)[0]
    probability = best_model.predict_proba(full_input)[0][1]
    if prediction == 1:
        st.success(f"Low Risk (Confidence: {probability*100:.2f}%)")
    else:
        st.error(f"High Risk (Confidence: {probability*100:.2f}%)")