import streamlit as st
import pandas as pd
from fraud_data_generator import generate_synthetic_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")

st.title("💳 Synthetic Credit Card Fraud Detection")
st.markdown("Generate synthetic transactions and test fraud detection models.")

st.sidebar.header("Dataset Parameters")
n_rows = st.sidebar.number_input("Number of rows", min_value=1000, max_value=200000, value=100000, step=1000)
fraud_rate = st.sidebar.slider("Fraud rate (%)", min_value=0.1, max_value=10.0, value=2.5, step=0.1) / 100

if st.sidebar.button("Generate Dataset"):
    df = generate_synthetic_data(n_rows, fraud_rate)
    st.session_state.df = df

if "df" in st.session_state:
    st.subheader("📊 Dataset Preview")
    st.dataframe(st.session_state.df.head(20))
    st.write(f"Fraud rate in dataset: {st.session_state.df['is_fraud'].mean()*100:.2f}%")

    df = st.session_state.df.copy()
    le = LabelEncoder()
    df["merchant"] = le.fit_transform(df["merchant"])
    df["location"] = le.fit_transform(df["location"])

    X = df.drop(["is_fraud", "timestamp", "card_number"], axis=1)
    y = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    st.subheader("⚙️ Model Training: Logistic Regression & Random Forest")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=500),
        "Random Forest": RandomForestClassifier(n_estimators=100)
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        st.write(f"### {name}")
        st.text(classification_report(y_test, y_pred))

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

    csv = st.session_state.df.to_csv(index=False).encode('utf-8')
    st.download_button(label="📥 Download Dataset as CSV", data=csv, file_name="synthetic_fraud_data.csv", mime="text/csv")
else:
    st.info("⬅️ Use the sidebar to generate the dataset.")
