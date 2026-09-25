import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ---------------------------------------
# PAGE SETTINGS
# ---------------------------------------

st.set_page_config(
    page_title="Motor Fault Prediction",
    page_icon="⚙️",
    layout="wide"
)

# ---------------------------------------
# CUSTOM CSS
# ---------------------------------------

st.markdown("""
<style>

.title {
    font-size: 42px;
    font-weight: bold;
    text-align: center;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    margin-bottom: 30px;
}

.result-box {
    padding: 25px;
    border-radius: 12px;
    text-align: center;
    font-size: 26px;
    font-weight: bold;
}

.normal {
    background-color: #d4edda;
    color: #155724;
}

.fault {
    background-color: #f8d7da;
    color: #721c24;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------
# LOAD MODEL AND DATASET
# ---------------------------------------

model = joblib.load("motor_fault_model.pkl")

data = pd.read_csv("balanced_motor_data.csv")

features = [
    "Voltage",
    "Current",
    "Temperature",
    "Vibration",
    "Power"
]

X = data[features]
y = data["Fault"]

# ---------------------------------------
# TRAIN / TEST SPLIT
# ---------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---------------------------------------
# MODEL PREDICTION
# ---------------------------------------

y_pred = model.predict(X_test)

# ---------------------------------------
# MODEL EVALUATION
# ---------------------------------------

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted"
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted"
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted"
)

# ---------------------------------------
# HEADER
# ---------------------------------------

st.markdown(
    '<div class="title">⚙️ Motor Fault Prediction System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-based motor condition monitoring using electrical and mechanical parameters'
    '</div>',
    unsafe_allow_html=True
)

st.divider()

# ---------------------------------------
# PROJECT METRICS
# ---------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Model Accuracy",
        f"{accuracy * 100:.2f}%"
    )

with col2:
    st.metric(
        "Dataset Records",
        len(data)
    )

with col3:
    st.metric(
        "Fault Classes",
        data["Fault"].nunique()
    )

st.divider()

# ---------------------------------------
# MODEL PERFORMANCE
# ---------------------------------------

st.subheader("📈 Model Performance")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Accuracy",
        f"{accuracy * 100:.2f}%"
    )

with col2:
    st.metric(
        "Precision",
        f"{precision * 100:.2f}%"
    )

with col3:
    st.metric(
        "Recall",
        f"{recall * 100:.2f}%"
    )

with col4:
    st.metric(
        "F1-Score",
        f"{f1 * 100:.2f}%"
    )

st.info(
    "Evaluation is based on the held-out test set from the balanced simulated dataset."
)

st.divider()

# ---------------------------------------
# INPUT SECTION
# ---------------------------------------

st.subheader("🔧 Enter Motor Parameters")

col1, col2 = st.columns(2)

with col1:

    voltage = st.number_input(
        "Voltage (V)",
        min_value=0.0,
        value=233.97,
        step=0.01
    )

    current = st.number_input(
        "Current (A)",
        min_value=0.0,
        value=4.32,
        step=0.01
    )

    temperature = st.number_input(
        "Temperature (°C)",
        min_value=0.0,
        value=46.36,
        step=0.01
    )

with col2:

    vibration = st.number_input(
        "Vibration",
        min_value=0.0,
        value=1.60,
        step=0.01
    )

    power = st.number_input(
        "Power (W)",
        min_value=0.0,
        value=1011.89,
        step=0.01
    )

st.write("")

# ---------------------------------------
# PREDICTION
# ---------------------------------------

if st.button(
    "🔍 PREDICT MOTOR CONDITION",
    use_container_width=True
):

    input_data = pd.DataFrame(
        [[
            voltage,
            current,
            temperature,
            vibration,
            power
        ]],
        columns=features
    )

    prediction = model.predict(input_data)[0]

    probabilities = model.predict_proba(input_data)[0]

    confidence = max(probabilities) * 100

    st.divider()

    st.subheader("📋 Prediction Result")

    # -----------------------------------
    # NORMAL CONDITION
    # -----------------------------------

    if prediction == "Normal":

        st.markdown(
            f"""
            <div class="result-box normal">
            🟢 MOTOR CONDITION: NORMAL<br>
            Confidence: {confidence:.2f}%
            </div>
            """,
            unsafe_allow_html=True
        )

        st.success(
            "The motor parameters indicate a normal operating condition."
        )

    # -----------------------------------
    # FAULT CONDITION
    # -----------------------------------

    else:

        st.markdown(
            f"""
            <div class="result-box fault">
            ⚠️ DETECTED CONDITION: {prediction}<br>
            Confidence: {confidence:.2f}%
            </div>
            """,
            unsafe_allow_html=True
        )

        explanations = {

            "Abnormal Vibration":
                "High vibration may indicate mechanical imbalance, misalignment, "
                "bearing problems, or other mechanical abnormalities.",

            "Overheating":
                "High temperature may indicate excessive heating or insufficient cooling.",

            "Overload":
                "High current or power consumption may indicate that the motor "
                "is operating under excessive load.",

            "Voltage Fault":
                "Unusual voltage conditions may indicate an electrical supply problem."
        }

        actions = {

            "Abnormal Vibration":
                "Inspect motor alignment, bearings, mounting, and mechanical components.",

            "Overheating":
                "Check cooling system, ventilation, temperature, and motor loading.",

            "Overload":
                "Check the connected load and motor current consumption.",

            "Voltage Fault":
                "Check the incoming voltage supply, wiring, and electrical connections."
        }

        if prediction in explanations:

            st.info(
                f"💡 **Possible Cause:** {explanations[prediction]}"
            )

            st.warning(
                f"🔧 **Recommended Action:** {actions[prediction]}"
            )

    # -----------------------------------
    # INPUT VALUES
    # -----------------------------------

    st.subheader("📊 Entered Motor Parameters")

    result_table = pd.DataFrame({

        "Parameter": features,

        "Value": [
            voltage,
            current,
            temperature,
            vibration,
            power
        ]
    })

    st.dataframe(
        result_table,
        use_container_width=True,
        hide_index=True
    )

# ---------------------------------------
# CONFUSION MATRIX
# ---------------------------------------

st.divider()

st.subheader("🔲 Confusion Matrix")

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=model.classes_
)

fig_cm, ax_cm = plt.subplots(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=model.classes_,
    yticklabels=model.classes_,
    ax=ax_cm
)

ax_cm.set_xlabel("Predicted Fault")
ax_cm.set_ylabel("Actual Fault")
ax_cm.set_title("Motor Fault Prediction - Confusion Matrix")

plt.xticks(rotation=30)
plt.yticks(rotation=0)
plt.tight_layout()

st.pyplot(fig_cm)

# ---------------------------------------
# CLASSIFICATION REPORT
# ---------------------------------------

st.subheader("📋 Classification Report")

report = classification_report(
    y_test,
    y_pred,
    output_dict=True
)

report_df = pd.DataFrame(report).transpose()

report_df["precision"] = report_df["precision"] * 100
report_df["recall"] = report_df["recall"] * 100
report_df["f1-score"] = report_df["f1-score"] * 100

report_df["support"] = report_df["support"].astype(int)

report_df = report_df.rename(columns={
    "precision": "Precision (%)",
    "recall": "Recall (%)",
    "f1-score": "F1-Score (%)",
    "support": "Support"
})

st.dataframe(
    report_df.round(2),
    use_container_width=True
)

# ---------------------------------------
# FEATURE IMPORTANCE
# ---------------------------------------

st.divider()

st.subheader("📊 Motor Parameter Importance")

importance = model.feature_importances_

importance_df = pd.DataFrame({

    "Parameter": features,

    "Importance": importance

}).sort_values(
    "Importance",
    ascending=False
)

fig, ax = plt.subplots(figsize=(8, 4))

ax.bar(
    importance_df["Parameter"],
    importance_df["Importance"]
)

ax.set_xlabel("Motor Parameter")
ax.set_ylabel("Importance")
ax.set_title("Feature Importance")

plt.xticks(rotation=20)
plt.tight_layout()

st.pyplot(fig)

# ---------------------------------------
# FAULT DISTRIBUTION
# ---------------------------------------

st.subheader("📈 Fault Distribution")

fault_counts = data["Fault"].value_counts()

fig2, ax2 = plt.subplots(figsize=(8, 4))

ax2.bar(
    fault_counts.index,
    fault_counts.values
)

ax2.set_xlabel("Fault Type")
ax2.set_ylabel("Number of Records")
ax2.set_title("Motor Fault Distribution")

plt.xticks(rotation=25)
plt.tight_layout()

st.pyplot(fig2)

# ---------------------------------------
# FOOTER
# ---------------------------------------

st.divider()

st.caption(
    "Motor Fault Prediction System | Random Forest Machine Learning Prototype | Simulated Dataset"
)
