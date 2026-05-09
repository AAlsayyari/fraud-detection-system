### run "python -m streamlit run notebooks/dashboard_V3.py" in terminal to start the dashboard

import streamlit as st
import joblib
import numpy as np
import os

# Resolve paths relative to this script's location
_dir = os.path.dirname(os.path.abspath(__file__))
_src = os.path.normpath(os.path.join(_dir, '..', 'src'))

# Load artifacts
model = joblib.load(os.path.join(_src, 'fraud_model.pkl'))
le = joblib.load(os.path.join(_src, 'label_encoder.pkl'))
feature_names = joblib.load(os.path.join(_src, 'feature_names.pkl'))
optimal_threshold = joblib.load(os.path.join(_src, 'optimal_threshold.pkl'))

# --- Example transactions ---
EXAMPLES = {
    "Fraud - TRANSFER (account emptied)": {
        "amount": 200000.0,
        "oldbalanceOrg": 200000.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
        "type": "TRANSFER",
        "description": "The sender transfers their entire balance. The receiver balance does not increase, suggesting the money vanished into a mule account."
    },
    "Fraud - CASH_OUT (account emptied)": {
        "amount": 150000.0,
        "oldbalanceOrg": 150000.0,
        "oldbalanceDest": 500000.0,
        "newbalanceDest": 500000.0,
        "type": "CASH_OUT",
        "description": "The sender cashes out their entire balance. The receiver balance stays the same even though they received 150k, which is a clear sign of data tampering."
    },
    "Legitimate - TRANSFER": {
        "amount": 5000.0,
        "oldbalanceOrg": 80000.0,
        "oldbalanceDest": 12000.0,
        "newbalanceDest": 17000.0,
        "type": "TRANSFER",
        "description": "A normal transfer of 5000. The sender has plenty of remaining balance, and the receiver balance increases by the expected amount."
    },
    "Legitimate - CASH_OUT": {
        "amount": 2500.0,
        "oldbalanceOrg": 45000.0,
        "oldbalanceDest": 100000.0,
        "newbalanceDest": 102500.0,
        "type": "CASH_OUT",
        "description": "A normal cash out of 2500. The sender keeps most of their balance, and the receiver balance increases correctly."
    },
}

# --- Page config ---
st.set_page_config(page_title="Fraud Detection System", layout="wide")
st.title("Financial Fraud Detection System")
st.markdown("Enter transaction details below, or load an example to see how the model works.")

# --- Example selector ---
st.subheader("Example Transactions")
cols = st.columns(len(EXAMPLES))
selected_example = None

for i, (name, data) in enumerate(EXAMPLES.items()):
    with cols[i]:
        if st.button(name, use_container_width=True):
            selected_example = name

# Persist selected example in session state
if selected_example:
    st.session_state["example"] = selected_example

active_example = EXAMPLES.get(st.session_state.get("example", ""), None)

# Show example description if one is selected
if active_example:
    st.info(active_example["description"])

st.markdown("---")

# --- Input form ---
st.subheader("Transaction Details")

col1, col2 = st.columns(2)

with col1:
    amount = st.number_input(
        "Transaction Amount",
        min_value=0.0,
        value=active_example["amount"] if active_example else 0.0,
        help="How much money is being sent in this transaction."
    )
    oldbalanceOrg = st.number_input(
        "Sender Balance Before Transaction",
        min_value=0.0,
        value=active_example["oldbalanceOrg"] if active_example else 0.0,
        help="The sender's account balance right before this transaction."
    )

with col2:
    oldbalanceDest = st.number_input(
        "Receiver Balance Before Transaction",
        min_value=0.0,
        value=active_example["oldbalanceDest"] if active_example else 0.0,
        help="The receiver's account balance right before this transaction."
    )
    newbalanceDest = st.number_input(
        "Receiver Balance After Transaction",
        min_value=0.0,
        value=active_example["newbalanceDest"] if active_example else 0.0,
        help="The receiver's account balance right after this transaction."
    )

type_options = list(le.classes_)
default_type_idx = 0
if active_example:
    try:
        default_type_idx = type_options.index(active_example["type"])
    except ValueError:
        default_type_idx = 0

type_choice = st.selectbox(
    "Transaction Type",
    type_options,
    index=default_type_idx,
    help="TRANSFER = sending money to another person. CASH_OUT = withdrawing money from an agent."
)
type_code = le.transform([type_choice])[0]

# Computed features
is_balance_emptied = 1 if (oldbalanceOrg == amount and amount > 0) else 0
dest_error = newbalanceDest - oldbalanceDest - amount

st.markdown("---")

# --- Analyze button ---
if st.button("Analyze Transaction", type="primary", use_container_width=True):
    feature_values = {
        'amount': amount,
        'oldbalanceOrg': oldbalanceOrg,
        'newbalanceDest': newbalanceDest,
        'oldbalanceDest': oldbalanceDest,
        'type_code': type_code,
        'is_balance_emptied': is_balance_emptied,
        'dest_error': dest_error
    }
    features = np.array([[feature_values[f] for f in feature_names]])
    probability = model.predict_proba(features)[0][1]

    st.markdown("---")
    st.subheader("Result")

    result_col1, result_col2, result_col3 = st.columns(3)
    with result_col1:
        st.metric("Risk Score", f"{probability:.2%}")
    with result_col2:
        st.metric("Threshold", f"{optimal_threshold:.2%}")
    with result_col3:
        st.metric("Account Emptied", "Yes" if is_balance_emptied else "No")

    if probability > optimal_threshold:
        st.error(f"FRAUD DETECTED - This transaction has a {probability:.2%} probability of being fraudulent.")
    else:
        st.success(f"LEGITIMATE - This transaction has a {probability:.2%} probability of being fraudulent.")

    # Show what the model sees
    with st.expander("What the model analyzed"):
        detail_col1, detail_col2 = st.columns(2)
        with detail_col1:
            st.write(f"**Amount:** {amount:,.2f}")
            st.write(f"**Sender Balance Before:** {oldbalanceOrg:,.2f}")
            st.write(f"**Sender Balance After:** {oldbalanceOrg - amount:,.2f}")
            st.write(f"**Account Emptied:** {'Yes' if is_balance_emptied else 'No'}")
        with detail_col2:
            st.write(f"**Receiver Balance Before:** {oldbalanceDest:,.2f}")
            st.write(f"**Receiver Balance After:** {newbalanceDest:,.2f}")
            st.write(f"**Destination Error:** {dest_error:,.2f}")
            st.write(f"**Type:** {type_choice} (code: {type_code})")
        if dest_error != 0:
            st.warning(f"The receiver balance changed by {newbalanceDest - oldbalanceDest:,.2f} but should have changed by {amount:,.2f}. This mismatch (dest_error = {dest_error:,.2f}) is suspicious.")
