import streamlit as st
import pandas as pd
import plotly.express as px

from utils.preprocessing import (
    load_data,
    add_rolling_features,
    encode_categorical,
    features
)

from utils.predict import (
    predict_single,
    predict_sequence,
    ensemble_prediction
)

st.title("🏧 ATM Details & Prediction")

# Load data
df = load_data("data/atm_data.csv")

df = add_rolling_features(df)
df, encoders = encode_categorical(df)

# ATM selector
atm = st.selectbox(
    "Select ATM",
    sorted(df["atm_id"].unique())
)

atm_df = df[df["atm_id"] == atm].copy()

latest = atm_df.iloc[-1]

# ATM info
st.subheader("ATM Information")

col1, col2, col3 = st.columns(3)

location_text = encoders["location_type"].inverse_transform(
    [int(latest["location_type"])]
)[0]

weather_text = encoders["weather_condition"].inverse_transform(
    [int(latest["weather_condition"])]
)[0]

col1.metric("Location Type", location_text)
col2.metric("Weather", weather_text)
col3.metric("Nearby Competitors", int(latest["nearby_competitor_atms"]))

col4, col5 = st.columns(2)

col4.metric("Previous Cash Level", f"₹{latest['previous_day_cash_level']:,.0f}")
col5.metric("Withdrawals Today", f"₹{latest['total_withdrawals']:,.0f}")

# -----------------------------
# MODEL PREDICTIONS
# -----------------------------

X = latest[features].values.reshape(1, -1)

lr_pred, rf_pred, xgb_pred = predict_single(X)

# sequence prediction
seq_data = atm_df[features].values[-30:]
seq_data = seq_data.reshape(1, 30, len(features))

lstm_pred, cnn_pred = predict_sequence(seq_data)

final_pred = ensemble_prediction(
    lr_pred,
    rf_pred,
    xgb_pred,
    lstm_pred,
    cnn_pred
)

st.subheader("Model Predictions")

pred_df = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Random Forest",
        "XGBoost",
        "LSTM",
        "CNN"
    ],
    "Prediction": [
        lr_pred[0],
        rf_pred[0],
        xgb_pred[0],
        lstm_pred[0],
        cnn_pred[0]
    ]
})

st.dataframe(pred_df)

st.success(f"Final Ensemble Prediction: ₹{final_pred[0]:,.0f}")

# -----------------------------
# REFILL RECOMMENDATION
# -----------------------------

cash_level = latest["previous_day_cash_level"]

st.subheader("Refill Recommendation")

if final_pred > cash_level:

    refill_amount = final_pred[0] - cash_level + 20000

    st.error(f"⚠ Refill Required: ₹{refill_amount:,.0f}")

else:

    st.success("✅ Cash level sufficient")

# -----------------------------
# DEMAND HISTORY
# -----------------------------

st.subheader("Demand History")

history = atm_df.tail(30)
history = history.loc[:, ~history.columns.duplicated()]

fig = px.line(
    history,
    x="date",
    y="cash_demand_next_day",
    title="Last 30 Days Cash Demand"
)

st.plotly_chart(fig, use_container_width=True)
