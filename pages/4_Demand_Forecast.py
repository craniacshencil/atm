import pandas as pd
import plotly.express as px
import streamlit as st

from utils.predict import forecast_next_days
from utils.preprocessing import (add_rolling_features, encode_categorical,
                                 load_data, atm_locations)

st.title("📈 ATM Demand Forecast")

df = load_data("data/atm_data.csv")

df = add_rolling_features(df)
df, encoders = encode_categorical(df)

# Convert to ATM_0001 format
atm_locations["atm_id"] = atm_locations["atm_number"].apply(
    lambda x: f"ATM_{x:04d}"
)
# Map name → ID
atm_map = dict(zip(atm_locations["location"], atm_locations["atm_id"]))
atm_name = st.selectbox(
    "Select ATM",
    sorted(atm_map.keys())
)

atm = atm_map[atm_name]

atm_df = df[df["atm_id"] == atm].copy()

if st.button("Generate 7-Day Forecast"):

    forecast = forecast_next_days(atm_df)

    forecast_df = pd.DataFrame(
        {"Day": [f"Day {i + 1}" for i in range(7)], "Predicted Demand": forecast}
    )

    st.subheader("Next 7 Days Forecast")

    st.table(forecast_df)

    fig = px.line(
        forecast_df,
        x="Day",
        y="Predicted Demand",
        markers=True,
        title="7 Day ATM Cash Demand Forecast",
    )

    st.plotly_chart(fig, use_container_width=True)
