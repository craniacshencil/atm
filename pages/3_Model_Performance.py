import streamlit as st
import pandas as pd
import plotly.express as px

st.title("📊 Model Performance")
st.caption("MAE and RMSE: lower is better.")

data = {
    "Model": [
        "Linear Regression",
        "Random Forest",
        "XGBoost",
        "LSTM",
        "CNN",
        "Voting Ensemble",
    ],
    "MAE": [
        5135.973519,
        5355.320151,
        5363.322266,
        16276.544677,
        17616.109265,
        7889.416226,
    ],
    "RMSE": [
        6210.410998,
        6550.832419,
        6550.538298,
        20145.660571,
        22114.438053,
        9702.673992,
    ],
}

df = pd.DataFrame(data)

fig = px.bar(df, x="Model", y="MAE", title="Model MAE Comparison")

col1, col2 = st.columns([2, 1])
with col1:
    st.plotly_chart(fig, use_container_width=True)
with col2:
    st.table(df)
