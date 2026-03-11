import streamlit as st
import pandas as pd
import plotly.express as px
from utils.preprocessing import atm_locations 

st.title("🗺 ATM Demand Map")

df = pd.read_csv("data/atm_data.csv")


atm_locations["ATM_ID"] = atm_locations["atm_number"].map(lambda n: f"ATM_{n:04d}")

# Attach real coordinates to every row in the time series.
df = df.merge(atm_locations[["ATM_ID", "location", "lat", "lon"]], on="ATM_ID", how="left")

missing_coords = int(df["lat"].isna().sum())
if missing_coords:
    st.warning(
        f"{missing_coords:,} rows are missing coordinates (ATM_ID not in 1–50 list). "
        "Those points will not be plotted."
    )
df = df.dropna(subset=["lat", "lon"])

# Plot one marker per ATM (latest record in the file).
if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.sort_values(["ATM_ID", "Date"]).groupby("ATM_ID", as_index=False).tail(1)

fig = px.scatter_mapbox(
    df,
    lat="lat",
    lon="lon",
    color="Cash_Demand_Next_Day",
    size="Cash_Demand_Next_Day",
    hover_name="ATM_ID",
    hover_data={"location": True, "Cash_Demand_Next_Day": True, "lat": False, "lon": False},
    zoom=10,
)

fig.update_layout(mapbox_style="open-street-map")

st.plotly_chart(fig, use_container_width=True)
