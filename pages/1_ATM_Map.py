import streamlit as st
import pandas as pd
import plotly.express as px

st.title("🗺 ATM Demand Map")

df = pd.read_csv("data/atm_data.csv")

atm_locations = pd.DataFrame(
    [
        (1, "Colaba ATM", 18.9067, 72.8147),
        (2, "Nariman Point ATM", 18.9256, 72.8233),
        (3, "Churchgate ATM", 18.9322, 72.8264),
        (4, "Marine Lines ATM", 18.9431, 72.8248),
        (5, "Charni Road ATM", 18.9519, 72.8186),
        (6, "Grant Road ATM", 18.9612, 72.8135),
        (7, "Mumbai Central ATM", 18.9690, 72.8190),
        (8, "Mahalaxmi ATM", 18.9823, 72.8113),
        (9, "Lower Parel ATM", 18.9985, 72.8311),
        (10, "Dadar West ATM", 19.0183, 72.8424),
        (11, "Dadar East ATM", 19.0186, 72.8448),
        (12, "Matunga ATM", 19.0272, 72.8570),
        (13, "Sion ATM", 19.0412, 72.8610),
        (14, "Kurla ATM", 19.0728, 72.8826),
        (15, "Bandra West ATM", 19.0596, 72.8295),
        (16, "Bandra East ATM", 19.0615, 72.8415),
        (17, "Khar ATM", 19.0693, 72.8397),
        (18, "Santacruz West ATM", 19.0810, 72.8347),
        (19, "Santacruz East ATM", 19.0816, 72.8484),
        (20, "Vile Parle West ATM", 19.1004, 72.8360),
        (21, "Vile Parle East ATM", 19.1008, 72.8510),
        (22, "Andheri West ATM", 19.1197, 72.8464),
        (23, "Andheri East ATM", 19.1136, 72.8697),
        (24, "Jogeshwari ATM", 19.1348, 72.8481),
        (25, "Goregaon West ATM", 19.1550, 72.8376),
        (26, "Goregaon East ATM", 19.1643, 72.8606),
        (27, "Malad West ATM", 19.1870, 72.8420),
        (28, "Malad East ATM", 19.1864, 72.8679),
        (29, "Kandivali West ATM", 19.2052, 72.8347),
        (30, "Kandivali East ATM", 19.2093, 72.8701),
        (31, "Borivali West ATM", 19.2307, 72.8567),
        (32, "Borivali East ATM", 19.2290, 72.8576),
        (33, "Dahisar ATM", 19.2502, 72.8597),
        (34, "Powai ATM", 19.1176, 72.9060),
        (35, "Chandivali ATM", 19.1100, 72.8965),
        (36, "Ghatkopar West ATM", 19.0860, 72.9080),
        (37, "Ghatkopar East ATM", 19.0855, 72.9110),
        (38, "Chembur ATM", 19.0522, 72.9005),
        (39, "Govandi ATM", 19.0550, 72.9250),
        (40, "Mankhurd ATM", 19.0497, 72.9305),
        (41, "Vashi ATM", 19.0760, 72.9987),
        (42, "Sanpada ATM", 19.0610, 73.0080),
        (43, "Nerul ATM", 19.0330, 73.0297),
        (44, "Belapur ATM", 19.0184, 73.0397),
        (45, "Airoli ATM", 19.1585, 72.9997),
        (46, "Mulund West ATM", 19.1726, 72.9567),
        (47, "Mulund East ATM", 19.1745, 72.9634),
        (48, "Bhandup ATM", 19.1456, 72.9396),
        (49, "Kanjurmarg ATM", 19.1309, 72.9314),
        (50, "Thane ATM", 19.2183, 72.9781),
    ],
    columns=["atm_number", "location", "lat", "lon"],
)

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
