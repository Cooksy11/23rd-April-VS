import streamlit as st
import pandas as pd
import numpy as np
import requests
import zipfile
import os
from io import BytesIO
import matplotlib.pyplot as plt
from datetime import datetime
from fpdf import FPDF

# Config
st.set_page_config(layout="wide")
st.title("📊 Expectancy Change Viewer")

# Constants
GOOGLE_DRIVE_FILE_ID = "1IBvy-k0yCDKMynfRTQzXJAoWJpRhFPKk"
GOOGLE_DRIVE_DOWNLOAD_URL = f"https://drive.google.com/uc?export=download&id={GOOGLE_DRIVE_FILE_ID}"

# Helper - Download and load Parquet file
@st.cache_data(show_spinner="Downloading file from Google Drive...")
def load_parquet_from_drive():
    response = requests.get(GOOGLE_DRIVE_DOWNLOAD_URL)
    response.raise_for_status()
    with open("data.parquet", "wb") as f:
        f.write(response.content)
    df = pd.read_parquet("data.parquet")
    return df

# Load data
st.markdown("*Downloading file from Google Drive...*")
try:
    df = load_parquet_from_drive()
    st.success("✅ File downloaded successfully.")
    st.success(f"✅ Parquet loaded: {df.shape[0]:,} rows")
    df['EVENT_START_TIMESTAMP'] = pd.to_datetime(df['EVENT_START_TIMESTAMP'], errors='coerce')
    st.success("✅ Data loaded:")
    st.write(df.shape)
    st.write(df.head())
except Exception as e:
    st.error(f"❌ Error loading Parquet file: {e}")
    st.stop()

# Define expectancy types
exp_types = {
    "Favourite Goals": ("GOAL_EXP_HOME", "GOAL_EXP_AWAY"),
    "Underdog Goals": ("GOAL_EXP_HOME", "GOAL_EXP_AWAY"),
    "Total Goals": ("GOAL_EXP_HOME", "GOAL_EXP_AWAY"),
    "Favourite Corners": ("CORNERS_EXP_HOME", "CORNERS_EXP_AWAY"),
    "Underdog Corners": ("CORNERS_EXP_HOME", "CORNERS_EXP_AWAY"),
    "Total Corners": ("CORNERS_EXP_HOME", "CORNERS_EXP_AWAY"),
    "Favourite Yellow": ("YELLOW_CARDS_EXP_HOME", "YELLOW_CARDS_EXP_AWAY"),
    "Underdog Yellow": ("YELLOW_CARDS_EXP_HOME", "YELLOW_CARDS_EXP_AWAY"),
    "Total Yellow": ("YELLOW_CARDS_EXP_HOME", "YELLOW_CARDS_EXP_AWAY"),
}

# Sidebar filters
st.sidebar.header("🔍 Filters")
exp_selected = st.sidebar.multiselect("Select Expectancy Types (up to 6)", list(exp_types.keys()), max_selections=6)
date_range = st.sidebar.date_input("Select Date Range", [df['EVENT_START_TIMESTAMP'].min(), df['EVENT_START_TIMESTAMP'].max()])
score_filters = st.sidebar.multiselect("Scoreline Filter", ["Favourite Winning", "Scores Level", "Underdog Winning"])
fav_filters = st.sidebar.multiselect("Favouritism Level", ["Strong Favourite", "Medium Favourite", "Slight Favourite"])

# Identify favourite by earliest minute
@st.cache_data()
def compute_exp_change(df, exp_type):
    home_col, away_col = exp_types[exp_type]
    df_filtered = df.copy()

    # Ensure valid minute sorting
    df_filtered = df_filtered.sort_values(['SRC_EVENT_ID', 'MINUTES'])

    # Group by match
    changes = []
    for event_id, group in df_filtered.groupby('SRC_EVENT_ID'):
        first_row = group.iloc[0]
        home_exp0 = first_row[home_col]
        away_exp0 = first_row[away_col]

        if 'Favourite' in exp_type:
            if home_exp0 >= away_exp0:
                base_val = home_exp0
                exp_series = group[home_col]
            else:
                base_val = away_exp0
                exp_series = group[away_col]
        elif 'Underdog' in exp_type:
            if home_exp0 < away_exp0:
                base_val = home_exp0
                exp_series = group[home_col]
            else:
                base_val = away_exp0
                exp_series = group[away_col]
        else:
            base_val = home_exp0 + away_exp0
            exp_series = group[home_col] + group[away_col]

        for _, row in group.iterrows():
            row_val = row[exp_series.name]
            if pd.isna(row_val):
                change_val = row_val
            else:
                change_val = row_val - base_val
            changes.append({
                'SRC_EVENT_ID': event_id,
                'MINUTES': row['MINUTES'],
                'Change': change_val
            })

    df_changes = pd.DataFrame(changes)
    df_changes['Time Band'] = pd.cut(df_changes['MINUTES'], bins=np.arange(0, 95, 5), right=False, labels=[f"{i}-{i+5}" for i in range(0, 90, 5)])
    return df_changes

# Plotting
def plot_exp_change(data, exp_type):
    fig, ax = plt.subplots(figsize=(6, 4))
    avg_change = data.groupby('Time Band')['Change'].mean()
    ax.plot(avg_change.index, avg_change.values, marker='o', color='black')
    ax.set_title(f"{exp_type} Expectancy Change")
    ax.set_xlabel("Time Band (Minutes)")
    ax.set_ylabel("Avg Change")
    ax.grid(True)
    return fig

# Layout
st.markdown("*Favourites are determined using Goal Expectancy at the earliest available minute in each match*")
if exp_selected:
    layout_cols = st.columns(min(3, len(exp_selected)))
    for i, exp in enumerate(exp_selected):
        df_change = compute_exp_change(df, exp)
        fig = plot_exp_change(df_change, exp)
        with layout_cols[i % len(layout_cols)]:
            st.pyplot(fig, use_container_width=True)
else:
    st.warning("Please select at least one expectancy type to display graphs.")
