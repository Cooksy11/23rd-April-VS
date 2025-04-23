import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import requests
from fpdf import FPDF
from datetime import datetime

# -------------------- Page config --------------------
st.set_page_config(page_title="Expectancy Change Viewer", layout="wide")
st.title("📊 Expectancy Change Viewer")

st.markdown("*Favourites are determined using Goal Expectancy at the earliest available minute in each match*")

# -------------------- Load data --------------------
@st.cache_data(show_spinner=False)
def load_parquet_from_drive():
    url = "https://drive.google.com/uc?export=download&id=1IBvy-k0yCDKMynfRTQzXJAoWJpRhFPKk"
    response = requests.get(url)
    response.raise_for_status()
    df = pd.read_parquet(BytesIO(response.content))
    df['EVENT_START_TIMESTAMP'] = pd.to_datetime(df['EVENT_START_TIMESTAMP'], errors='coerce')
    return df

with st.spinner("🔄 Downloading file from Google Drive..."):
    df = load_parquet_from_drive()

st.success("✅ File downloaded successfully.")
st.success(f"✅ Parquet loaded: {df.shape[0]:,} rows")

# -------------------- Helper Functions --------------------
def compute_exp_changes(df, role):
    changes = []
    for _, group in df.groupby('SRC_EVENT_ID'):
        base_minute = group['MINUTES'].min()
        base_row = group[group['MINUTES'] == base_minute]
        if base_row.empty:
            continue

        base = base_row.iloc[0]

        if role == "Favourite Goals":
            fav = 'HOME' if base['GOAL_EXP_HOME'] > base['GOAL_EXP_AWAY'] else 'AWAY'
            base_val = base[f'GOAL_EXP_{fav}']
            group_val = group[f'GOAL_EXP_{fav}']
            label = f"{role}"

        elif role == "Underdog Goals":
            dog = 'HOME' if base['GOAL_EXP_HOME'] < base['GOAL_EXP_AWAY'] else 'AWAY'
            base_val = base[f'GOAL_EXP_{dog}']
            group_val = group[f'GOAL_EXP_{dog}']
            label = f"{role}"

        elif role == "Total Goals":
            base_val = base['GOAL_EXP_HOME'] + base['GOAL_EXP_AWAY']
            group_val = group['GOAL_EXP_HOME'] + group['GOAL_EXP_AWAY']
            label = "Total Goals"

        elif role == "Favourite Corners":
            fav = 'HOME' if base['GOAL_EXP_HOME'] > base['GOAL_EXP_AWAY'] else 'AWAY'
            base_val = base[f'CORNERS_EXP_{fav}']
            group_val = group[f'CORNERS_EXP_{fav}']
            label = f"{role}"

        elif role == "Underdog Corners":
            dog = 'HOME' if base['GOAL_EXP_HOME'] < base['GOAL_EXP_AWAY'] else 'AWAY'
            base_val = base[f'CORNERS_EXP_{dog}']
            group_val = group[f'CORNERS_EXP_{dog}']
            label = f"{role}"

        elif role == "Total Corners":
            base_val = base['CORNERS_EXP_HOME'] + base['CORNERS_EXP_AWAY']
            group_val = group['CORNERS_EXP_HOME'] + group['CORNERS_EXP_AWAY']
            label = "Total Corners"

        elif role == "Favourite Yellow":
            fav = 'HOME' if base['GOAL_EXP_HOME'] > base['GOAL_EXP_AWAY'] else 'AWAY'
            base_val = base[f'YELLOW_CARDS_EXP_{fav}']
            group_val = group[f'YELLOW_CARDS_EXP_{fav}']
            label = f"{role}"

        elif role == "Underdog Yellow":
            dog = 'HOME' if base['GOAL_EXP_HOME'] < base['GOAL_EXP_AWAY'] else 'AWAY'
            base_val = base[f'YELLOW_CARDS_EXP_{dog}']
            group_val = group[f'YELLOW_CARDS_EXP_{dog}']
            label = f"{role}"

        elif role == "Total Yellow":
            base_val = base['YELLOW_CARDS_EXP_HOME'] + base['YELLOW_CARDS_EXP_AWAY']
            group_val = group['YELLOW_CARDS_EXP_HOME'] + group['YELLOW_CARDS_EXP_AWAY']
            label = "Total Yellow"

        for _, row in group.iterrows():
            changes.append({
                'Time Band': f"{int(row['MINUTES'] // 5 * 5)}-{int(row['MINUTES'] // 5 * 5 + 5)}",
                'Change': row_val := row_val if pd.isna((row_val := row.get(group_val.name))) else row_val - base_val,
                'Type': label
            })

    return pd.DataFrame(changes)

# -------------------- Sidebar Filters --------------------
st.sidebar.header("Filters")
exp_options = [
    "Favourite Goals", "Underdog Goals", "Total Goals",
    "Favourite Corners", "Underdog Corners", "Total Corners",
    "Favourite Yellow", "Underdog Yellow", "Total Yellow"
]
selected_types = st.sidebar.multiselect("Select Expectancy Types (up to 6)", exp_options, default=exp_options[:2], max_selections=6)

# -------------------- Main Graphing --------------------
if selected_types:
    cols = st.columns(min(len(selected_types), 3))
    for i, exp in enumerate(selected_types):
        with cols[i % len(cols)]:
            df_changes = compute_exp_changes(df, exp)
            avg_change = df_changes.groupby("Time Band")['Change'].mean()

            fig, ax = plt.subplots()
            ax.plot(avg_change.index, avg_change.values, marker='o')
            ax.set_title(f"{exp} Expectancy Change")
            ax.set_xlabel("Time Band (Minutes)")
            ax.set_ylabel("Avg Change")
            st.pyplot(fig, use_container_width=True)

else:
    st.warning("Please select at least one expectancy type to display.")
