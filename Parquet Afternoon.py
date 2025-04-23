import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF
import tempfile
import requests
import os
from io import BytesIO

# ---------------------- CONFIGURATION ----------------------
st.set_page_config(layout="wide")
st.title("📊 Expectancy Change Viewer")

# ---------------------- FILE LOADING ----------------------
@st.cache_data(show_spinner=True)
def load_parquet_from_gdrive(url):
    file_id = url.split("/d/")[-1].split("/")[0]
    download_url = f"https://drive.google.com/uc?id={file_id}"
    response = requests.get(download_url)
    response.raise_for_status()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".parquet") as tmp_file:
        tmp_file.write(response.content)
        tmp_path = tmp_file.name
    df = pd.read_parquet(tmp_path)
    os.remove(tmp_path)
    return df

st.markdown("<sub>*Favourites are determined using Goal Expectancy at the earliest available minute in each match</sub>", unsafe_allow_html=True)

# Google Drive Parquet file URL
gdrive_url = "https://drive.google.com/file/d/1IBvy-k0yCDKMynfRTQzXJAoWJpRhFPKk/view?usp=sharing"

with st.spinner("🔄 Loading Parquet file..."):
    df = load_parquet_from_gdrive(gdrive_url)
    st.success("✅ File downloaded successfully.")
    st.success(f"✅ Parquet loaded: {df.shape[0]:,} rows")

# Date parsing
if 'EVENT_START_TIMESTAMP' in df.columns:
    df['EVENT_START_TIMESTAMP'] = pd.to_datetime(df['EVENT_START_TIMESTAMP'], errors='coerce', dayfirst=True)
    st.success("🗓️ Datetime parsing completed.")

# ---------------------- EXPECTANCY OPTIONS ----------------------
exp_options = [
    "Favourite Goals", "Underdog Goals", "Total Goals",
    "Favourite Corners", "Underdog Corners", "Total Corners",
    "Favourite Yellow", "Underdog Yellow", "Total Yellow"
]

# ---------------------- FILTERS ----------------------
st.sidebar.header("🔎 Filters")
exp_selected = st.sidebar.multiselect("Select Expectancy Types (up to 6)", exp_options, max_selections=6)
date_range = st.sidebar.date_input("Select Date Range", [])
fav_levels = st.sidebar.multiselect("Goal Favouritism Level", ["Strong Favourite", "Medium Favourite", "Slight Favourite"])
scoreline_filter = st.sidebar.multiselect("Goal Scoreline Filter", ["Favourite Winning", "Scores Level", "Underdog Winning"])

# ---------------------- PROCESSING ----------------------
@st.cache_data(show_spinner=True)
def compute_exp_change(df, exp_type):
    incident = "Goals" if "Goals" in exp_type else ("Corners" if "Corners" in exp_type else "Yellow")
    home_col = f"{incident.upper()}_EXP_HOME"
    away_col = f"{incident.upper()}_EXP_AWAY"

    df_filtered = df.dropna(subset=[home_col, away_col, 'MINUTES'])

    df_sorted = df_filtered.sort_values(['SRC_EVENT_ID', 'MINUTES'])
    base_exp = df_sorted.groupby('SRC_EVENT_ID').first().reset_index()

    base_exp['FAVOURITE'] = np.where(base_exp[home_col] > base_exp[away_col], 'Home', 'Away')
    df_merged = df_sorted.merge(base_exp[['SRC_EVENT_ID', 'FAVOURITE', home_col, away_col]], on='SRC_EVENT_ID', suffixes=('', '_BASE'))

    rows = []
    for _, row in df_merged.iterrows():
        fav = row['FAVOURITE']
        base_home = row[f'{home_col}_BASE']
        base_away = row[f'{away_col}_BASE']

        if "Favourite" in exp_type:
            col = home_col if fav == 'Home' else away_col
            base = base_home if fav == 'Home' else base_away
            row_val = row[col]
        elif "Underdog" in exp_type:
            col = away_col if fav == 'Home' else home_col
            base = base_away if fav == 'Home' else base_home
            row_val = row[col]
        else:  # Total
            home_val = row[home_col]
            away_val = row[away_col]
            row_val = home_val + away_val if pd.notna(home_val) and pd.notna(away_val) else np.nan
            base = base_home + base_away if pd.notna(base_home) and pd.notna(base_away) else np.nan

        if pd.isna(row_val) or pd.isna(base):
            change = np.nan
        else:
            change = row_val - base

        rows.append({
            'SRC_EVENT_ID': row['SRC_EVENT_ID'],
            'Time Band': f"{int(row['MINUTES']//5)*5}-{int(row['MINUTES']//5)*5 + 5}",
            'Change': change
        })

    return pd.DataFrame(rows)

# ---------------------- PLOTTING ----------------------
def plot_exp_change(df_changes, title):
    avg_change = df_changes.groupby('Time Band')['Change'].mean()
    fig, ax = plt.subplots(figsize=(6, 4))
    avg_change.plot(marker='o', ax=ax)
    ax.set_ylabel("Avg Change")
    ax.set_xlabel("Time Band (Minutes)")
    ax.set_title(f"{title} Expectancy Change")
    ax.grid(True)
    return fig

# ---------------------- MAIN APP ----------------------
if exp_selected:
    layout_cols = st.columns(min(3, len(exp_selected)))
    plots = []
    for i, exp in enumerate(exp_selected):
        df_change = compute_exp_change(df, exp)
        fig = plot_exp_change(df_change, exp)
        with layout_cols[i % len(layout_cols)]:
            st.pyplot(fig, use_container_width=True)
        plots.append(fig)

    # PDF Export
    if st.button("📥 Download All Charts as PDF"):
        pdf = FPDF()
        for fig in plots:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                fig.savefig(tmpfile.name)
                pdf.add_page()
                pdf.image(tmpfile.name, x=10, y=10, w=180)
                os.unlink(tmpfile.name)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            pdf.output(f.name)
            with open(f.name, "rb") as f_pdf:
                st.download_button("📄 Download PDF", f_pdf.read(), "expectancy_charts.pdf")
else:
    st.warning("Please select at least one expectancy type to display charts.")
