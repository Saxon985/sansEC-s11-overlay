import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from io import BytesIO

# -------------------------------------------------
# Page settings
# -------------------------------------------------
st.set_page_config(page_title="SansEC S11 Overlay", layout="wide")
st.title("SansEC High-Temperature S11 Overlay")

# -------------------------------------------------
# Ensure data folder exists
# -------------------------------------------------
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# -------------------------------------------------
# Multiple file uploader
# -------------------------------------------------
uploaded_files = st.file_uploader(
    "Upload one or more temperature .s1p files",
    type=["s1p"],
    accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        save_path = os.path.join(DATA_DIR, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Saved: {uploaded_file.name}")

# -------------------------------------------------
# Load all existing .s1p files + extract resonant frequency
# -------------------------------------------------
files = sorted(
    glob.glob(os.path.join(DATA_DIR, "*.s1p")),
    key=lambda x: float(os.path.basename(x).replace(".s1p", ""))
)

file_info = []

for file in files:
    temp = float(os.path.basename(file).replace(".s1p", ""))
    try:
        data = np.loadtxt(file, skiprows=1)
        freq_hz = data[:, 0]
        real = data[:, 1]
        imag = data[:, 2]
        s11_db = 20 * np.log10(np.sqrt(real**2 + imag**2))
        min_idx = np.argmin(s11_db)
        resonant_mhz = freq_hz[min_idx] / 1e6
        min_s11 = s11_db[min_idx]
    except Exception:
        resonant_mhz = None
        min_s11 = None

    file_info.append({
        "Filename": os.path.basename(file),
        "Temperature (°C)": temp,
        "Resonant Freq (MHz)": round(resonant_mhz, 4) if resonant_mhz else "Error",
        "Min S11 (dB)": round(min_s11, 2) if min_s11 else "Error"
    })

st.write(f"Currently stored files: **{len(files)}**")

# -------------------------------------------------
# Feature 1 & 3: Table of files + resonant frequencies
# -------------------------------------------------
if file_info:
    df = pd.DataFrame(file_info)
    st.dataframe(df, use_container_width=True)

    # -------------------------------------------------
    # Feature 2: Delete selected files
    # -------------------------------------------------
    st.subheader("Delete files")
    files_to_delete = st.multiselect(
        "Select files to delete",
        options=[info["Filename"] for info in file_info]
    )
    if st.button("Delete selected files") and files_to_delete:
        for fname in files_to_delete:
            os.remove(os.path.join(DATA_DIR, fname))
        st.success(f"Deleted: {', '.join(files_to_delete)}")
        st.rerun()

# -------------------------------------------------
# Plot
# -------------------------------------------------
if len(files) == 0:
    st.info("No .s1p files yet. Upload one or more to get started.")
else:
    fig, ax = plt.subplots(figsize=(12, 6))

    temps = [float(os.path.basename(f).replace(".s1p", "")) for f in files]
    cmap = plt.cm.plasma
    norm = Normalize(vmin=min(temps), vmax=max(temps))

    for file in files:
        temp = float(os.path.basename(file).replace(".s1p", ""))
        data = np.loadtxt(file, skiprows=1)
        freq_mhz = data[:, 0] / 1e6
        real = data[:, 1]
        imag = data[:, 2]
        s11_db = 20 * np.log10(np.sqrt(real**2 + imag**2))
        ax.plot(freq_mhz, s11_db, color=cmap(norm(temp)), linewidth=0.8, alpha=0.85)

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Temperature (°C)")

    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("S11 Magnitude (dB)")
    ax.set_title("HighTempOven S11 Overlay")
    ax.grid(True, alpha=0.3)

    st.pyplot(fig)

    # -------------------------------------------------
    # Feature 4: Download the plot
    # -------------------------------------------------
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    st.download_button(
        label="Download plot as PNG",
        data=buf.getvalue(),
        file_name="s11_overlay.png",
        mime="image/png"
    )