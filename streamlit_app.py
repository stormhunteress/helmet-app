import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import tempfile
import os

from acceleration_analysis import (
    load_acceleration_data,
    compute_welch_psd,
    plot_psd,
    plot_individual_axes,
    acquire_from_hardware,
    plot_time_domain,
    NIDAQMX_AVAILABLE
)

# Page configuration
st.set_page_config(
    page_title="Acceleration Analysis",
    page_icon="📊",
    layout="wide"
)

st.title("⚙️ Acceleration Data Analysis")
st.write("Analyse acceleration data from CSV or acquire directly from NI-DAQ hardware.")

# Sidebar for parameters
st.sidebar.header("Analysis Parameters")
fs = st.sidebar.number_input("Sampling Rate (Hz)", value=5000, min_value=100, max_value=100000)
nperseg = st.sidebar.slider("Welch Segment Length", value=1024, min_value=256, max_value=4096, step=256)
noverlap = st.sidebar.slider("Welch Overlap Points", value=50, min_value=0, max_value=200)
x_limit = st.sidebar.number_input("Frequency X-axis Limit (Hz)", value=1000, min_value=10, max_value=10000)

# Data source selection
st.sidebar.header("Data Source")
data_source = st.sidebar.radio(
    "Choose data source:",
    ["Upload CSV", "Hardware Acquisition"] if NIDAQMX_AVAILABLE else ["Upload CSV"]
)

# Initialize data variables
Accelx = None
Accely = None
Accelz = None
data_loaded = False

if data_source == "Hardware Acquisition" and NIDAQMX_AVAILABLE:
    st.subheader("🔧 Hardware Acquisition Settings")
    col1, col2 = st.columns(2)
    
    with col1:
        duration = st.number_input("Acquisition Duration (s)", value=10, min_value=1, max_value=60)
        f_start = st.number_input("Sweep Start Frequency (Hz)", value=400, min_value=10)
    
    with col2:
        f_end = st.number_input("Sweep End Frequency (Hz)", value=1000, min_value=10)
        channel_name = st.text_input("NI-DAQ Channel", value="Dev1/ai0:2")
    
    if st.button("🚀 Start Acquisition", type="primary"):
        try:
            with st.spinner("Acquiring data from hardware..."):
                Accelx, Accely, Accelz = acquire_from_hardware(
                    duration=duration,
                    fs=fs,
                    f_start=f_start,
                    f_end=f_end,
                    channel_name=channel_name
                )
                data_loaded = True
                st.success("✅ Data acquired successfully!")
        except Exception as e:
            st.error(f"❌ Acquisition failed: {str(e)}")

else:
    # File uploader for CSV
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Create temporary file to work with
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_path = tmp_file.name
        
        try:
            # Load data
            st.info("Loading acceleration data...")
            Accelx, Accely, Accelz, colnum = load_acceleration_data(tmp_path)
            data_loaded = True
            
            # Display file info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Number of Samples", len(Accelx))
            with col2:
                st.metric("Duration (s)", f"{len(Accelx)/fs:.2f}")
            with col3:
                st.metric("CSV Columns", colnum)
            
            st.success("✅ Data loaded successfully!")
        
        except ValueError as e:
            st.error(f"❌ Error: {str(e)}")
            data_loaded = False
        except Exception as e:
            st.error(f"❌ An unexpected error occurred: {str(e)}")
            data_loaded = False
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

if data_loaded and Accelx is not None:
        
    # Compute Welch PSD
    st.info("Computing power spectral density...")
    results = compute_welch_psd(Accelx, Accely, Accelz, fs=fs, nperseg=nperseg, noverlap=noverlap)
    
    # Display results
    st.subheader("📈 Analysis Results")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Mode 1 Frequency", f"{results['mode_1_freq']:.2f} Hz", 
                 delta=f"Index: {results['mode_1_index']}")
    
    with col2:
        st.metric("Peak PSD Value", f"{abs(results['psd_total'][results['mode_1_index']]):.6e}")
    
    # Plot tabs
    tab1, tab2, tab3 = st.tabs(["Average PSD", "Individual Axes", "Time Domain"])
        
    with tab1:
        st.pyplot(plot_psd(results, x_limit=x_limit))
    
    with tab2:
        st.pyplot(plot_individual_axes(results, x_limit=x_limit))
    
    with tab3:
        st.pyplot(plot_time_domain(Accelx, Accely, Accelz, fs=fs))
    
    # Data export
    st.subheader("📊 Export Results")
    col1, col2 = st.columns(2)
    
    with col1:
        # Export frequency and PSD data
        export_df = pd.DataFrame({
            'Frequency (Hz)': results['frequency'],
            'PSD X (V²/Hz)': results['psd_x'],
            'PSD Y (V²/Hz)': results['psd_y'],
            'PSD Z (V²/Hz)': results['psd_z'],
            'PSD Total (V²/Hz)': results['psd_total']
        })
        
        csv_data = export_df.to_csv(index=False)
        st.download_button(
            label="Download PSD Data (CSV)",
            data=csv_data,
            file_name="psd_results.csv",
            mime="text/csv"
        )
    
    with col2:
        # Export summary statistics
        summary_df = pd.DataFrame({
            'Parameter': ['Mode 1 Frequency (Hz)', 'Sampling Rate (Hz)', 'Num Samples', 'Duration (s)'],
            'Value': [
                f"{results['mode_1_freq']:.2f}",
                str(fs),
                str(len(Accelx)),
                f"{len(Accelx)/fs:.2f}"
            ]
        })
        
        summary_csv = summary_df.to_csv(index=False)
        st.download_button(
            label="Download Summary (CSV)",
            data=summary_csv,
            file_name="analysis_summary.csv",
            mime="text/csv"
        )
    
    st.success("✅ Analysis complete!")

else:
    if data_source == "Upload CSV":
        st.info("👆 Please upload a CSV file to get started")
    else:
        st.info("Configure hardware settings and click 'Start Acquisition' to begin")
