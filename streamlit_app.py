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
    plot_individual_axes
)

# Page configuration
st.set_page_config(
    page_title="Acceleration Analysis",
    page_icon="📊",
    layout="wide"
)

st.title("⚙️ Acceleration Data Analysis")
st.write("Upload acceleration data (CSV) and analyse the power spectral density using Welch's method.")

# Display supported formats info
with st.expander("📋 Supported CSV Formats"):
    st.write("""
    Your CSV file should contain acceleration data with columns for X, Y, and Z axes.
    The code automatically detects columns named 'x', 'y', 'z' (case-insensitive) regardless of their position.
    
    **Example formats:**
    - 3 columns: `x, y, z`
    - 4 columns: `time, x, y, z`
    - 5 columns: `time, status, x, y, z`
    
    The code will extract the x, y, z data and skip any rows with missing/non-numeric values.
    """)

# Sidebar for parameters
st.sidebar.header("Analysis Parameters")
fs = st.sidebar.number_input("Sampling Rate (Hz)", value=5000, min_value=100, max_value=100000)
nperseg = st.sidebar.slider("Welch Segment Length", value=1024, min_value=256, max_value=4096, step=256)
noverlap = st.sidebar.slider("Welch Overlap Points", value=50, min_value=0, max_value=200)
x_limit = st.sidebar.number_input("Frequency X-axis Limit (Hz)", value=1000, min_value=10, max_value=10000)

# File uploader
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
        
        # Display file info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Number of Samples", len(Accelx))
        with col2:
            st.metric("Duration (s)", f"{len(Accelx)/fs:.2f}")
        with col3:
            st.metric("CSV Columns", colnum)
        
        st.success("✅ Data loaded successfully!")
        
        # Show preview of parsed data
        with st.expander("📊 Data Preview"):
            preview_df = pd.DataFrame({
                'X': Accelx[:10],
                'Y': Accely[:10],
                'Z': Accelz[:10]
            })
            st.dataframe(preview_df)
        
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
        tab1, tab2 = st.tabs(["Average PSD", "Individual Axes"])
        
        with tab1:
            st.pyplot(plot_psd(results, x_limit=x_limit))
        
        with tab2:
            st.pyplot(plot_individual_axes(results, x_limit=x_limit))
        
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
        
    except ValueError as e:
        st.error(f"❌ Error: {str(e)}")
        st.info("💡 **Expected CSV Format:** Your CSV should contain columns named 'x', 'y', and 'z' (or similar variations). "
                "The code will automatically find these columns regardless of their position.\n\n"
                "**Troubleshooting:**\n"
                "- Check that rows 6+ have the same number of columns as the header\n"
                "- Remove any empty rows or rows with extra/missing values\n"
                "- Ensure all data values are numeric or empty (no text in data columns)")
    except Exception as e:
        st.error(f"❌ An unexpected error occurred: {str(e)}")
        st.info("💡 **Tips for CSV files:**\n"
                "- Make sure columns are named 'x', 'y', 'z' or contain these letters\n"
                "- Ensure all data rows have consistent formatting\n"
                "- Check for inconsistent delimiters (commas, semicolons, tabs)\n"
                "- Remove any special characters in headers if possible")
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
else:
    st.info("👆 Please upload a CSV file to get started")
