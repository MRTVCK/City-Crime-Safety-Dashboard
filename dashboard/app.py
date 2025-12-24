"""Main Streamlit Crime Dashboard Application - Production Ready."""
import streamlit as st
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

# Import styling
from dashboard.styles.custom_css import apply_custom_css

# Import components
from dashboard.components.sidebar import render_title_and_map_key
from dashboard.components.info_boxes import render_predictions_callout
from dashboard.components.filters import render_filters
from dashboard.components.map_view import render_crime_map

# Import data and visualization functions
from src.data_loader import (
    load_crime_data, 
    process_datetime_columns, 
    apply_filters, 
    get_filter_options
)
from src.visualizations import (
    create_hourly_chart, 
    create_day_of_week_chart, 
    create_monthly_trend_chart,
    create_top_crimes_chart
)

# ========== PAGE CONFIGURATION ==========
st.set_page_config(
    page_title="City Crime & Safety Dashboard", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Apply custom cyberpunk theme
apply_custom_css()

# ========== DATA LOADING ==========
df = load_crime_data()
if df.empty:
    st.error("⚠️ No data available. Please check data files in data/processed/")
    st.stop()

df = process_datetime_columns(df)
filter_options = get_filter_options(df)

# ========== TOP ROW LAYOUT ==========
top_left, top_center, top_right = st.columns([1, 2.5, 1])

# Left column: Dashboard title and map key
with top_left:
    render_title_and_map_key()

# Right column: Filters and predictions callout
with top_right:
    selected_filters = render_filters(filter_options)
    render_predictions_callout()

# Apply user-selected filters
fdf = apply_filters(
    df,
    years=selected_filters["years"],
    crime_types=selected_filters["crime_types"],
    neighborhoods=selected_filters["neighborhoods"],
    arrest_made=selected_filters["arrest_made"]
)

# Center column: Crime hotspot map
with top_center:
    render_crime_map(fdf)

# ========== ANALYTICS SECTION ==========
st.divider()
st.markdown(f"### 📊 {len(fdf):,} incidents analyzed")

# Create tabs for different visualizations - PREDICT FIRST!
tab_predict, tab_hour, tab_dow, tab_month, tab_crimes = st.tabs([
    "🤖 PREDICT",
    "⏰ HOUR", 
    "📅 DAY OF WEEK", 
    "📈 MONTHLY TREND", 
    "🚨 TOP CRIMES"
])

# ML predictions (first tab - auto-selected on load!)
with tab_predict:
    st.subheader("🤖 Machine Learning Crime Predictions")
    
    # Lazy import - only loads ML libraries when this tab is clicked
    from dashboard.components.predictions import render_ml_predictions
    render_ml_predictions(fdf)

# Hourly analysis
with tab_hour:
    st.subheader("Crime Distribution by Hour of Day")
    fig = create_hourly_chart(fdf)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("⚠️ Hourly data not available in current dataset.")

# Day of week analysis
with tab_dow:
    st.subheader("Crime Distribution by Day of Week")
    fig = create_day_of_week_chart(fdf)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("⚠️ Day of week data not available in current dataset.")

# Monthly trend analysis
with tab_month:
    st.subheader("Crime Trends Over Time")
    fig = create_monthly_trend_chart(fdf)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("⚠️ Time series data not available in current dataset.")

# Top crime types
with tab_crimes:
    st.subheader("Most Common Crime Types")
    fig = create_top_crimes_chart(fdf)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("⚠️ Crime type data not available in current dataset.")

# ========== FOOTER ==========
st.divider()

footer_left, footer_right = st.columns([3, 1])

with footer_left:
    st.markdown("_Developed by **Destin 'TUCK' Tucker** • Data Science Portfolio Project_")
    st.caption("📊 Data Source: [LA Open Data Portal](https://data.lacity.org/)")

with footer_right:
    st.download_button(
        label="⬇️ Download Filtered Data",
        data=fdf.to_csv(index=False).encode("utf-8"),
        file_name="filtered_crime_data.csv",
        mime="text/csv",
        help="Download the currently filtered dataset as CSV"
    )