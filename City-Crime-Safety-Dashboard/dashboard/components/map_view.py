"""Map visualization component."""
import streamlit as st
import folium
from streamlit_folium import st_folium
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

# Import map utilities
scripts_path = project_root / "scripts"
sys.path.append(str(scripts_path))
from geo_utils import add_heatmap, add_clusters

from config.settings import MAX_MAP_POINTS, DEFAULT_ZOOM, MAP_TILES


def render_map_header():
    """Render the map header with styling."""
    st.markdown("""
    <div style='text-align: center; border: 2px solid #00d4ff; padding: 10px; 
                border-radius: 10px; background: rgba(0,30,60,0.6); 
                box-shadow: 0 0 20px rgba(0,212,255,0.4); margin-bottom: 10px;'>
        <h2 style='margin: 0; color: #00d4ff; text-shadow: 0 0 10px rgba(0,212,255,0.8);
                   font-family: "Courier New", monospace; letter-spacing: 2px;'>
            🗺️ CRIME HOTSPOT MAP
        </h2>
    </div>
    """, unsafe_allow_html=True)


def render_crime_map(df):
    """
    Render interactive crime map with heatmap and clusters.
    
    Args:
        df: Filtered DataFrame with latitude/longitude columns
    """
    render_map_header()
    
    if (
        {"latitude", "longitude"}.issubset(df.columns)
        and not df[["latitude", "longitude"]].dropna().empty
    ):
        # Prepare map data
        map_df = df.dropna(subset=["latitude", "longitude"]).copy()
        
        # Sample if too many points
        if len(map_df) > MAX_MAP_POINTS:
            map_df = map_df.sample(n=MAX_MAP_POINTS, random_state=42)
            st.caption(f"⚠️ Showing {MAX_MAP_POINTS:,} of {len(df):,} incidents for performance")
        
        # Get center coordinates
        lat = map_df["latitude"].median()
        lon = map_df["longitude"].median()
        
        # Create map with dark theme
        m = folium.Map(
            location=[lat, lon], 
            zoom_start=DEFAULT_ZOOM, 
            tiles=None,  # Remove default tiles to add custom dark tiles
            prefer_canvas=True
        )
        
        # Add dark blue tiles
        folium.TileLayer(
            tiles='https://{s}.basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}{r}.png',
            attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
            name='Dark Matter',
            control=False,
            opacity=0.8
        ).add_to(m)
        
        try:
            # Add heatmap layer
            m = add_heatmap(m, map_df)
            
            # Add cluster markers
            tooltip_cols = [
                c for c in ["datetime", "crime_type", "neighborhood", "zip_code", "arrest_made"]
                if c in map_df.columns
            ]
            m = add_clusters(m, map_df, tooltip_cols=tooltip_cols)
            
        except Exception as e:
            st.error(f"Map error: {e}")
        
        # Render map
        st_folium(m, width=None, height=400)
        
    else:
        st.info("No latitude/longitude columns found or no data available.")