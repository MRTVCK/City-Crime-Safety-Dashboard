"""Sidebar components for dashboard title and map key."""
import streamlit as st
from datetime import datetime


def render_dashboard_title():
    """Render the main CRIME DASHBOARD title box."""
    st.markdown("""
    <div style='border: 2px solid #00d4ff; padding: 12px 15px; border-radius: 10px; 
                background: linear-gradient(135deg, rgba(0,102,204,0.2) 0%, rgba(0,136,204,0.1) 100%);
                box-shadow: 0 0 20px rgba(0,212,255,0.3); height: fit-content;'>
        <h2 style='margin: 0; color: #00d4ff; text-shadow: 0 0 10px rgba(0,212,255,0.8);
                   font-size: 1.4rem; line-height: 1.3; padding: 0;
                   letter-spacing: 2px;'>
            🚨 CRIME<br/>DASHBOARD
        </h2>
    </div>
    """, unsafe_allow_html=True)
    st.caption(f"📊 Data: LA Open Data")
    st.caption(f"🕐 Updated {datetime.now().strftime('%Y-%m-%d')}")


def render_map_key():
    """Render the map legend explaining circle colors and markers."""
    st.markdown("""
    <div style='border: 2px solid #00d4ff; padding: 10px; border-radius: 8px; 
                background: linear-gradient(135deg, rgba(0,102,204,0.2) 0%, rgba(0,136,204,0.1) 100%);
                margin-top: 10px;
                box-shadow: 0 0 20px rgba(0,212,255,0.6);'>
        <h4 style='margin: 0 0 8px 0; color: #00d4ff; font-size: 0.9rem; 
                   text-align: center; letter-spacing: 1px; font-weight: bold;
                   text-shadow: 0 0 10px rgba(0,212,255,0.8);'>
            🗺️ MAP KEY
        </h4>
        <div style='font-size: 0.75rem; color: #66b3ff; line-height: 1.6;'>
            <div style='display: flex; align-items: center; margin: 4px 0;'>
                <span style='display: inline-block; width: 12px; height: 12px; 
                             background: linear-gradient(135deg, #ff4444 0%, #ff8844 100%); 
                             border-radius: 50%; margin-right: 8px; 
                             box-shadow: 0 0 8px rgba(255,68,68,0.6);'></span>
                <span>High Crime Cluster (1000+)</span>
            </div>
            <div style='display: flex; align-items: center; margin: 4px 0;'>
                <span style='display: inline-block; width: 12px; height: 12px; 
                             background: linear-gradient(135deg, #ffaa44 0%, #ffcc66 100%); 
                             border-radius: 50%; margin-right: 8px;
                             box-shadow: 0 0 8px rgba(255,170,68,0.6);'></span>
                <span>Medium Cluster (100-999)</span>
            </div>
            <div style='display: flex; align-items: center; margin: 4px 0;'>
                <span style='display: inline-block; width: 12px; height: 12px; 
                             background: linear-gradient(135deg, #88cc44 0%, #aadd66 100%); 
                             border-radius: 50%; margin-right: 8px;
                             box-shadow: 0 0 8px rgba(136,204,68,0.6);'></span>
                <span>Low Cluster (<100)</span>
            </div>
            <div style='display: flex; align-items: center; margin: 4px 0;'>
                <span style='display: inline-block; width: 12px; height: 12px; 
                             background: linear-gradient(135deg, #0088ff 30%, #00ccff 100%); 
                             border-radius: 50%; margin-right: 8px; opacity: 0.7;
                             box-shadow: 0 0 8px rgba(0,136,255,0.6);'></span>
                <span>Blue Pin = Crime Info</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_title_and_map_key():
    """
    Render the complete sidebar with dashboard title and map key.
    
    This is the main left column component that includes:
    - CRIME DASHBOARD title box
    - Data source information
    - Map key legend
    """
    render_dashboard_title()
    st.markdown("---")
    render_map_key()