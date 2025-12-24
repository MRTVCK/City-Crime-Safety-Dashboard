"""Custom CSS styling for cyberpunk blue theme with Cyberfall font."""
import streamlit as st
from pathlib import Path
import base64

def get_font_base64():
    """Convert Cyberfall font to base64 for embedding in CSS."""
    # Your font is at: dashboard/assets/fonts/Cyberfall.otf
    font_path = Path(__file__).parent.parent / "assets" / "fonts" / "Cyberfall.otf"
    
    if font_path.exists():
        with open(font_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

def apply_custom_css():
    """Apply cyberpunk blue theme with Cyberfall font to the dashboard."""
    
    # Get the font as base64
    font_base64 = get_font_base64()
    
    # Font-face declaration
    font_face = ""
    if font_base64:
        font_face = f"""
        @font-face {{
            font-family: 'Cyberfall';
            src: url(data:font/otf;base64,{font_base64}) format('opentype');
            font-weight: normal;
            font-style: normal;
        }}
        """
        header_font = "'Cyberfall', 'Courier New', monospace"
    else:
        # Fallback if font not found
        header_font = "'Courier New', monospace"
    
    st.markdown(f"""
    <style>
        {font_face}
        
        .stApp {{
            background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0d1128 100%);
            color: #00d4ff;
        }}
        
        /* ALL HEADERS USE CYBERFALL FONT */
        h1, h2, h3, h4, h5, h6 {{
            color: #00d4ff !important;
            text-shadow: 0 0 10px rgba(0, 212, 255, 0.5);
            font-family: {header_font} !important;
            letter-spacing: 2px;
            font-weight: bold !important;
        }}
        
        /* Specific size adjustments for main headers */
        h1 {{
            font-size: 1.8rem !important;
            line-height: 1.2 !important;
            margin: 0 !important;
            padding: 5px 0 !important;
        }}
        
        h2 {{
            font-size: 1.5rem !important;
            line-height: 1.2 !important;
            margin: 0 !important;
            padding: 5px 0 !important;
        }}
        
        h3 {{
            font-size: 1.2rem !important;
            line-height: 1.2 !important;
        }}
        
        h4, h5, h6 {{
            font-size: 1rem !important;
            line-height: 1.2 !important;
        }}
        
        /* Sidebar */
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #0a1128 0%, #0f1a3d 100%);
            border-right: 2px solid #00d4ff;
        }}
        
        /* Sidebar titles also use Cyberfall */
        [data-testid="stSidebar"] h1, 
        [data-testid="stSidebar"] h2, 
        [data-testid="stSidebar"] h3 {{
            font-family: {header_font} !important;
            letter-spacing: 3px;
        }}
        
        /* Filter labels use Cyberfall */
        .stSelectbox label, .stMultiSelect label, .stSlider label {{
            font-family: {header_font} !important;
            color: #00d4ff !important;
            font-weight: bold !important;
            letter-spacing: 1px;
        }}
        
        .stMultiSelect, .stSelectbox {{
            background-color: rgba(0, 30, 60, 0.8) !important;
            border: 1px solid #0088cc !important;
            box-shadow: 0 0 10px rgba(0, 136, 204, 0.3);
        }}
        
        .stMultiSelect > div, .stSelectbox > div {{
            background-color: rgba(0, 20, 40, 0.9) !important;
            border: 1px solid #0088cc !important;
        }}
        
        /* Tabs use Cyberfall */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
            background-color: rgba(0, 20, 40, 0.6);
            border-bottom: 2px solid #00d4ff;
        }}
        
        .stTabs [data-baseweb="tab"] {{
            background-color: rgba(0, 30, 60, 0.8);
            border: 1px solid #0088cc;
            color: #00d4ff;
            font-family: {header_font} !important;
            font-weight: bold;
            letter-spacing: 1px;
            box-shadow: 0 0 5px rgba(0, 136, 204, 0.3);
        }}
        
        .stTabs [aria-selected="true"] {{
            background: linear-gradient(135deg, #0088cc 0%, #00d4ff 100%);
            color: #000 !important;
            box-shadow: 0 0 15px rgba(0, 212, 255, 0.6);
        }}
        
        hr {{
            border-color: #00d4ff !important;
            box-shadow: 0 0 5px rgba(0, 212, 255, 0.5);
        }}
        
        .css-1dp5vir, .css-16huue1, p {{
            color: #66b3ff !important;
        }}
        
        /* Buttons use Cyberfall */
        .stButton > button, .stDownloadButton > button {{
            background: linear-gradient(135deg, #0066cc 0%, #0088ff 100%);
            color: #ffffff;
            border: 1px solid #00d4ff;
            font-family: {header_font} !important;
            font-weight: bold;
            letter-spacing: 1px;
            box-shadow: 0 0 10px rgba(0, 136, 255, 0.4);
            transition: all 0.3s ease;
        }}
        
        .stButton > button:hover, .stDownloadButton > button:hover {{
            background: linear-gradient(135deg, #0088ff 0%, #00aaff 100%);
            box-shadow: 0 0 20px rgba(0, 212, 255, 0.8);
            transform: translateY(-2px);
        }}
        
        /* Metrics use Cyberfall */
        [data-testid="stMetricValue"] {{
            color: #00d4ff !important;
            text-shadow: 0 0 8px rgba(0, 212, 255, 0.6);
            font-family: {header_font} !important;
        }}
        
        [data-testid="stMetricLabel"] {{
            font-family: {header_font} !important;
            color: #66b3ff !important;
        }}
        
        .dataframe {{
            border: 1px solid #0088cc !important;
            background-color: rgba(0, 20, 40, 0.8) !important;
        }}
        
        .dataframe th {{
            font-family: {header_font} !important;
            background-color: rgba(0, 136, 204, 0.3) !important;
            color: #00d4ff !important;
        }}
        
        .js-plotly-plot {{
            border: 1px solid #0088cc;
            box-shadow: 0 0 15px rgba(0, 136, 204, 0.3);
        }}
        
        ::-webkit-scrollbar {{
            width: 10px;
            height: 10px;
        }}
        
        ::-webkit-scrollbar-track {{
            background: #0a0e27;
        }}
        
        ::-webkit-scrollbar-thumb {{
            background: linear-gradient(135deg, #0066cc 0%, #00d4ff 100%);
            border-radius: 5px;
        }}
        
        ::-webkit-scrollbar-thumb:hover {{
            background: #00d4ff;
        }}
        
        /* Expander headers use Cyberfall */
        .streamlit-expanderHeader {{
            font-family: {header_font} !important;
            color: #00d4ff !important;
            font-weight: bold !important;
        }}
    </style>
    """, unsafe_allow_html=True)