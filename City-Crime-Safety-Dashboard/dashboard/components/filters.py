"""Filter controls component."""
import streamlit as st
from config.settings import DEFAULT_YEAR


def render_filter_header():
    """Render the FILTERS header with matching gradient background."""
    st.markdown("""
    <div style='border: 2px solid #00d4ff; padding: 8px 15px; border-radius: 10px; 
                background: linear-gradient(135deg, rgba(0,102,204,0.2) 0%, rgba(0,136,204,0.1) 100%);
                box-shadow: 0 0 20px rgba(0,212,255,0.6);
                height: fit-content;'>
        <h3 style='margin: 0; color: #00d4ff; text-shadow: 0 0 10px rgba(0,212,255,0.8);
                   font-size: 1.2rem; line-height: 1.2; padding: 0;
                   letter-spacing: 2px; font-weight: bold;'>
            ⚙️ FILTERS
        </h3>
    </div>
    """, unsafe_allow_html=True)


def render_filters(filter_options):
    """
    Render filter controls and return selected values.
    
    Args:
        filter_options: Dict with keys 'years', 'crime_types', 'neighborhoods'
    
    Returns:
        Dict with selected filter values
    """
    render_filter_header()
    
    year_opts = filter_options.get("years", [])
    crime_types_all = filter_options.get("crime_types", [])
    neighborhoods_all = filter_options.get("neighborhoods", [])
    
    # Set default year
    default_year = [DEFAULT_YEAR] if DEFAULT_YEAR in year_opts else (year_opts[-1:] if year_opts else None)
    
    # Filter controls
    sel_years = st.multiselect("🗓️ YEAR", year_opts, default=default_year)
    crime_types = st.multiselect("🔫 Crime Type", crime_types_all)
    neighborhoods = st.multiselect("🏘️ Neighborhood", neighborhoods_all)
    arrest_filter = st.selectbox("👮 Arrest Made?", ["All", "Yes", "No"], index=0)
    
    # Convert arrest filter to boolean
    arrest_made = None
    if arrest_filter == "Yes":
        arrest_made = True
    elif arrest_filter == "No":
        arrest_made = False
    
    return {
        "years": sel_years,
        "crime_types": crime_types,
        "neighborhoods": neighborhoods,
        "arrest_made": arrest_made
    }