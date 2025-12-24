"""Informational box components for dashboard callouts and alerts."""
import streamlit as st


def render_predictions_callout():
    """
    Render the CRIME PREDICTIONS callout box.
    
    This box appears below the filters and directs users to
    scroll down to see the ML predictions tab.
    """
    st.markdown("---")
    st.markdown("""
    <div style='border: 2px solid #00d4ff; padding: 10px 15px; border-radius: 8px; 
                background: linear-gradient(135deg, rgba(0,102,204,0.2) 0%, rgba(0,136,204,0.1) 100%);
                box-shadow: 0 0 20px rgba(0,212,255,0.6); 
                margin-top: 10px; text-align: center;'>
        <h4 style='margin: 0; color: #00d4ff; text-shadow: 0 0 10px rgba(0,212,255,0.8);
                   font-size: 1rem; line-height: 1.2; padding: 0;
                   letter-spacing: 2px; font-weight: bold;'>
            CRIME PREDICTIONS ⬇️
        </h4>
    </div>
    """, unsafe_allow_html=True)
    st.caption("🤖 Scroll down to see ML predictions")


def render_info_box(title, message, icon="ℹ️"):
    """
    Render a generic info box with custom title and message.
    
    Args:
        title: Box title text
        message: Caption message below the box
        icon: Emoji icon to display with title
    
    Example:
        render_info_box("DATA INSIGHTS ⬇️", "📊 View trends below")
    """
    st.markdown("---")
    st.markdown(f"""
    <div style='border: 2px solid #00d4ff; padding: 10px 15px; border-radius: 8px; 
                background: linear-gradient(135deg, rgba(0,102,204,0.2) 0%, rgba(0,136,204,0.1) 100%);
                box-shadow: 0 0 20px rgba(0,212,255,0.6); 
                margin-top: 10px; text-align: center;'>
        <h4 style='margin: 0; color: #00d4ff; text-shadow: 0 0 10px rgba(0,212,255,0.8);
                   font-size: 1rem; line-height: 1.2; padding: 0;
                   letter-spacing: 2px; font-weight: bold;'>
            {icon} {title}
        </h4>
    </div>
    """, unsafe_allow_html=True)
    if message:
        st.caption(message)