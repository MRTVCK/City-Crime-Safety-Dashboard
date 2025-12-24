"""ML Predictions component - Risk Assessment focused."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from config.settings import MIN_PREDICTION_DATA, MIN_RISK_ASSESSMENT_POINTS
from src.ml_models import TimeLocationRiskClassifier
from src.visualizations import (
    create_feature_importance_chart,
    create_risk_by_hour_chart, 
    create_risk_by_day_chart, 
    create_high_risk_hours_chart,
    neonize
)


def render_ml_predictions(fdf):
    """Render ML-powered risk assessment predictions."""
    
    if len(fdf) < MIN_PREDICTION_DATA:
        st.warning(f"⚠️ Need at least {MIN_PREDICTION_DATA} incidents for predictions. Please adjust filters.")
        return
    
    # Introduction
    st.markdown("""
    ### 🎯 ML-Powered Crime Risk Assessment
    
    Advanced machine learning analysis identifying **high-risk periods and locations** through 
    Gradient Boosting classification. Provides actionable insights for optimized resource allocation 
    and patrol scheduling.
    """)
    
    st.divider()
    
    # Main Risk Assessment Section - EXPANDED BY DEFAULT
    with st.expander("🎯 ML Risk Assessment", expanded=True):
        render_ml_risk_assessment(fdf)


def render_ml_risk_assessment(fdf):
    """Render comprehensive ML risk assessment with Gradient Boosting."""
    
    if {"hour", "weekday", "neighborhood"}.issubset(fdf.columns):
        try:
            st.markdown("#### 🤖 Gradient Boosting Risk Prediction")
            st.info("Using Gradient Boosting ML to predict crime risk levels based on time and location patterns")
            
            ml_data = fdf.dropna(subset=["hour", "weekday", "neighborhood"]).copy()
            
            if len(ml_data) >= MIN_RISK_ASSESSMENT_POINTS:
                classifier = TimeLocationRiskClassifier()
                risk_features = classifier.prepare_features(ml_data)
                
                # Use only legitimate predictive features (NOT incident_count)
                X = risk_features[["hour", "weekday_encoded", "neighborhood_encoded", "is_night", "is_evening", "is_weekend"]]
                y = risk_features["risk"]
                
                if y.nunique() >= 2 and len(X) >= 50:
                    from sklearn.model_selection import train_test_split
                    
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.25, random_state=42, stratify=y
                    )
                    
                    with st.spinner("🤖 Training Gradient Boosting classifier..."):
                        train_acc, test_acc = classifier.train(X_train, y_train, X_test, y_test)
                    
                    # Model Performance Metrics
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Training Accuracy", f"{train_acc*100:.1f}%")
                    col2.metric("Test Accuracy", f"{test_acc*100:.1f}%")
                    col3.metric("Scenarios Analyzed", len(risk_features))
                    
                    # Make predictions
                    risk_features["predicted_risk"] = classifier.predict(X)
                    high_risk_scenarios = risk_features[risk_features["predicted_risk"] == "High"].sort_values(
                        "incident_count", ascending=False
                    )
                    
                    if len(high_risk_scenarios) > 0:
                        st.warning(f"⚠️ **ML Identified {len(high_risk_scenarios)} HIGH-RISK scenarios**")
                        
                        # Top 10 Highest Risk Predictions
                        st.markdown("#### 🔴 Top 10 Highest Risk Predictions")
                        risk_display = high_risk_scenarios.head(10)[[
                            "hour", "weekday", "neighborhood", "incident_count", "predicted_risk"
                        ]].copy()
                        risk_display.columns = ["Hour", "Day", "Neighborhood", "Historical Incidents", "ML Prediction"]
                        st.dataframe(risk_display, use_container_width=True)
                        
                        st.divider()
                        
                        # ========== RISK BY TIME OF DAY ==========
                        st.markdown("#### ⏰ Risk by Time of Day")
                        
                        high_risk_hours = high_risk_scenarios.groupby("hour").size().reset_index(name="high_risk_count")
                        fig_hours = create_high_risk_hours_chart(high_risk_hours)
                        st.plotly_chart(fig_hours, use_container_width=True)
                        
                        # Key insights
                        peak_hours = high_risk_hours.nlargest(3, "high_risk_count")["hour"].tolist()
                        st.info(f"🔴 **Peak High-Risk Hours:** {', '.join([f'{int(h)}:00' for h in peak_hours])} - Recommend increased patrol presence")
                        
                        st.divider()
                        
                        # ========== RISK BY DAY OF WEEK ==========
                        st.markdown("#### 📅 Risk by Day of Week")
                        
                        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                        high_risk_days = high_risk_scenarios.groupby("weekday").size().reset_index(name="high_risk_count")
                        high_risk_days["weekday"] = pd.Categorical(
                            high_risk_days["weekday"], categories=day_order, ordered=True
                        )
                        high_risk_days = high_risk_days.sort_values("weekday")
                        
                        # Color code by risk level
                        high_risk_days["risk_category"] = pd.cut(
                            high_risk_days["high_risk_count"],
                            bins=3,
                            labels=["Lower Risk", "Moderate Risk", "Higher Risk"]
                        )
                        
                        fig_days = px.bar(
                            high_risk_days,
                            x="weekday",
                            y="high_risk_count",
                            color="risk_category",
                            color_discrete_map={
                                "Lower Risk": "#88cc44",
                                "Moderate Risk": "#ffaa44",
                                "Higher Risk": "#ff4444"
                            },
                            labels={"weekday": "Day of Week", "high_risk_count": "High-Risk Scenarios"},
                            title="High-Risk Scenario Distribution by Day"
                        )
                        fig_days = neonize(fig_days)
                        st.plotly_chart(fig_days, use_container_width=True)
                        
                        # Key insights
                        peak_days = high_risk_days.nlargest(2, "high_risk_count")["weekday"].tolist()
                        st.info(f"🔴 **Peak High-Risk Days:** {', '.join([str(d) for d in peak_days])} - Plan enhanced weekend/special event patrols")
                        
                        st.divider()
                        
                        # ========== RISK BY LOCATION (TOP 15 NEIGHBORHOODS) ==========
                        st.markdown("#### 🗺️ Risk by Location")
                        
                        location_risk = high_risk_scenarios.groupby("neighborhood").agg({
                            "incident_count": "sum",
                            "predicted_risk": "count"
                        }).reset_index()
                        location_risk.columns = ["neighborhood", "total_incidents", "high_risk_scenarios"]
                        location_risk = location_risk.sort_values("total_incidents", ascending=False).head(15)
                        
                        # Categorize risk
                        location_risk["risk_category"] = pd.cut(
                            location_risk["total_incidents"],
                            bins=3,
                            labels=["Medium Risk", "High Risk", "Critical Risk"]
                        )
                        
                        fig_location = px.bar(
                            location_risk,
                            x="total_incidents",
                            y="neighborhood",
                            color="risk_category",
                            orientation="h",
                            color_discrete_map={
                                "Medium Risk": "#ffaa44",
                                "High Risk": "#ff6644",
                                "Critical Risk": "#ff4444"
                            },
                            labels={"neighborhood": "Neighborhood", "total_incidents": "Total Crime Incidents"},
                            title="Top 15 Neighborhoods by Crime Risk Level"
                        )
                        fig_location = neonize(fig_location)
                        st.plotly_chart(fig_location, use_container_width=True)
                        
                        # Critical zones
                        critical_zones = location_risk[
                            location_risk["risk_category"] == "Critical Risk"
                        ]["neighborhood"].tolist()
                        if critical_zones:
                            st.error(f"🚨 **Critical Risk Zones:** {', '.join(critical_zones)} - Recommend priority resource allocation")
                        
                        st.divider()
                        
                        # ========== COMBINED TEMPORAL RISK HEATMAP ==========
                        st.markdown("#### 🔥 Combined Temporal Risk Pattern")
                        
                        # Create pivot table for heatmap
                        heatmap_data = high_risk_scenarios.groupby(["weekday", "hour"]).size().reset_index(name="count")
                        heatmap_pivot = heatmap_data.pivot(
                            index="weekday", columns="hour", values="count"
                        ).fillna(0)
                        
                        # Reorder days
                        heatmap_pivot = heatmap_pivot.reindex(day_order)
                        
                        fig_heatmap = go.Figure(data=go.Heatmap(
                            z=heatmap_pivot.values,
                            x=heatmap_pivot.columns,
                            y=heatmap_pivot.index,
                            colorscale=[
                                [0, "#1a1a2e"],
                                [0.33, "#88cc44"],
                                [0.66, "#ffaa44"],
                                [1, "#ff4444"]
                            ],
                            colorbar=dict(title="High-Risk<br>Scenarios"),
                        ))
                        
                        fig_heatmap.update_layout(
                            title="ML Predicted High-Risk Periods: Day of Week × Hour of Day",
                            xaxis_title="Hour of Day",
                            yaxis_title="Day of Week",
                            template="plotly_dark",
                            paper_bgcolor="rgba(10,14,39,0.8)",
                            plot_bgcolor="rgba(10,20,40,0.6)",
                            font=dict(family="'Courier New', monospace", color="#00d4ff", size=11),
                            xaxis=dict(gridcolor="rgba(0,136,204,0.2)", color="#66b3ff"),
                            yaxis=dict(gridcolor="rgba(0,136,204,0.2)", color="#66b3ff")
                        )
                        st.plotly_chart(fig_heatmap, use_container_width=True)
                        
                        st.success("💡 **Insight:** Dark red zones indicate ML-predicted peak crime periods - optimal times for preventative patrols")
                        
                        st.divider()
                        
                        # ========== MOST DANGEROUS TIME-LOCATION COMBINATIONS ==========
                        st.markdown("#### 🎯 Most Dangerous Time-Location Combinations (ML Ranked)")
                        top_combos = high_risk_scenarios.nlargest(5, "incident_count")
                        for idx, row in top_combos.iterrows():
                            st.error(
                                f"🔴 **{row['neighborhood']}** on **{row['weekday']}** at "
                                f"**{int(row['hour'])}:00** — {row['incident_count']} historical incidents"
                            )
                        
                        st.divider()
                        
                        # ========== ACTIONABLE RECOMMENDATIONS ==========
                        st.markdown("#### 📋 Actionable Recommendations")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("""
                            **🎯 Resource Allocation:**
                            - Deploy additional units during peak high-risk hours
                            - Focus patrols on critical risk zones identified by ML
                            - Adjust staffing based on day-of-week patterns
                            - Implement predictive policing strategies
                            """)
                        
                        with col2:
                            st.markdown("""
                            **📊 Data-Driven Benefits:**
                            - ML-optimized patrol scheduling
                            - Proactive crime prevention (not reactive)
                            - Efficient budget allocation
                            - Measurable risk reduction through targeted deployment
                            """)
                        
                        st.divider()
                        
                        # ========== SUMMARY METRICS ==========
                        st.markdown("#### 📊 Summary Metrics")
                        
                        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                        
                        with metric_col1:
                            st.metric(
                                "Total Incidents Analyzed",
                                f"{len(fdf):,}",
                                help="Number of crime incidents in ML analysis"
                            )
                        
                        with metric_col2:
                            st.metric(
                                "High-Risk Scenarios",
                                f"{len(high_risk_scenarios)}",
                                help="ML-predicted high-risk time/location combinations"
                            )
                        
                        with metric_col3:
                            st.metric(
                                "Locations Analyzed",
                                f"{fdf['neighborhood'].nunique()}",
                                help="Unique neighborhoods assessed by ML model"
                            )
                        
                        with metric_col4:
                            if "datetime" in fdf.columns:
                                date_range = (fdf["datetime"].max() - fdf["datetime"].min()).days
                                st.metric(
                                    "Analysis Period",
                                    f"{date_range} days",
                                    help="Time span of analyzed data"
                                )
                        
                    else:
                        st.success("✅ No high-risk scenarios predicted by the ML model based on current filters")
                    
                    # ========== ML MODEL DETAILS ==========
                    with st.expander("📊 ML Model Feature Importance"):
                        st.markdown("**Understanding which factors most influence crime risk predictions:**")
                        feature_importance = classifier.get_feature_importance()
                        fig = create_feature_importance_chart(
                            feature_importance, 
                            "Gradient Boosting Feature Importance"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.markdown("""
                        **Feature Explanations:**
                        - **Hour:** Time of day (0-23)
                        - **Weekday:** Day of week encoding
                        - **Neighborhood:** Location encoding
                        - **Is Night/Evening:** Time period flags
                        - **Is Weekend:** Weekend flag
                        """)
                    
                    st.success("✅ **ML Model:** Gradient Boosting successfully trained and validated")
                else:
                    st.info("Not enough data variance for ML prediction. Need at least 2 risk categories and 50+ samples.")
            else:
                st.warning(f"Need at least {MIN_RISK_ASSESSMENT_POINTS} incidents for ML risk assessment.")
        
        except ImportError:
            st.error("❌ scikit-learn not installed. Install with: `pip install scikit-learn --break-system-packages`")
            st.info("ML features require scikit-learn. Installing it will enable advanced predictions.")
        except Exception as e:
            st.error(f"ML error: {str(e)}")
            import traceback
            with st.expander("🔍 Error Details (for debugging)"):
                st.code(traceback.format_exc())
    
    # Footer note
    st.markdown("---")
    st.caption("""
    💡 **Methodology:** Risk levels predicted using Gradient Boosting machine learning classifier 
    trained on historical spatial and temporal crime patterns. Model accuracy validated through 
    train-test split methodology. Higher predicted risk indicates elevated probability of crime 
    occurrence based on learned patterns.
    """)