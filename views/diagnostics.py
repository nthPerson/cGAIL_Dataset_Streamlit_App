"""
Comprehensive diagnostic view for inspecting all 126 dimensions of state vectors.
This view helps validate feature interpretations and identify misalignments.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from config import HELPER_PATHS
from data_access import get_state_vector, load_lengths
from sidebar import SidebarSelection


# Known dimension ranges from empirical analysis
EMPIRICAL_RANGES = {
    "min": np.array([
        3.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # 0-9
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # 10-19
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # 20-29
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # 30-39
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # 40-49
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # 50-59
        0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # 60-69
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # 70-79
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # 80-89
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # 90-99
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # 100-109
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # 110-119
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # 120-125
    ]),
    "max": np.array([
        48.0, 81.0, 288.0, 6.0, 82.0, 74.0, 82.0, 86.0, 99.0, 103.0,  # 0-9
        87.0, 80.0, 99.0, 79.0, 84.0, 103.0, 93.0, 84.0, 80.0, 99.0,  # 10-19
        67.0, 100.0, 87.0, 85.0, 88.0, 496.0, 496.0, 496.0, 496.0, 496.0,  # 20-29
        495.0, 496.0, 496.0, 496.0, 495.0, 496.0, 496.0, 496.0, 496.0, 496.0,  # 30-39
        343.0, 430.0, 496.0, 496.0, 496.0, 318.0, 329.0, 364.0, 364.0, 328.0,  # 40-49
        29187.0, 29055.0, 28700.0, 28700.0, 28700.0, 29187.0, 28700.0, 28700.0, 28700.0, 28121.0,  # 50-59
        28686.0, 28700.0, 29187.0, 29187.0, 28700.0, 28686.0, 28575.0, 29187.0, 28700.0, 28700.0,  # 60-69
        29187.0, 28358.0, 28575.0, 23911.0, 21737.0, 0.335887, 0.194239, 0.194239, 0.194239, 0.145653,  # 70-79
        0.194239, 0.194239, 0.136819, 0.207208, 0.143281, 0.378792, 0.197565, 0.207208, 0.207208, 0.155155,  # 80-89
        0.248020, 0.245534, 0.252399, 0.197565, 0.272866, 0.240582, 0.245534, 0.145653, 0.263423, 0.272866,  # 90-99
        688.322222, 688.322222, 688.322222, 688.322222, 829.033333, 649.462698, 688.322222, 688.322222, 684.266667, 792.464848,  # 100-109
        649.462698, 688.322222, 653.688889, 829.033333, 829.033333, 688.322222, 688.322222, 688.322222, 829.033333, 829.033333,  # 110-119
        688.322222, 653.688889, 680.700000, 698.833333, 829.033333, 18.0  # 120-125
    ])
}


# Hypothesized feature groups based on analysis
FEATURE_GROUPS = {
    "Spatial Coordinates": (0, 2),  # x, y
    "Temporal": (2, 4),  # t, day_of_week
    "Traffic 5x5 (Speed?)": (4, 25),  # 21 dims, but expected 25 for 5x5
    "Traffic 5x5 (Volume?)": (25, 50),  # 25 dims
    "Traffic 5x5 (Demand?)": (50, 75),  # 25 dims  
    "Traffic 5x5 (Waiting?)": (75, 100),  # 25 dims
    "Volume Pickup / High Values": (100, 125),  # 25 dims
    "Action (Expert Label)": (125, 126),  # 1 dim
}


def render_dimension_table(vec: np.ndarray) -> None:
    """Render a detailed table of all 126 dimensions with metadata."""
    
    df_rows = []
    for i in range(126):
        val = vec[i] if i < len(vec) else np.nan
        emp_min = EMPIRICAL_RANGES["min"][i]
        emp_max = EMPIRICAL_RANGES["max"][i]
        emp_range = emp_max - emp_min
        
        # Determine which group this dimension belongs to
        group_name = "Unknown"
        for gname, (start, end) in FEATURE_GROUPS.items():
            if start <= i < end:
                group_name = gname
                break
        
        # Flag suspicious values
        is_suspicious = ""
        if i < 125 and not np.isnan(val):  # State dimensions
            if val < emp_min or val > emp_max:
                is_suspicious = "⚠️ OUT OF RANGE"
        
        df_rows.append({
            "Dim": i,
            "Group": group_name,
            "Value": f"{val:.6f}" if not np.isnan(val) else "N/A",
            "Emp Min": f"{emp_min:.2f}",
            "Emp Max": f"{emp_max:.2f}",
            "Emp Range": f"{emp_range:.2f}",
            "Flag": is_suspicious
        })
    
    df = pd.DataFrame(df_rows)
    
    st.markdown("### 📊 All 126 Dimensions — Detailed View")
    st.caption("Empirical ranges computed from 666,729 vectors across entire dataset")
    
    # Display with formatting
    st.dataframe(
        df,
        use_container_width=True,
        height=600,
        column_config={
            "Dim": st.column_config.NumberColumn("Dim", width="small"),
            "Group": st.column_config.TextColumn("Feature Group", width="medium"),
            "Value": st.column_config.TextColumn("Current Value", width="small"),
            "Emp Min": st.column_config.TextColumn("Min", width="small"),
            "Emp Max": st.column_config.TextColumn("Max", width="small"),
            "Emp Range": st.column_config.TextColumn("Range", width="small"),
            "Flag": st.column_config.TextColumn("Status", width="small"),
        }
    )


def render_grouped_bar_charts(vec: np.ndarray) -> None:
    """Render bar charts grouped by feature type."""
    
    st.markdown("### 📈 Grouped Feature Visualizations")
    
    for group_name, (start, end) in FEATURE_GROUPS.items():
        with st.expander(f"{group_name} (Dims {start}–{end-1})", expanded=(group_name == "Spatial Coordinates")):
            if start >= len(vec):
                st.info(f"No data available for dimensions {start}-{end-1}")
                continue
            
            actual_end = min(end, len(vec))
            group_vec = vec[start:actual_end]
            dims = list(range(start, actual_end))
            
            df = pd.DataFrame({
                "Dimension": dims,
                "Value": group_vec,
                "Dim Label": [f"D{i}" for i in dims]
            })
            
            fig = px.bar(
                df,
                x="Dim Label",
                y="Value",
                title=f"{group_name}",
                labels={"Dim Label": "Dimension", "Value": "Feature Value"},
                hover_data={"Dimension": True, "Value": ":.6f"}
            )
            
            fig.update_layout(
                height=300,
                margin=dict(l=0, r=0, t=40, b=20),
                xaxis=dict(tickangle=-45) if (end - start) <= 30 else dict(showticklabels=False)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show statistics for this group
            stats_cols = st.columns(4)
            stats_cols[0].metric("Min", f"{np.min(group_vec):.3f}")
            stats_cols[1].metric("Max", f"{np.max(group_vec):.3f}")
            stats_cols[2].metric("Mean", f"{np.mean(group_vec):.3f}")
            stats_cols[3].metric("Std", f"{np.std(group_vec):.3f}")


def render_heatmap_interpretations(vec: np.ndarray) -> None:
    """Render multiple interpretations of the 5x5 traffic neighborhood."""
    
    st.markdown("### 🗺️ Traffic Neighborhood Interpretations (5×5 Grid)")
    st.caption("Testing different hypotheses about how traffic features are organized")
    
    # Current interpretation from config (4 features × 5×5)
    st.markdown("#### Hypothesis 1: Current Code Interpretation (Dims 4-103)")
    st.caption("Assumes: 25 cells × 4 features = 100 dims starting at index 4")
    
    if len(vec) >= 104:
        traffic_block = vec[4:104]
        
        # Reshape as [5, 5, 4]
        try:
            traffic_3d = traffic_block.reshape(5, 5, 4)
            feature_names = ["Speed", "Volume", "Demand", "Waiting"]
            
            cols = st.columns(4)
            for i, (col, name) in enumerate(zip(cols, feature_names)):
                heatmap_data = traffic_3d[:, :, i]
                fig = go.Figure(
                    data=go.Heatmap(
                        z=heatmap_data,
                        x=list(range(5)),
                        y=list(range(5)),
                        colorscale="Viridis",
                        xgap=1,
                        ygap=1,
                        hovertemplate=f"{name}<br>Row=%{{y}}, Col=%{{x}}<br>Value=%{{z:.3f}}<extra></extra>"
                    )
                )
                fig.update_layout(
                    title=name,
                    height=250,
                    margin=dict(l=0, r=0, t=30, b=0),
                    yaxis=dict(autorange="reversed", scaleanchor="x", scaleratio=1)
                )
                col.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Failed to reshape: {e}")
    
    # Alternative: separate 5×5 blocks
    st.markdown("#### Hypothesis 2: Four Separate 5×5 Blocks")
    st.caption("Assumes: Dims 4-28 (Speed), 25-49 (Volume), 50-74 (Demand), 75-99 (Waiting)")
    
    blocks = [
        ("Speed", 4, 29, "Blues"),
        ("Volume", 25, 50, "Greens"),
        ("Demand", 50, 75, "Oranges"),
        ("Waiting", 75, 100, "Reds")
    ]
    
    cols = st.columns(4)
    for col, (name, start, end, colorscale) in zip(cols, blocks):
        if len(vec) >= end:
            block = vec[start:end]
            if len(block) == 25:
                grid = block.reshape(5, 5)
                fig = go.Figure(
                    data=go.Heatmap(
                        z=grid,
                        x=list(range(5)),
                        y=list(range(5)),
                        colorscale=colorscale,
                        xgap=1,
                        ygap=1,
                        hovertemplate=f"{name}<br>Row=%{{y}}, Col=%{{x}}<br>Value=%{{z:.3f}}<extra></extra>"
                    )
                )
                fig.update_layout(
                    title=f"{name} (D{start}-{end-1})",
                    height=250,
                    margin=dict(l=0, r=0, t=30, b=0),
                    yaxis=dict(autorange="reversed", scaleanchor="x", scaleratio=1)
                )
                col.plotly_chart(fig, use_container_width=True)


def render_action_analysis(vec: np.ndarray) -> None:
    """Analyze the action dimension (125)."""
    
    st.markdown("### 🎯 Action Analysis (Dimension 125)")
    
    if len(vec) >= 126:
        action_val = int(vec[125])
        
        cols = st.columns(3)
        cols[0].metric("Action Value", action_val)
        cols[1].metric("Expected Range", "0-18")
        cols[2].metric("Valid?", "✅ Yes" if 0 <= action_val <= 18 else "❌ No")
        
        # Action interpretation
        st.markdown("#### Action Space Interpretation")
        st.caption("18 discrete actions = 9 spatial moves × 2 time options")
        
        if 0 <= action_val <= 18:
            spatial_action = action_val % 9
            time_advance = action_val // 9
            
            spatial_map = {
                0: "Stay",
                1: "Up-Left",
                2: "Up",
                3: "Up-Right",
                4: "Left",
                5: "Right",
                6: "Down-Left",
                7: "Down",
                8: "Down-Right"
            }
            
            st.info(
                f"**Action {action_val}**: {spatial_map.get(spatial_action, 'Unknown')} + "
                f"{'Advance Time' if time_advance == 1 else 'Current Time'}"
            )
    else:
        st.warning("Action dimension not available in this vector")


def render_diagnostics_section(selection: SidebarSelection) -> None:
    """Main diagnostic view entry point."""
    
    st.header("🔬 Feature Diagnostics (All 126 Dimensions)")
    st.markdown(
        "This diagnostic view helps validate feature interpretations and identify "
        "dimension misalignments. **Note:** Dimension 125 contains the expert action label."
    )
    
    sel_expert = selection.expert
    lens = load_lengths(HELPER_PATHS["lengths_dir"], sel_expert)
    traj_len = int(lens[selection.traj_idx]) if lens.size and selection.traj_idx < lens.size else 0
    
    # Metrics
    metric_cols = st.columns(3)
    metric_cols[0].metric("Expert", sel_expert)
    metric_cols[1].metric("Trajectory", selection.traj_idx)
    metric_cols[2].metric("State Index", selection.state_idx)
    
    # Get state vector
    vec = get_state_vector(sel_expert, selection.traj_idx, selection.state_idx) if traj_len else np.array([])
    
    if vec.size == 0:
        st.warning("No state vector available for the selected trajectory/state.")
        return
    
    if vec.size != 126:
        st.error(f"⚠️ Expected 126 dimensions, got {vec.size}. Data may be corrupted or incorrectly formatted.")
        return
    
    st.success(f"✅ Loaded vector with {vec.size} dimensions")
    
    # Tabbed interface
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Dimension Table",
        "📊 Grouped Charts",
        "🗺️ Traffic Grids",
        "🎯 Action"
    ])
    
    with tab1:
        render_dimension_table(vec)
    
    with tab2:
        render_grouped_bar_charts(vec)
    
    with tab3:
        render_heatmap_interpretations(vec)
    
    with tab4:
        render_action_analysis(vec)
    
    # Export option
    st.markdown("---")
    st.markdown("### 💾 Export Current Vector")
    if st.button("Download as CSV"):
        df_export = pd.DataFrame({
            "dimension": range(126),
            "value": vec,
            "empirical_min": EMPIRICAL_RANGES["min"],
            "empirical_max": EMPIRICAL_RANGES["max"]
        })
        csv = df_export.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"state_vector_expert{sel_expert}_traj{selection.traj_idx}_state{selection.state_idx}.csv",
            mime="text/csv"
        )
