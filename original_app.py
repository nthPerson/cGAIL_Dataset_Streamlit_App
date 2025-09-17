# app.py
import json
import pickle
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --------------------------
# 0) LOAD DATA
# --------------------------
st.set_page_config(page_title="Imitation Learning Dataset Explorer", layout="wide")
st.title("Imitation Learning Dataset Explorer")
st.caption("Experts → Trajectories → States → 126-dim vectors")

@st.cache_data(show_spinner=False)
def load_dataset(path: str) -> Dict[str, List[List[List[float]]]]:
    p = Path(path)
    if p.suffix in [".pkl", ".pickle"]:
        return pickle.loads(p.read_bytes())
    elif p.suffix in [".json"]:
        return json.loads(p.read_text())
    else:
        st.stop()

# DATA_PATH = "../all_trajs.pkl"  # expert dataset
# DATA_PATH = "data/Processed_Data/all_trajs.pkl"  # expert dataset
DATA_PATH = "/home/robert/FAMAIL/data/Processed_Data/all_trajs.pkl"  # expert dataset
st.sidebar.header("Data")
# data_path = st.sidebar.text_input("Path to dataset (.pkl or .json)", DATA_PATH)
dataset = load_dataset(DATA_PATH)

# if not data_path or not Path(data_path).exists():
#     st.info("Provide a valid path to your dataset to enable exploration. Showing a tiny mock dataset.")
#     # ---- mock minimal example ----
#     rng = np.random.default_rng(0)
#     mock = {
#         f"driver_{i:02d}": [
#             [rng.normal(size=126).tolist() for _ in range(rng.integers(15, 50))]  # states
#             for _ in range(rng.integers(120, 300))  # trajectories
#         ] for i in range(6)  # fewer experts for demo
#     }
#     dataset = mock
# else:
#     dataset = load_dataset(data_path)

# --------------------------
# 1) SUMMARY TABLES

# --------------------------
def summarize(dataset):
    rows = []
    total_states = 0
    for e_id, trajs in dataset.items():
        traj_lengths = [len(t) for t in trajs]
        n_traj = len(trajs)
        n_states = int(np.sum(traj_lengths)) if len(traj_lengths) else 0
        total_states += n_states
        rows.append({
            "expert": e_id,
            "trajectories": n_traj,
            "states_total": n_states,
            "states_min": int(np.min(traj_lengths)) if traj_lengths else 0,
            "states_med": float(np.median(traj_lengths)) if traj_lengths else 0,
            "states_max": int(np.max(traj_lengths)) if traj_lengths else 0,
        })
    df = pd.DataFrame(rows).sort_values("trajectories", ascending=False)
    return df, total_states

summary_df, total_states = summarize(dataset)

with st.expander("Dataset summary", expanded=True):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Experts", len(dataset))
    c2.metric("Total trajectories", int(summary_df["trajectories"].sum()))
    c3.metric("Total states", int(total_states))
    c4.metric("Features per state", 126)

    st.dataframe(summary_df, width='stretch', height=240)

# --------------------------
# 2) "Index Map" Heatmap (experts x trajectories)
#    
# --------------------------
# ---- Build long DF of trajectory lengths ----
def to_long_df(dataset):
    rows = []
    for e, trajs in dataset.items():
        for t_idx, t in enumerate(trajs):
            rows.append({"expert": str(e), "traj_idx": t_idx, "states": len(t)})
    return pd.DataFrame(rows)

long_df = to_long_df(dataset)
# print(f'[DATA] Shape of dataset: {long_df.shape}')
long_df_shape = f'[DATA] Shape of dataset: {long_df.shape}'
st.text(long_df_shape)

st.subheader("Hierarchy overview")
##### Quantile-Binned Icicle ############################################################################################
def quantile_rows_no_root(dataset, q=5):
    rows = []
    for e, trajs in dataset.items():
        lengths = np.array([len(t) for t in trajs])
        if lengths.size == 0:
            continue

        # Bin by quantiles (allow duplicates drop)
        qbins = pd.qcut(lengths, q, duplicates="drop")
        k = len(qbins.categories)
        labels = [f"Q{i+1}" for i in range(k)]
        qbins = qbins.rename_categories(labels)

        # Root-level expert node (parent is empty string → no common root)
        rows.append({
            "id": f"expert::{e}",
            "label": str(e),
            "parent": "",
            "count": int(len(lengths)),
            "states_total": int(lengths.sum()),
        })

        # Bin leaves under this expert
        for lab in qbins.categories:
            idx = np.where(qbins == lab)[0]
            rows.append({
                "id": f"expert::{e}::bin::{lab}",
                "label": lab,                # pretty label on the box
                "parent": f"expert::{e}",
                "count": int(len(idx)),
                "states_total": int(lengths[idx].sum()),
            })

    return pd.DataFrame(rows)

qdf = quantile_rows_no_root(dataset, q=5)

fig = px.icicle(
    qdf,
    ids="id",            # ensure uniqueness
    names="label",       # what’s shown on each box
    parents="parent",
    values="states_total",
    color="count",
    color_continuous_scale="Blues",
    # branchvalues default works fine here
)
fig.update_layout(height=500, margin=dict(l=0, r=0, t=10, b=0))
st.subheader("Quantile-binned hierarchy")
st.plotly_chart(fig, width='stretch')

# def quantile_rows(dataset, q=5):
#     rows = []
#     for e, trajs in dataset.items():
#         lengths = np.array([len(t) for t in trajs])
#         if lengths.size == 0:
#             continue
#         # Let qcut decide how many bins after dropping duplicates
#         qbins = pd.qcut(lengths, q, duplicates="drop")
#         # Rename categories to Q1..Qk where k is actual number of bins
#         k = len(qbins.categories)
#         new_labels = [f"trajectory-{i+1}" for i in range(k)]
#         qbins = qbins.rename_categories(new_labels)

#         for lab in qbins.categories:
#             idx = np.where(qbins == lab)[0]
#             rows.append({
#                 "name": lab,
#                 "expert": str(e),
#                 "count": int(len(idx)),
#                 "states_total": int(lengths[idx].sum())
#             })
#         rows.append({
#             "name": str(e),
#             "expert": "Dataset",
#             "count": int(len(lengths)),
#             "states_total": int(lengths.sum())
#         })
#     rows.append({"name": "Dataset", "expert": ""})
#     return pd.DataFrame(rows)

# qdf = quantile_rows(dataset, q=5).drop_duplicates(subset=["name","expert"])
# fig = px.icicle(qdf, names="name", parents="expert", values="states_total",
#                 color="count", color_continuous_scale="Blues")
# fig.update_layout(height=500, margin=dict(l=0,r=0,t=10,b=0))
# st.subheader("Quantile-binned hierarchy")
# st.plotly_chart(fig, use_container_width=True)

############################################################################################


##### Expert Treemap ############################################################################################
# summary = (
#     long_df.groupby("expert")
#     .agg(trajectories=("traj_idx","count"),
#          states_total=("states","sum"),
#          states_median=("states","median"))
#     .reset_index()
# )

# st.subheader("Experts overview")
# treemap = px.treemap(
#     summary, path=["expert"],
#     values="states_total", color="states_median",
#     color_continuous_scale="Blues",
#     hover_data={"trajectories":":,","states_total":":,","states_median":":.0f"}
# )
# treemap.update_layout(margin=dict(l=0,r=0,t=10,b=0), height=420)
# st.plotly_chart(treemap, use_container_width=True)
############################################################################################

###### Beeswarm / Strip Plot ############################################################################################
st.subheader("Trajectory length distribution by expert")
st.caption("Each dot is a trajectory (jittered); x = # states")
strip = px.strip(long_df, x="states", y="expert", orientation="h", hover_data=["traj_idx"],
                 template="plotly_white")
strip.update_traces(jitter=0.4, opacity=0.5, marker=dict(size=4))
strip.update_layout(height=800, margin=dict(l=0,r=0,t=10,b=0))
st.plotly_chart(strip, width='stretch')
############################################################################################

##### Index Map Heatmap ############################################################################################
st.caption("Each square is a trajectory; color shows # states.")

# Make a rectangular matrix by padding missing trajectories per expert with NaN
max_traj = long_df.groupby("expert")["traj_idx"].max().max()
experts_sorted = sorted(long_df["expert"].unique())
z = []
for e in experts_sorted:
    row = [np.nan]*(max_traj+1)
    sub = long_df[long_df["expert"] == e]
    for _, r in sub.iterrows():
        row[int(r["traj_idx"])] = r["states"]
    z.append(row)

heat = go.Figure(
    data=go.Heatmap(
        z=z, x=list(range(max_traj+1)), y=experts_sorted,
        coloraxis="coloraxis", hovertemplate="expert=%{y}<br>traj=%{x}<br>#states=%{z}<extra></extra>"
    )
)
heat.update_layout(height=520, margin=dict(l=0,r=0,t=10,b=0), coloraxis=dict(colorscale="Blues"))
st.plotly_chart(heat, width='stretch')
############################################################################################

# # --------------------------
# # 2) SUNBURST (Experts → Trajectories)
# #    value = #states (for visual weight)
# # --------------------------
# sun_rows = []
# for e_id, trajs in dataset.items():
#     # we do NOT include states as leaves here (would be too big).
#     sun_rows.append({"level": "root", "name": "Dataset", "parent": ""})
#     # Note: Plotly will deduplicate identical rows; we only need one root row overall
#     break

# for e_id, trajs in dataset.items():
#     n_states = sum(len(t) for t in trajs)
#     sun_rows.append({"level": "expert", "name": e_id, "parent": "Dataset", "value": n_states})
#     # Represent each trajectory as a leaf under the expert (value = #states)
#     for idx, t in enumerate(trajs):
#         sun_rows.append({"level": "trajectory", "name": f"traj_{idx}", "parent": e_id, "value": len(t)})

# sun_df = pd.DataFrame(sun_rows).drop_duplicates(subset=["name", "parent"])

# fig = px.sunburst(
#     sun_df, names="name", parents="parent", values="value",
#     color="value", color_continuous_scale="Blues",
#     maxdepth=-1
# )
# fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=500)

# st.subheader("Hierarchy overview")
# sun_click = st.plotly_chart(fig, use_container_width=True)

# st.caption("Tip: Click an expert or trajectory to drill down below.")
###########################################################################################

##### Custom Hierarcical Display #######################################################
# import numpy as np
# import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go
# import streamlit as st
# from streamlit_plotly_events import plotly_events

# # ---------- Helpers ----------
# def to_long_df(dataset):
#     rows = []
#     for e, trajs in dataset.items():
#         for t_idx, t in enumerate(trajs):
#             rows.append({"expert": str(e), "traj_idx": t_idx, "states": len(t)})
#     return pd.DataFrame(rows)

# @st.cache_data(show_spinner=False)
# def compute_summaries(dataset):
#     long_df = to_long_df(dataset)
#     # per-expert summary
#     g = long_df.groupby("expert")
#     summary = pd.DataFrame({
#         "expert": g.size().index,
#         "trajectories": g.size().values,
#         "states_total": g["states"].sum().values,
#         "states_avg": g["states"].mean().values,
#         "states_min": g["states"].min().values,
#         "states_med": g["states"].median().values,
#         "states_max": g["states"].max().values,
#     }).sort_values("trajectories", ascending=False).reset_index(drop=True)
#     return long_df, summary

# long_df, summary = compute_summaries(dataset)

# # ---------- LEVEL 1: Experts treemap ----------
# st.subheader("Experts overview")
# treemap = px.treemap(
#     summary, path=["expert"], values="trajectories",
#     color="states_avg", color_continuous_scale="Blues",
#     hover_data={
#         "trajectories":":,",
#         "states_total":":,",
#         "states_avg":":.1f",
#         "states_min":":.0f",
#         "states_med":":.0f",
#         "states_max":":.0f",
#     }
# )
# treemap.update_layout(margin=dict(l=0,r=0,t=10,b=0), height=420)
# sel = plotly_events(treemap, click_event=True, hover_event=False, select_event=False, override_height=420, override_width="100%")
# if sel:
#     # Plotly returns different keys depending on chart type; handle robustly:
#     sel_expert = sel[0].get("label") or sel[0].get("text") or sel[0].get("customdata") or sel[0].get("pointLabel")
# else:
#     sel_expert = st.selectbox("Pick an expert", summary["expert"].tolist())

# st.markdown(f"**Selected expert:** `{sel_expert}`")

# exp_df = long_df[long_df["expert"] == sel_expert].copy()
# exp_lengths = exp_df["states"].to_numpy()
# n_traj = len(exp_lengths)

# # ---------- Toggle for trajectory-level view ----------
# view_mode = st.radio(
#     "Trajectory level view",
#     ["Quartile bins (Q1-Q4)", "All trajectories heatmap"],
#     horizontal=True,
# )

# # ---------- LEVEL 2A: Quartile-binned icicle + samples ----------
# if view_mode.startswith("Quartile"):
#     st.subheader("Trajectory length quantiles")
#     # Compute quantiles and stats for hover
#     qlabels = ["Q1 (shortest 25%)", "Q2 (25-50%)", "Q3 (50-75%)", "Q4 (longest 25%)"]
#     q = pd.qcut(exp_lengths, q=4, duplicates="drop", labels=qlabels[:len(pd.qcut(exp_lengths, 4, duplicates='drop').categories)])
#     exp_df["quantile"] = q.astype(str)

#     bins = (
#         exp_df.groupby("quantile")["states"]
#         .agg(count="count", total="sum", avg="mean", min="min", med="median", max="max")
#         .reset_index()
#         .sort_values("quantile")
#     )

#     # Build rootless icicle: expert → quantile bins
#     icile_rows = [{"id": f"expert::{sel_expert}", "label": sel_expert, "parent": "", "value": int(exp_lengths.sum()), "count": n_traj}]
#     for _, r in bins.iterrows():
#         icile_rows.append({
#             "id": f"expert::{sel_expert}::{r['quantile']}",
#             "label": r["quantile"],
#             "parent": f"expert::{sel_expert}",
#             "value": int(r["total"]),
#             "count": int(r["count"]),
#             "avg": float(r["avg"]), "min": int(r["min"]), "med": float(r["med"]), "max": int(r["max"]),
#         })
#     qdf = pd.DataFrame(icile_rows)

#     fig = px.icicle(
#         qdf, ids="id", names="label", parents="parent", values="value",
#         color="count", color_continuous_scale="Blues"
#     )
#     fig.update_traces(
#         hovertemplate=(
#             "<b>%{label}</b><br>"
#             "# trajectories: %{customdata[0]:,}<br>"
#             "total states: %{value:,}<br>"
#             "avg states/trajectory: %{customdata[1]:.1f}<br>"
#             "min / median / max: %{customdata[2]} / %{customdata[3]:.0f} / %{customdata[4]}"
#             "<extra></extra>"
#         ),
#         customdata=np.vstack([
#             qdf.get("count", pd.Series([n_traj] + [None]*(len(qdf)-1))).fillna(0).to_numpy(),
#             qdf.get("avg", pd.Series([exp_lengths.mean()] + [None]*(len(qdf)-1))).fillna(0).to_numpy(),
#             qdf.get("min", pd.Series([exp_lengths.min()] + [None]*(len(qdf)-1))).fillna(0).to_numpy(),
#             qdf.get("med", pd.Series([np.median(exp_lengths)] + [None]*(len(qdf)-1))).fillna(0).to_numpy(),
#             qdf.get("max", pd.Series([exp_lengths.max()] + [None]*(len(qdf)-1))).fillna(0).to_numpy(),
#         ]).T
#     )
#     fig.update_layout(margin=dict(l=0,r=0,t=10,b=0), height=380)
#     _ = st.plotly_chart(fig, use_container_width=True)

#     # ---- Samples: few shortest, few typical, few longest ----
#     st.markdown("**Samples: shortest • typical • longest**")
#     k = st.slider("How many from each group?", 2, 6, 3, key="sample_k")

#     # indices
#     sort_idx = np.argsort(exp_lengths)
#     shortest_idx = sort_idx[:k]
#     longest_idx  = sort_idx[-k:][::-1]
#     # "typical": pick around median
#     med = int(np.median(exp_lengths))
#     typical_idx = np.argsort(np.abs(exp_lengths - med))[:k]

#     def samples_bar(indices, title):
#         df = exp_df.iloc[indices][["traj_idx", "states"]].sort_values("states")
#         fig = px.bar(df, x="states", y="traj_idx", orientation="h",
#                      labels={"states":"states", "traj_idx":"traj"},
#                      title=title)
#         fig.update_layout(height=180, margin=dict(l=0,r=0,t=40,b=0), showlegend=False)
#         return fig, df

#     c1, c2, c3 = st.columns(3)
#     with c1:
#         fig_s, df_s = samples_bar(shortest_idx, "Shortest")
#         st.plotly_chart(fig_s, use_container_width=True)
#     with c2:
#         fig_t, df_t = samples_bar(typical_idx, "Typical (≈ median)")
#         st.plotly_chart(fig_t, use_container_width=True)
#     with c3:
#         fig_l, df_l = samples_bar(longest_idx, "Longest")
#         st.plotly_chart(fig_l, use_container_width=True)

#     # pick a trajectory to drill down
#     st.markdown("Pick a trajectory to inspect states")
#     sel_traj_idx = st.number_input("trajectory index", min_value=0, max_value=int(exp_df["traj_idx"].max()), value=int(df_t["traj_idx"].iloc[0]))
# else:
#     # ---------- LEVEL 2B: All trajectories heatmap ----------
#     st.subheader("All trajectories (one cell = a trajectory; color = # states)")
#     max_traj = int(exp_df["traj_idx"].max())
#     # pack the expert's trajectories into one row to keep it compact
#     z = [ [np.nan]*(max_traj+1) ]
#     for _, r in exp_df.iterrows():
#         z[0][int(r["traj_idx"])] = int(r["states"])
#     heat = go.Figure(data=go.Heatmap(
#         z=z, x=list(range(max_traj+1)), y=[sel_expert],
#         coloraxis="coloraxis",
#         hovertemplate="expert=%{y}<br>traj=%{x}<br>#states=%{z}<extra></extra>"
#     ))
#     heat.update_layout(height=140, margin=dict(l=0,r=0,t=0,b=0), coloraxis=dict(colorscale="Blues"))
#     chosen = plotly_events(heat, click_event=True, select_event=False)
#     if chosen:
#         sel_traj_idx = int(chosen[0]["x"])
#     else:
#         sel_traj_idx = st.slider("Choose a trajectory index", 0, max_traj, 0)

# # ---------- LEVEL 3: States viewer ----------
# states = dataset[sel_expert][sel_traj_idx]
# state_count = len(states)
# st.markdown(f"### States for `{sel_expert}` · traj `{sel_traj_idx}`  —  {state_count} states")

# # A thin "state runway" to show hover = 126-d vector (truncated preview)
# X = np.array(states, dtype=float)   # [n_states, 126]
# preview = np.round(X[:,:6], 3).tolist()  # show first 6 dims in hover; full vector on click below

# state_scatter = go.Figure(data=go.Scatter(
#     x=list(range(state_count)), y=[0]*state_count, mode="markers",
#     marker=dict(size=6),
#     customdata=preview,
#     hovertemplate=("state=%{x}<br>dims[0..5]=%{customdata}<br>(126-dim vector)<extra></extra>")
# ))
# state_scatter.update_layout(height=180, margin=dict(l=0,r=0,t=10,b=0), yaxis=dict(visible=False))
# st.plotly_chart(state_scatter, use_container_width=True)

# # Detailed vector view (click or slider)
# sel_state = st.slider("Inspect state index", 0, max(0, state_count-1), 0)
# vec = X[sel_state]
# feat_df = pd.DataFrame({"feature":[f"f{i}" for i in range(len(vec))], "value":vec})
# vec_bar = px.bar(feat_df, x="feature", y="value", labels={"value":"feature value"})
# vec_bar.update_layout(height=300, margin=dict(l=0,r=0,t=10,b=20), xaxis=dict(showticklabels=False))
# st.plotly_chart(vec_bar, use_container_width=True)
# st.dataframe(feat_df, use_container_width=True, height=220)

###########################################################################################

# --------------------------
# 3) SELECTION LOGIC
# --------------------------
# Streamlit doesn’t capture clickData directly from st.plotly_chart,
# so we give users explicit selectors that mirror the sunburst.
# (In Dash, you could use fig clickData to drive selection.)
expert_ids = list(dataset.keys())
sel_expert = st.selectbox("Pick an expert", expert_ids)

traj_count = len(dataset[sel_expert])
st.write(f"**{sel_expert}** has **{traj_count}** trajectories.")

# Trajectory summary for this expert
traj_lengths = [len(t) for t in dataset[sel_expert]]
traj_df = pd.DataFrame({"trajectory_idx": range(traj_count), "states": traj_lengths})

cA, cB = st.columns([2, 3], gap="large")

with cA:
    st.markdown("**Trajectory lengths** (states per trajectory)")
    tl_fig = px.histogram(traj_df, x="states", nbins=30)
    tl_fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=300)
    st.plotly_chart(tl_fig, width='stretch')

with cB:
    st.markdown("**Trajectories list** (click one)")
    sel_traj_idx = st.selectbox("Trajectory", traj_df["trajectory_idx"], index=0)
    st.dataframe(traj_df, width='stretch', height=300)

# --------------------------
# 4) STATES VIEW + FEATURE VECTOR
# --------------------------
states = dataset[sel_expert][sel_traj_idx]
state_count = len(states)

# st.markdown(f"### States in {sel_expert} / traj_{sel_traj_idx}  —  {state_count} states")
st.markdown(f"### States in expert[{sel_expert}]/traj[{sel_traj_idx}]:  {state_count} states")

# Optional PCA view of states (each state is 126-d vector)
if st.checkbox("Show PCA of states (2D)"):
    X = np.array(states)  # shape: [n_states, 126]
    if X.ndim == 2 and X.shape[0] >= 2:
        # Simple PCA (no sklearn dependency)
        Xc = X - X.mean(0, keepdims=True)
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        Z = Xc @ Vt[:2].T
        pca_df = pd.DataFrame({"pc1": Z[:,0], "pc2": Z[:,1], "state_idx": range(len(states))})
        pfig = px.scatter(pca_df, x="pc1", y="pc2", hover_data=["state_idx"])
        pfig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=350)
        st.plotly_chart(pfig, width='stretch')
    else:
        st.info("Need at least 2 states to show PCA.")

# Choose a state to inspect
sel_state_idx = st.slider("Choose a state index", 0, max(0, state_count - 1), 0)
vec = np.array(states[sel_state_idx], dtype=float).reshape(-1)

st.markdown(f"**Feature vector (126 dims) for state {sel_state_idx}**")
feat_df = pd.DataFrame({"feature": [f"f{i}" for i in range(len(vec))], "value": vec})

bar = px.bar(feat_df, x="feature", y="value")
bar.update_layout(margin=dict(l=0, r=0, t=10, b=20), height=350, xaxis=dict(showticklabels=False))
st.plotly_chart(bar, width='stretch')
st.dataframe(feat_df, width='stretch', height=250)
