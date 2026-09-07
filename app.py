import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from src.skill_extraction import detect_columns, extract_all_skills_for_row
from src.preprocessing import clean_text, preprocessing_demo_example
from src.similarity import compute_similarity_matrix
from src.graph import build_graph
from src.community import run_louvain
from src.labeling import label_all_communities
from src.recommendations import (
    recommend_related_skills, recommend_similar_students,
    recommend_communities_for_skills, related_skill_dictionary_lookup,
)
from src.skill_dictionary import CATEGORY_COLORS
from src.data_generator import generate_demo_dataset

st.set_page_config(
    page_title="Student Skill Community Analyzer",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------
# GLOBAL STYLE
# ----------------------------------------------------------------------
st.markdown("""
<style>
    .main > div {padding-top: 1.2rem;}
    .kpi-card {
        background: linear-gradient(135deg, #ffffff 0%, #f4f6ff 100%);
        border: 1px solid #e6e9f5;
        border-radius: 14px;
        padding: 18px 16px;
        text-align: center;
        box-shadow: 0 2px 10px rgba(60, 60, 120, 0.06);
    }
    .kpi-value {font-size: 28px; font-weight: 700; color: #2b2d42;}
    .kpi-label {font-size: 13px; color: #6c6f80; margin-top: 2px; text-transform: uppercase; letter-spacing: .04em;}
    .section-title {font-size: 22px; font-weight: 700; color: #2b2d42; margin-top: 6px;}
    .section-sub {color: #6c6f80; font-size: 14px; margin-bottom: 10px;}
    .pill {
        display: inline-block; padding: 3px 12px; border-radius: 999px;
        font-size: 12px; font-weight: 600; color: white; margin: 2px 4px 2px 0;
    }
    div[data-testid="stMetricValue"] {font-weight: 700;}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------
# SESSION STATE
# ----------------------------------------------------------------------
def init_state():
    defaults = {
        "df": None, "colmap": None, "student_ids": None, "skills_map": None,
        "similarity_matrix": None, "graph": None, "partition": None,
        "modularity": None, "community_info": None, "vectorizer": None,
        "detected": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


# ----------------------------------------------------------------------
# SIDEBAR - DATA INPUT + SETTINGS
# ----------------------------------------------------------------------
st.sidebar.title("🕸️ Skill Community Analyzer")
st.sidebar.caption("NLP + Louvain Community Detection")
st.sidebar.markdown("---")

st.sidebar.subheader("1. Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV dataset", type=["csv"])
load_demo = st.sidebar.button("📂 Load Demo Dataset", use_container_width=True)

if uploaded_file is not None:
    st.session_state.df = pd.read_csv(uploaded_file)
    st.session_state.detected = False
    st.sidebar.success(f"Loaded {len(st.session_state.df)} rows from upload.")

if load_demo:
    rows = generate_demo_dataset(n_students=90, seed=42)
    st.session_state.df = pd.DataFrame(rows)
    st.session_state.detected = False
    st.sidebar.success("Demo dataset loaded (90 synthetic student profiles).")

st.sidebar.markdown("---")
st.sidebar.subheader("2. Graph Settings")
graph_mode = st.sidebar.radio("Edge rule", ["Similarity threshold", "K-Nearest Neighbours"], index=0)
if graph_mode == "Similarity threshold":
    threshold = st.sidebar.slider("Similarity threshold", 0.05, 0.9, 0.25, 0.05)
    k_val = 5
    st.sidebar.caption(
        f"An edge is created between two students only if their skill-set "
        f"cosine similarity is **≥ {threshold}**. Higher values → sparser, "
        f"tighter graph."
    )
else:
    k_val = st.sidebar.slider("K (neighbours per student)", 2, 15, 5, 1)
    threshold = 0.0
    st.sidebar.caption(
        f"Each student is connected to their **top-{k_val}** most similar "
        f"peers. This keeps the graph connected even for niche skill sets."
    )

resolution = st.sidebar.slider("Louvain resolution", 0.5, 2.0, 1.0, 0.1,
                                help="Higher resolution -> more, smaller communities.")

st.sidebar.markdown("---")
run_btn = st.sidebar.button("🚀 Detect Communities", type="primary", use_container_width=True)

st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["Dashboard", "NLP & Skills", "Communities", "Network", "Students", "Skills", "Recommendations", "About Project"],
)


# ----------------------------------------------------------------------
# PIPELINE (cached where expensive)
# ----------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def extract_pipeline(df: pd.DataFrame):
    colmap = detect_columns(df.columns)
    if not colmap.get("student_id"):
        df = df.reset_index().rename(columns={"index": "student_id"})
        colmap["student_id"] = "student_id"
        df["student_id"] = df["student_id"].apply(lambda i: f"S{int(i)+1:03d}")

    student_ids = df[colmap["student_id"]].astype(str).tolist()
    skills_list = [extract_all_skills_for_row(row, colmap) for _, row in df.iterrows()]
    skills_map = dict(zip(student_ids, skills_list))
    return colmap, student_ids, skills_list, skills_map


@st.cache_data(show_spinner=False)
def similarity_pipeline(skills_list):
    sim, _ = compute_similarity_matrix(skills_list)
    return sim


def run_full_pipeline():
    df = st.session_state.df
    colmap, student_ids, skills_list, skills_map = extract_pipeline(df)
    sim = similarity_pipeline(skills_list)
    mode = "threshold" if graph_mode == "Similarity threshold" else "knn"
    G = build_graph(sim, student_ids, mode=mode, threshold=threshold, k=k_val)
    partition, modularity = run_louvain(G, resolution=resolution)
    community_info = label_all_communities(partition, skills_map)

    st.session_state.colmap = colmap
    st.session_state.student_ids = student_ids
    st.session_state.skills_map = skills_map
    st.session_state.similarity_matrix = sim
    st.session_state.graph = G
    st.session_state.partition = partition
    st.session_state.modularity = modularity
    st.session_state.community_info = community_info
    st.session_state.detected = True


if run_btn:
    if st.session_state.df is None:
        st.sidebar.error("Please load or upload a dataset first.")
    else:
        with st.spinner("Running NLP extraction, similarity, graph build & Louvain..."):
            run_full_pipeline()


def kpi_card(col, value, label):
    col.markdown(
        f"""<div class="kpi-card"><div class="kpi-value">{value}</div>
        <div class="kpi-label">{label}</div></div>""",
        unsafe_allow_html=True,
    )


def community_color(cid):
    palette = px.colors.qualitative.Set2 + px.colors.qualitative.Set3
    return palette[cid % len(palette)]


@st.cache_data(show_spinner=False)
def compute_layout(_G, seed=42):
    return nx.spring_layout(_G, seed=seed, k=None, weight="weight")


def render_network(G, partition, skills_map, community_info, max_nodes_label=300):
    pos = compute_layout(G)

    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=0.5, color="rgba(150,150,170,0.35)"),
        hoverinfo="none", showlegend=False,
    )

    traces = [edge_trace]
    communities = sorted(set(partition.values()))
    for cid in communities:
        nodes = [n for n in G.nodes() if partition.get(n) == cid]
        xs = [pos[n][0] for n in nodes]
        ys = [pos[n][1] for n in nodes]
        sizes = [8 + 1.5 * G.degree(n) for n in nodes]
        hover = [
            f"<b>{n}</b><br>Community: {community_info[cid]['label']}<br>"
            f"Skills: {', '.join(skills_map.get(n, [])[:8])}"
            for n in nodes
        ]
        traces.append(go.Scatter(
            x=xs, y=ys, mode="markers", name=f"{community_info[cid]['label']} ({len(nodes)})",
            marker=dict(size=sizes, color=community_color(cid), line=dict(width=1, color="white")),
            text=hover, hoverinfo="text",
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        height=580,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, x=0.5, xanchor="center"),
        margin=dict(t=10, b=10, l=10, r=10),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        dragmode="pan",
    )
    return fig


# ----------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------
st.markdown(
    "<div style='display:flex; align-items:center; gap:14px;'>"
    "<div style='font-size:38px;'>🕸️</div>"
    "<div><div style='font-size:32px; font-weight:800; color:#2b2d42;'>Student Skill Community Analyzer</div>"
    "<div style='font-size:15px; color:#6c6f80;'>Discover hidden skill communities using NLP and Graph Analytics</div></div>"
    "</div>", unsafe_allow_html=True,
)
st.markdown("<hr style='margin-top:10px; margin-bottom: 18px;'>", unsafe_allow_html=True)

if st.session_state.df is None:
    st.info("👈 Load the **Demo Dataset** or upload your own CSV from the sidebar, then click **Detect Communities**.")
    st.stop()


# ========================================================================
# PAGE: DASHBOARD
# ========================================================================
if page == "Dashboard":
    st.markdown("<div class='section-title'>📊 Dataset Preview</div>", unsafe_allow_html=True)
    st.dataframe(st.session_state.df.head(10), use_container_width=True, height=250)

    if not st.session_state.detected:
        st.warning("Click **🚀 Detect Communities** in the sidebar to run the full pipeline.")
        st.stop()

    G = st.session_state.graph
    partition = st.session_state.partition
    n_communities = len(set(partition.values()))
    connected_students = sum(1 for n in G.nodes() if G.degree(n) > 0)

    st.markdown("<div class='section-title'>📈 Key Metrics</div>", unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    kpi_card(c1, len(st.session_state.student_ids), "Total Students")
    kpi_card(c2, connected_students, "Students Connected")
    kpi_card(c3, n_communities, "Communities Detected")
    kpi_card(c4, G.number_of_edges(), "Graph Edges")
    kpi_card(c5, f"{st.session_state.modularity:.3f}", "Modularity Score")

    st.write("")
    st.markdown("<div class='section-title'>🥧 Community Distribution</div>", unsafe_allow_html=True)
    info = st.session_state.community_info
    dist_df = pd.DataFrame({
        "Community": [f"{info[c]['label']}" for c in info],
        "Size": [info[c]["size"] for c in info],
    })
    fig_pie = px.pie(dist_df, names="Community", values="Size", hole=0.45,
                      color_discrete_sequence=px.colors.qualitative.Set2)
    fig_pie.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=340)
    st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("<div class='section-title'>🌐 Interactive Student Network</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Each color = one Louvain-detected community. Hover a node for details. Scroll/drag to zoom & pan.</div>", unsafe_allow_html=True)
    st.plotly_chart(render_network(G, partition, st.session_state.skills_map, st.session_state.community_info),
                     use_container_width=True)

    st.markdown("<div class='section-title'>📋 Community Overview</div>", unsafe_allow_html=True)
    info = st.session_state.community_info
    total = len(st.session_state.student_ids)
    rows = []
    for cid, d in info.items():
        top_skills = ", ".join([s for s, _ in d["top_skills"][:5]])
        rows.append({
            "Community": f"Community {cid}",
            "Label": d["label"],
            "Students": d["size"],
            "Top Skills": top_skills,
            "% of Students": f"{100 * d['size'] / total:.1f}%",
        })
    st.dataframe(pd.DataFrame(rows).sort_values("Students", ascending=False),
                 use_container_width=True, hide_index=True)

    colC, colD = st.columns(2)
    with colC:
        st.markdown("<div class='section-title'>🏆 Top Skills Overall</div>", unsafe_allow_html=True)
        from collections import Counter
        overall = Counter()
        for skills in st.session_state.skills_map.values():
            overall.update(skills)
        top_overall = overall.most_common(15)
        skills_df = pd.DataFrame(top_overall, columns=["Skill", "Count"])
        fig_bar = px.bar(skills_df, x="Count", y="Skill", orientation="h",
                          color="Count", color_continuous_scale="Purples")
        fig_bar.update_layout(yaxis=dict(autorange="reversed"), height=420,
                               margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_bar, use_container_width=True)

    with colD:
        st.markdown("<div class='section-title'>📦 Community Sizes</div>", unsafe_allow_html=True)
        size_df = pd.DataFrame({
            "Community": [info[c]["label"] for c in info],
            "Students": [info[c]["size"] for c in info],
        }).sort_values("Students", ascending=False)
        fig_bar2 = px.bar(size_df, x="Community", y="Students", color="Community",
                           color_discrete_sequence=px.colors.qualitative.Set2)
        fig_bar2.update_layout(height=420, margin=dict(t=10, b=10, l=10, r=10), showlegend=False)
        st.plotly_chart(fig_bar2, use_container_width=True)


# ========================================================================
# PAGE: NLP & SKILLS
# ========================================================================
elif page == "NLP & Skills":
    st.markdown("<div class='section-title'>🧹 NLP Preprocessing</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Lowercasing → tokenization → punctuation cleaning → stopword removal → lemmatization.</div>", unsafe_allow_html=True)

    before, after = preprocessing_demo_example()
    c1, c2 = st.columns(2)
    c1.text_area("Before", before, height=90, disabled=True)
    c2.text_area("After", after, height=90, disabled=True)

    st.markdown("##### Try it yourself")
    custom_text = st.text_input("Enter any sentence", "Skilled in React, Node.js and MongoDB for full-stack web apps.")
    if custom_text:
        st.code(clean_text(custom_text), language=None)

    st.markdown("---")
    st.markdown("<div class='section-title'>🏷️ Skill Extraction</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Hybrid approach: structured 'skills' column parsing + dictionary-based multi-word phrase matching over free text.</div>", unsafe_allow_html=True)

    if not st.session_state.detected:
        st.info("Run **Detect Communities** first to see extracted skills for the whole dataset (or preview below).")

    df = st.session_state.df
    colmap = detect_columns(df.columns)
    st.write("**Detected column mapping:**")
    st.json({k: v for k, v in colmap.items() if v})

    preview_n = min(8, len(df))
    preview_rows = []
    for _, row in df.head(preview_n).iterrows():
        skills = extract_all_skills_for_row(row, colmap)
        sid_col = colmap.get("student_id")
        sid = row[sid_col] if sid_col else "?"
        preview_rows.append({"Student": sid, "Extracted Skills": ", ".join(skills)})
    st.dataframe(pd.DataFrame(preview_rows), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("<div class='section-title'>📐 Similarity Example</div>", unsafe_allow_html=True)
    from src.similarity import similarity_demo_example
    ex_a = ["python", "machine learning", "tensorflow", "deep learning"]
    ex_b = ["python", "pandas", "sql", "power bi"]
    ex_c = ["html", "css", "javascript", "react"]
    sim_ex = similarity_demo_example(ex_a, ex_b, ex_c)
    st.write("Skill sets compared:")
    st.write(f"- **Student A** (AI/ML-leaning): {', '.join(ex_a)}")
    st.write(f"- **Student B** (Data Science-leaning): {', '.join(ex_b)}")
    st.write(f"- **Student C** (Web Dev-leaning): {', '.join(ex_c)}")
    sim_df = pd.DataFrame(sim_ex, index=["A", "B", "C"], columns=["A", "B", "C"])
    st.dataframe(sim_df.style.background_gradient(cmap="Purples", axis=None).format("{:.2f}"),
                 use_container_width=True)
    st.caption("A & B share 'python' → moderate similarity. A & C share nothing → similarity ≈ 0.")


# ========================================================================
# PAGE: COMMUNITIES
# ========================================================================
elif page == "Communities":
    if not st.session_state.detected:
        st.warning("Click **🚀 Detect Communities** in the sidebar first.")
        st.stop()

    info = st.session_state.community_info
    st.markdown("<div class='section-title'>🧩 Detected Communities</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-sub'>Louvain discovers these groups purely from the weighted similarity graph. "
        "Labels below are generated <b>afterwards</b> by inspecting each community's dominant skills.</div>",
        unsafe_allow_html=True,
    )

    community_options = {f"Community {cid} — {d['label']} ({d['size']} students)": cid for cid, d in info.items()}
    selected_label = st.selectbox("Select a community to explore", list(community_options.keys()))
    cid = community_options[selected_label]
    d = info[cid]

    c1, c2, c3 = st.columns(3)
    kpi_card(c1, d["label"], "Community Label")
    kpi_card(c2, d["size"], "Number of Students")

    members = d["students"]
    sim = st.session_state.similarity_matrix
    student_ids = st.session_state.student_ids
    idxs = [student_ids.index(m) for m in members]
    if len(idxs) > 1:
        sub_sim = sim[np.ix_(idxs, idxs)]
        avg_sim = sub_sim[np.triu_indices(len(idxs), k=1)].mean() if len(idxs) > 1 else 0.0
    else:
        avg_sim = 0.0
    kpi_card(c3, f"{avg_sim*100:.1f}%", "Avg. Intra-Community Similarity")

    st.write("")
    colA, colB = st.columns([1, 1])
    with colA:
        st.markdown("##### Top 10 Skills")
        top_df = pd.DataFrame(d["top_skills"], columns=["Skill", "Count"])
        fig = px.bar(top_df, x="Count", y="Skill", orientation="h", color="Count",
                      color_continuous_scale="Teal")
        fig.update_layout(yaxis=dict(autorange="reversed"), height=380, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

    with colB:
        st.markdown("##### Category Score Breakdown (labeling logic)")
        cat_scores = d["category_scores"]
        if cat_scores:
            cat_df = pd.DataFrame(list(cat_scores.items()), columns=["Category", "Score"]).sort_values("Score", ascending=False)
            fig2 = px.bar(cat_df, x="Category", y="Score", color="Category",
                          color_discrete_map=CATEGORY_COLORS)
            fig2.update_layout(height=380, margin=dict(t=10, b=10, l=10, r=10), showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown("##### Students in this Community")
    rows = []
    skills_map = st.session_state.skills_map
    for m in members:
        rows.append({"Student ID": m, "Skills": ", ".join(skills_map.get(m, []))})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=300)


# ========================================================================
# PAGE: NETWORK
# ========================================================================
elif page == "Network":
    if not st.session_state.detected:
        st.warning("Click **🚀 Detect Communities** in the sidebar first.")
        st.stop()

    st.markdown("<div class='section-title'>🌐 Full Interactive Network</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Node size = degree (number of connections). Color = detected community. Zoom, pan, and hover freely.</div>", unsafe_allow_html=True)
    st.plotly_chart(
        render_network(st.session_state.graph, st.session_state.partition,
                        st.session_state.skills_map, st.session_state.community_info),
        use_container_width=True,
    )

    G = st.session_state.graph
    degrees = dict(G.degree())
    top_connected = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:10]
    st.markdown("##### Most Connected Students (Hubs)")
    hub_df = pd.DataFrame(top_connected, columns=["Student", "Connections"])
    hub_df["Community"] = hub_df["Student"].apply(
        lambda s: st.session_state.community_info[st.session_state.partition[s]]["label"]
    )
    st.dataframe(hub_df, use_container_width=True, hide_index=True)


# ========================================================================
# PAGE: STUDENTS
# ========================================================================
elif page == "Students":
    if not st.session_state.detected:
        st.warning("Click **🚀 Detect Communities** in the sidebar first.")
        st.stop()

    st.markdown("<div class='section-title'>🎓 Student Explorer</div>", unsafe_allow_html=True)
    student_ids = st.session_state.student_ids
    selected_student = st.selectbox("Search / select a student", student_ids)

    skills_map = st.session_state.skills_map
    partition = st.session_state.partition
    community_info = st.session_state.community_info
    sim = st.session_state.similarity_matrix

    cid = partition[selected_student]
    label = community_info[cid]["label"]

    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown(f"### {selected_student}")
        st.markdown(
            f"<span class='pill' style='background:{CATEGORY_COLORS.get(label.split(' (')[0], '#636E72')}'>{label}</span>",
            unsafe_allow_html=True,
        )
        st.write("**Extracted Skills:**")
        st.write(", ".join(skills_map.get(selected_student, [])) or "_none detected_")

    with c2:
        st.markdown("##### Most Similar Students")
        sims = recommend_similar_students(selected_student, sim, student_ids, top_n=5)
        if sims:
            sim_df = pd.DataFrame(sims, columns=["Student", "Similarity %"])
            fig = px.bar(sim_df, x="Similarity %", y="Student", orientation="h", color="Similarity %",
                         color_continuous_scale="Blues", range_x=[0, 100])
            fig.update_layout(yaxis=dict(autorange="reversed"), height=280, margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)
            for sid, pct in sims:
                st.write(f"- **{sid}** — {pct}%")
        else:
            st.write("No similar students found (isolated node).")


# ========================================================================
# PAGE: SKILLS
# ========================================================================
elif page == "Skills":
    if not st.session_state.detected:
        st.warning("Click **🚀 Detect Communities** in the sidebar first.")
        st.stop()

    st.markdown("<div class='section-title'>🛠️ Skill Explorer</div>", unsafe_allow_html=True)
    skills_map = st.session_state.skills_map
    partition = st.session_state.partition
    community_info = st.session_state.community_info

    all_skills = sorted({s for skills in skills_map.values() for s in skills})
    selected_skill = st.selectbox("Select a skill", all_skills, index=all_skills.index("python") if "python" in all_skills else 0)

    students_with_skill = [sid for sid, sk in skills_map.items() if selected_skill in sk]
    st.metric("Students with this skill", len(students_with_skill))

    comm_counts = {}
    for sid in students_with_skill:
        cid = partition[sid]
        lbl = community_info[cid]["label"]
        comm_counts[lbl] = comm_counts.get(lbl, 0) + 1

    colA, colB = st.columns(2)
    with colA:
        st.markdown("##### Community Distribution for this Skill")
        if comm_counts:
            dist_df = pd.DataFrame(list(comm_counts.items()), columns=["Community", "Students"])
            fig = px.pie(dist_df, names="Community", values="Students", hole=0.4,
                         color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(height=340, margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)
            strongest = max(comm_counts, key=comm_counts.get)
            st.success(f"Most strongly associated community: **{strongest}**")

    with colB:
        st.markdown("##### Related Skills (dictionary-based)")
        related = related_skill_dictionary_lookup(selected_skill, top_n=10)
        if related:
            for r in related:
                st.write(f"- {r}")
        else:
            st.write("_No curated related skills found for this term._")

    st.markdown("##### Students with this skill")
    st.dataframe(pd.DataFrame({"Student": students_with_skill}), use_container_width=True, hide_index=True, height=200)


# ========================================================================
# PAGE: RECOMMENDATIONS
# ========================================================================
elif page == "Recommendations":
    if not st.session_state.detected:
        st.warning("Click **🚀 Detect Communities** in the sidebar first.")
        st.stop()

    st.markdown("<div class='section-title'>💡 Recommendations</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Rule / similarity based — no complex recommender needed for this demo.</div>", unsafe_allow_html=True)

    mode = st.radio("Recommend based on:", ["Existing student", "Custom skill list"], horizontal=True)

    skills_map = st.session_state.skills_map
    partition = st.session_state.partition
    community_info = st.session_state.community_info
    sim = st.session_state.similarity_matrix
    student_ids = st.session_state.student_ids

    if mode == "Existing student":
        sid = st.selectbox("Select student", student_ids)
        student_skills = skills_map.get(sid, [])
        cid = partition[sid]
        community_members = community_info[cid]["students"]
        community_skill_lists = [skills_map.get(m, []) for m in community_members]

        st.write(f"**{sid}'s current skills:** {', '.join(student_skills) or '—'}")
        st.write(f"**Community:** {community_info[cid]['label']}")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### 🎯 Recommended Skills to Learn")
            rec_skills = recommend_related_skills(student_skills, community_skill_lists, top_n=8)
            for s in rec_skills:
                st.write(f"- {s}")
        with c2:
            st.markdown("##### 👥 Recommended Peers")
            sims = recommend_similar_students(sid, sim, student_ids, top_n=5)
            for peer, pct in sims:
                st.write(f"- **{peer}** ({pct}% similar)")

    else:
        raw = st.text_input("Enter comma-separated skills", "python, react")
        typed_skills = [s.strip().lower() for s in raw.split(",") if s.strip()]
        if typed_skills:
            st.write(f"**Your skills:** {', '.join(typed_skills)}")

            st.markdown("##### 🏘️ Relevant Communities")
            comm_matches = recommend_communities_for_skills(typed_skills, community_info, top_n=3)
            if comm_matches:
                for cid, label, overlap in comm_matches:
                    st.write(f"- **{label}** — {overlap} matching skill(s)")
            else:
                st.write("_No strong community match found for these skills._")

            st.markdown("##### 🎯 Related Skills to Explore")
            related_all = set()
            for s in typed_skills:
                related_all |= set(related_skill_dictionary_lookup(s, top_n=6))
            related_all -= set(typed_skills)
            for r in sorted(related_all)[:10]:
                st.write(f"- {r}")


# ========================================================================
# PAGE: ABOUT PROJECT
# ========================================================================
elif page == "About Project":
    st.markdown("<div class='section-title'>📘 About This Project</div>", unsafe_allow_html=True)
    st.markdown("""
**Title:** Detecting Student Skill Communities on LinkedIn Using NLP and Louvain Community Detection

**Pipeline:**
1. **Dataset** — CSV upload or synthetic demo data (90 profiles) with realistic overlapping skills.
2. **NLP Preprocessing** — lowercasing, tokenization, punctuation cleaning, stopword removal, lemmatization.
3. **Skill Extraction** — hybrid: structured `skills` column parsing + dictionary-based multi-word phrase matching over free text (summary, projects, certifications, posts).
4. **Student Representation** — TF-IDF vectorization over each student's extracted skill set.
5. **Similarity Calculation** — cosine similarity between every pair of students.
6. **Weighted Graph** — NetworkX graph; edges via similarity threshold or k-NN (both configurable in the sidebar).
7. **Louvain Community Detection** — `python-louvain` (with a NetworkX built-in fallback), run directly on the weighted graph — communities are discovered, never hardcoded.
8. **Community Labeling** — *after* detection, dominant skills in each community are scored against a curated skill→category dictionary to generate a human-readable label (e.g. "AI/ML").
9. **Visualization** — interactive Plotly network graph, KPI dashboard, community/skill/student explorers.
10. **Recommendations** — rule/similarity-based skill, peer and community suggestions.

**Tech stack:** Python, Pandas, Scikit-learn, NetworkX, python-louvain, Streamlit, Plotly.

**Important academic note:** Louvain operates purely on the weighted similarity graph structure (edges + weights). It has no knowledge of the category dictionary used for labeling — the dictionary is applied strictly *after* community detection, only to name the communities for human readability.
""")

    if st.session_state.detected:
        st.markdown("---")
        st.markdown("##### Current Run Summary")
        st.json({
            "students": len(st.session_state.student_ids),
            "edges": st.session_state.graph.number_of_edges(),
            "communities_detected": len(set(st.session_state.partition.values())),
            "modularity": round(st.session_state.modularity, 4),
        })
