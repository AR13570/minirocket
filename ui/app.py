"""
app.py - MiniRocket Explorer: one web app with every feature of the project (Streamlit)

Author : Shreyas Shinde (Student ID 26104692)

Start it from the repository root:
    streamlit run ui/app.py
It opens in the browser at http://localhost:8501

This file is only the "frame": page set-up, the menu and the dataset chooser in the sidebar.
Each feature is its own page in views/ (listed in common.FEATURES):

    Overview            Home (all features at a glance)
    Classify & explain  Adaptive multi-pooling · Explain a prediction · Model insights
    Real time           Early classification
    Speed & size        Fast transform (joint PPV) · Group pruning · Class-stratified thresholds
    Evidence            Reproduction of the paper · About

Needs Streamlit 1.36 or newer (st.navigation).
"""
import streamlit as st

# Must be the first Streamlit call; it applies to every page.
st.set_page_config(page_title="MiniRocket Explorer", page_icon="📈", layout="wide")

import common                                     # noqa: E402  (paths, caches, feature list)
import service                                    # noqa: E402

# ---------------------------------------------------------------------------
# Menu: one page per feature, grouped into sections
# ---------------------------------------------------------------------------
sections = {"Overview": [st.Page("views/home.py", title="Home", icon="🏠", default=True)]}
for section, path, title, icon, _summary, _headline in common.FEATURES:
    sections.setdefault(section, []).append(st.Page(path, title=title, icon=icon))
sections["Evidence"].append(st.Page("views/about.py", title="About", icon="ℹ️"))
page = st.navigation(sections)

# ---------------------------------------------------------------------------
# Sidebar (below the menu): the dataset every page works on
# ---------------------------------------------------------------------------
names = list(service.DEMO_DATASETS)
st.sidebar.selectbox("Dataset", names, key="dataset")
st.sidebar.caption(service.DEMO_DATASETS[st.session_state["dataset"]])
st.sidebar.number_input("Random seed", min_value=0, max_value=99, value=0, step=1, key="seed")
st.sidebar.caption("Models for a dataset are trained once (about 10–60 s) and then reused by every page.")

page.run()
