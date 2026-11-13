"""
common.py - shared pieces for every page of the MiniRocket Explorer

Author : Shreyas Shinde (Student ID 26104692)

  * the list of features (used by the menu in app.py and the cards on the home page)
  * cached loading of datasets and models (shared by all pages, built once per session)
  * the dataset / seed chosen in the sidebar
  * small plotting helpers and colours
"""
from pathlib import Path

import service                                    # first: sets up paths and numba caches
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

UI_FOLDER = Path(__file__).resolve().parent
PROJECT_ROOT = service.PROJECT_ROOT

RED, BLUE, GREEN, GREY, ORANGE = "#C0392B", "#2471A3", "#1E8449", "#B3B6B7", "#E69F00"

# ---------------------------------------------------------------------------
# Every feature of the project: menu section, page file, title, icon, one-line summary, headline
# (headline numbers come from the project's result files; see each page for the source)
# ---------------------------------------------------------------------------
FEATURES = [
    ("Classify & explain", "views/adaptive_pooling.py", "Adaptive multi-pooling", "🎯",
     "Each dataset chooses, from its training data only, whether pattern strength (MPV) helps.",
     "89.73% → 89.97% on 20 unseen datasets"),
    ("Classify & explain", "views/explain.py", "Explain a prediction", "🔍",
     "A heat map shows which parts of a series drove the decision; it adds up exactly to the score.",
     "Exact: heat map sums to the model's score"),
    ("Classify & explain", "views/insights.py", "Model insights", "🧭",
     "Which pattern widths and kernels the model relies on, and what separates the classes.",
     "Global and per-class evidence"),
    ("Real time", "views/early.py", "Early classification", "⏱️",
     "Answers before the series ends, once a learned confidence threshold is reached.",
     "93.8% accuracy using 37% of each series"),
    ("Speed & size", "views/fast_transform.py", "Fast transform (joint PPV)", "⚡",
     "Counts all thresholds of a kernel in one pass: same features, less time.",
     "Up to 49% faster, bit-identical output"),
    ("Speed & size", "views/group_pruning.py", "Group pruning", "✂️",
     "Removes whole (dilation, kernel) groups: smaller, faster models.",
     "Half the groups: −0.05 pp, 43% faster"),
    ("Speed & size", "views/stratified.py", "Class-stratified thresholds", "⚖️",
     "Lets every class take turns setting the thresholds instead of one random series.",
     "Fairer by design; no reliable accuracy change"),
    ("Evidence", "views/reproduction.py", "Reproduction of the paper", "📄",
     "Our MiniRocket accuracy against the published results, dataset by dataset.",
     "85.69% vs. 85.67% published (109 datasets)"),
]


# ---------------------------------------------------------------------------
# Data and models (cached for the whole session)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_bundle(name, seed):
    """Adaptive model + explainer + early classifier for a dataset (loaded, or trained once and saved)."""
    return service.load_bundle(name, seed=seed)


@st.cache_data(show_spinner=False)
def get_data(name):
    """X_train, y_train, X_test, y_test for a dataset (no model training)."""
    return service.load_dataset(name)


def current_dataset():
    """The dataset and seed chosen in the sidebar (set in app.py)."""
    names = list(service.DEMO_DATASETS)
    return st.session_state.get("dataset", names[0]), int(st.session_state.get("seed", 0))


def bundle_for_current():
    name, seed = current_dataset()
    with st.spinner(f"Preparing models for {name} (first time only, about 10–60 s)..."):
        return get_bundle(name, seed)


def pruning_service():
    """The group-pruning layer, or None if the group-pruning code (code/minirocket_pruned.py) is missing."""
    if not (PROJECT_ROOT / "code" / "minirocket_pruned.py").exists():
        return None
    import pruning_service as ps
    return ps


def feature_unavailable(name):
    st.info(f"The {name} module is not installed in this copy of the project. "
            "All other pages work as usual.")
    st.stop()


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------
METHOD_LABELS = {"original_10k": "Original MiniRocket", "joint_10k": "Fast transform (joint PPV)",
                 "arnav_joint_10k": "Fast transform (joint PPV)",      # id used in the older result files
                 "adaptive_10k": "Adaptive multi-pooling", "multipool_10k": "Multi-pooling (4 pools)",
                 "multipool_diff_10k": "Multi-pooling + difference series",
                 "multipool_diff_50k": "Multi-pooling + difference series (50k)"}


def method_label(method):
    """Readable feature name for a method id stored in the result files."""
    return METHOD_LABELS.get(method, method.replace("_", " "))


def show_figure(fig):
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


def heat_figure(x, heat, title):
    """Series drawn over a red/blue evidence strip."""
    heat = np.asarray(heat)
    limit = float(np.max(np.abs(heat))) or 1.0
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.imshow(heat[None, :], aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit,
              extent=(0, len(x) - 1, float(np.min(x)) - 0.1, float(np.max(x)) + 0.1), alpha=0.85)
    ax.plot(np.arange(len(x)), x, color="black", linewidth=1.6)
    ax.set_title(title)
    ax.set_xlabel("time step")
    fig.colorbar(plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(-limit, limit)),
                 ax=ax, label="evidence (red = for, blue = against)")
    fig.tight_layout()
    return fig
