"""Home: every feature of the project at a glance, with links to its page."""
import streamlit as st

import common
import style

dataset, seed = common.current_dataset()
style.hero("MiniRocket Explorer",
           "Fast, accurate time series classification: reproduced, made faster and smaller, "
           "explained, and able to answer before a series has finished.",
           [("84", "fixed pattern detectors"), ("≈10,000", "features per series"), ("1", "linear classifier"),
            ("8", "features to explore")])
st.markdown("**MiniRocket** slides 84 fixed pattern detectors over a series, counts how often each "
            "pattern appears, and lets a linear classifier decide. This app brings together every "
            "improvement made in the project. Pick a dataset in the sidebar; every page works on it.")

bundle = common.bundle_for_current()
meta = bundle.metadata
c1, c2, c3, c4 = st.columns(4)
c1.metric(f"{dataset}: test accuracy", f"{100 * meta['test_accuracy']:.1f}%")
c2.metric("Feature set chosen", meta["chosen_feature_set"])
c3.metric("Early answer after", f"{100 * meta['early_metrics']['earliness']:.0f}% of a series")
c4.metric("Classes · train / test series", f"{len(meta['classes'])} · {meta['train_examples']} / {meta['test_examples']}")


def card(column, key):
    """A bordered box; with a key so the style sheet can animate it (plain box on older Streamlit)."""
    try:
        return column.container(border=True, key=key)
    except TypeError:                      # Streamlit without st.container(key=...)
        return column.container(border=True)


section_order = []
for feature in common.FEATURES:
    if feature[0] not in section_order:
        section_order.append(feature[0])
number = 0
for section in section_order:
    style.section_label(section)
    features = [f for f in common.FEATURES if f[0] == section]
    columns = st.columns(3)
    for i, (_section, path, title, icon, summary, headline) in enumerate(features):
        number += 1
        with card(columns[i % 3], f"mr_card_{number}"):
            style.card_head(icon, title, headline, section)
            st.write(summary)
            st.page_link(path, label="Open", icon="➡️")
