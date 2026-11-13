"""Home: every feature of the project at a glance, with links to its page."""
import streamlit as st

import common

dataset, seed = common.current_dataset()
st.title("MiniRocket Explorer")
st.markdown("Time series classification with **MiniRocket**: 84 fixed pattern detectors slide over a "
            "series, the features count how often each pattern appears, and a linear classifier decides. "
            "This app brings together every improvement made in the project. Pick a dataset in the "
            "sidebar; every page works on it.")

bundle = common.bundle_for_current()
meta = bundle.metadata
c1, c2, c3, c4 = st.columns(4)
c1.metric(f"{dataset}: test accuracy", f"{100 * meta['test_accuracy']:.1f}%")
c2.metric("Feature set chosen", meta["chosen_feature_set"])
c3.metric("Early answer after", f"{100 * meta['early_metrics']['earliness']:.0f}% of a series")
c4.metric("Classes · train / test series", f"{len(meta['classes'])} · {meta['train_examples']} / {meta['test_examples']}")

st.subheader("Features")
section_order = []
for feature in common.FEATURES:
    if feature[0] not in section_order:
        section_order.append(feature[0])
for section in section_order:
    st.markdown(f"**{section}**")
    features = [f for f in common.FEATURES if f[0] == section]
    columns = st.columns(3)
    for i, (_section, path, title, icon, summary, headline) in enumerate(features):
        with columns[i % 3].container(border=True):
            st.markdown(f"#### {icon} {title}")
            st.write(summary)
            st.caption(headline)
            st.page_link(path, label="Open", icon="➡️")
