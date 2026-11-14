"""Model insights: what the model relies on overall, and what separates the classes."""
import streamlit as st

import common

common.page_header("views/insights.py")
bundle = common.bundle_for_current()
meta = bundle.metadata

st.subheader("What does the model rely on?")
common.show_figure(bundle.explainer.plot_global_importance())
st.caption("Left: pattern widths (dilation 1 = 9 consecutive points). Right: the most important kernels.")
st.subheader("What separates the classes?")
summaries = bundle.class_summaries          # precomputed when the model was built
for s in summaries:
    st.write("• " + s["text"])
common.show_figure(bundle.explainer.plot_class_summaries(summaries))
st.caption(f"Cross-validation on the training set: {meta['cv_scores']} → chose {meta['chosen_feature_set']}.")
