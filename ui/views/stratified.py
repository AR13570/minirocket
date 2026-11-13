"""Class-stratified thresholds: every class takes turns setting MiniRocket's thresholds."""
import numpy as np
import pandas as pd
import streamlit as st

import common
import minirocket                                # code/minirocket.py (original MiniRocket)

ps = common.pruning_service()
st.title("⚖️ Class-stratified thresholds")
if ps is None:
    common.feature_unavailable("class-stratified thresholds")
dataset, seed = common.current_dataset()
st.markdown("Each group of features gets its thresholds (biases) from **one training series**. In the "
            "original MiniRocket that series is chosen at random, so a rare class may hardly be used. "
            "**Stratified** thresholds let the classes take turns, so every class supplies about the "
            "same number of groups. *Uniform* is the control: the same code with a random choice.")

X_train, y_train, X_test, y_test = common.get_data(dataset)
dilations, _ = minirocket._fit_dilations(X_train.shape[1], 10_000, 32)   # the model's layout (no fitting)
groups = len(dilations) * 84
sources = ps.threshold_sources(y_train, groups, seed)
left, right = st.columns(2)
left.caption(f"Training series per class ({dataset})")
left.bar_chart(pd.Series(ps.class_balance(y_train), name="series"))
right.caption(f"Which class supplied each of the {groups:,} groups' thresholds")
right.bar_chart(pd.DataFrame(sources).rename(columns={"uniform": "random (uniform)"}))


@st.cache_data(show_spinner=False)
def compare(name, seeds):
    X_train, y_train, X_test, y_test = common.get_data(name)
    return ps.compare_bias_strategies(X_train, y_train, X_test, y_test, seeds=seeds)


st.subheader("Does it change accuracy on this dataset?")
if st.button("Compare the three strategies (3 seeds, 9 models)"):
    with st.spinner("Fitting 9 full models..."):
        comparison = compare(dataset, (0, 1, 2))
    st.dataframe(pd.DataFrame([{"thresholds from": k, "mean accuracy %": round(100 * np.mean(v), 2),
                                "per seed %": ", ".join(f"{100 * a:.1f}" for a in v)}
                               for k, v in comparison.items()]), hide_index=True)
st.info("**Result on all 113 datasets × 20 seeds:** stratified thresholds did not change accuracy "
        "reliably (stratified − random: −0.002 pp, Wilcoxon p = 0.377). A fairness idea that was "
        "tested properly and found neutral, not an accuracy gain.")
files = ps.published_files()
label = "Stratified biases, 113 datasets x 20 seeds"
if label in files:
    with st.expander("Full analysis (113 datasets × 20 seeds)"):
        st.markdown(files[label].read_text(encoding="utf-8"))
