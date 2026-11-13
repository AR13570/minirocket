"""Group pruning: remove whole (dilation, kernel) groups for smaller, faster models."""
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import common

ps = common.pruning_service()
st.title("✂️ Group pruning")
if ps is None:
    common.feature_unavailable("group pruning")
dataset, seed = common.current_dataset()
st.markdown("MiniRocket's 10,000 features come in groups, one per (dilation, kernel) pair. Pruning "
            "ranks the groups **on the training data** and removes the least useful ones: their "
            "convolutions are never computed, so the model gets smaller and faster. Every kept feature "
            "is bit-for-bit the same as in the full model.")

c1, c2, c3 = st.columns([1.2, 1.2, 2])
scorer = c1.radio("Rank groups by", ps.SCORERS, key="prune_scorer", format_func={
    "ridge": "classifier weights (ridge, iterative)", "filter": "ANOVA F-score (filter, one shot)"}.get)
bias = c2.radio("Thresholds from", ps.BIAS_STRATEGIES, key="prune_bias", format_func={
    "original": "one random series (original)", "uniform": "random, NumPy RNG (control)",
    "stratified": "classes take turns (stratified)"}.get)
fractions = c3.multiselect("Share of groups to keep", [0.75, 0.5, 0.3, 0.2, 0.1],
                           default=[0.75, 0.5, 0.3, 0.2, 0.1], key="prune_fractions",
                           format_func=lambda f: f"{f:.0%}")


@st.cache_data(show_spinner=False)
def run_pruning(name, scorer, bias, seed, fractions):
    X_train, y_train, X_test, y_test = common.get_data(name)
    return ps.run_pruning(X_train, y_train, X_test, y_test, keep_fractions=fractions,
                          scorer=scorer, bias_strategy=bias, seed=seed)


try:
    with st.spinner(f"Pruning {dataset} (the first run also compiles the code, about 10–30 s)..."):
        result = run_pruning(dataset, scorer, bias, seed, tuple(sorted(fractions, reverse=True)))
except Exception as error:
    st.error(f"Could not run pruning on {dataset}: {error}")
    st.stop()

rows, full = result.rows, result.rows[0]
tab_try, tab_groups, tab_bench = st.tabs(["Try it", "Which groups survive", "Benchmark (113 datasets)"])

with tab_try:
    pruned_rows = rows[1:]
    if not pruned_rows:
        st.info("Choose at least one share of groups to keep.")
    else:
        good = [r for r in pruned_rows if r["accuracy_change_pp"] >= -1.0]   # within 1 pp of full
        pick = min(good, key=lambda r: r["keep_fraction"]) if good else pruned_rows[0]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Full model accuracy", f"{100 * full['accuracy']:.1f}%")
        m2.metric(f"Keeping {pick['keep_fraction']:.0%} of groups", f"{100 * pick['accuracy']:.1f}%",
                  f"{pick['accuracy_change_pp']:+.2f} pp")
        m3.metric("Test transform time", f"{pick['time_ratio']:.2f}× of full",
                  f"{(1 - pick['time_ratio']) * 100:.0f}% faster" if pick["time_ratio"] < 1 else None)
        m4.metric("Model size", f"{pick['model_kb']:.0f} KB", f"was {full['model_kb']:.0f} KB", delta_color="off")
        st.caption("Highlighted: the smallest model that loses at most 1 percentage point of accuracy.")

    st.dataframe(pd.DataFrame([{
        "keep": f"{r['keep_fraction']:.0%}", "groups kept": f"{r['groups_kept']:,} / {r['groups_total']:,}",
        "features": r["features_kept"], "accuracy %": round(100 * r["accuracy"], 2),
        "change (pp)": round(r["accuracy_change_pp"], 2), "test time vs full": round(r["time_ratio"], 2),
        "model size KB": round(r["model_kb"]), "identical to full model": "✓" if r["exact"] else "✗"}
        for r in rows]), hide_index=True)

    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.4))
    a.plot([r["keep_fraction"] for r in rows], [100 * r["accuracy"] for r in rows], marker="o", color=common.BLUE)
    a.set_xlabel("share of groups kept")
    a.set_ylabel("test accuracy (%)")
    a.invert_xaxis()
    a.set_title("Accuracy as groups are removed")
    a.grid(alpha=0.3)
    b.plot([r["time_ratio"] for r in rows], [100 * r["accuracy"] for r in rows], marker="o", color=common.ORANGE)
    for r in rows:
        b.annotate(f"{r['keep_fraction']:.0%}", (r["time_ratio"], 100 * r["accuracy"]),
                   textcoords="offset points", xytext=(4, 4), fontsize=8)
    b.set_xlabel("test transform time relative to the full model")
    b.set_ylabel("test accuracy (%)")
    b.set_title("Speed vs. accuracy (up and to the left is better)")
    b.grid(alpha=0.3)
    fig.tight_layout()
    common.show_figure(fig)
    st.caption("Times are measured on this computer (median of 3 runs), so tiny datasets are noisy. "
               "Controlled single-thread measurements are in the Benchmark tab.")

with tab_groups:
    options = sorted(r["keep_fraction"] for r in rows[1:])
    if not options:
        st.info("Choose at least one share of groups to keep.")
    else:
        chosen = st.select_slider("Share of groups kept", options=options, value=options[0],
                                  format_func=lambda f: f"{f:.0%}", key="prune_map")
        mask = result.masks[chosen]
        fig, ax = plt.subplots(figsize=(11, 0.35 * len(result.dilations) + 1.2))
        ax.imshow(mask, aspect="auto", cmap=matplotlib.colors.ListedColormap(["#EAECEE", common.GREEN]),
                  interpolation="nearest")
        ax.set_yticks(range(len(result.dilations)), [str(d) for d in result.dilations])
        ax.set_ylabel("dilation (pattern width)")
        ax.set_xlabel("kernel (0–83)")
        ax.set_title(f"Kept groups (green): {int(mask.sum())} of {mask.size}")
        fig.tight_layout()
        common.show_figure(fig)
        st.caption("Share of groups kept per dilation (small = short, sharp patterns; large = long, slow patterns):")
        st.bar_chart(pd.Series(mask.mean(axis=1), index=[f"d={d}" for d in result.dilations], name="share kept"))

with tab_bench:
    summary = ps.published_pruning_summary()
    if summary:
        st.dataframe(pd.DataFrame([{
            "ranking": r["scorer"], "keep": f"{r['keep_fraction']:.0%}", "features": round(r["features_kept"]),
            "accuracy %": round(r["accuracy_percent"], 2), "change (pp)": round(r["accuracy_change_pp"], 2),
            "test time vs full": round(r["time_ratio"], 3)} for r in summary]), hide_index=True)
        st.markdown("**Takeaways:** keeping half of the groups costs almost nothing with the *filter* "
                    "ranking (−0.05 pp) and makes the test transform about 43% faster; at 10% of the "
                    "groups the model loses 1.4–2.2 pp. Random groups do about as well as careful "
                    "selection: MiniRocket's features are highly redundant.")
    files = ps.published_files()
    plot = files.get("Accuracy vs. time plot (controls)")
    if plot:
        st.image(str(plot), caption="Accuracy vs. test-transform time, with random-group and smaller-model controls")
    for label in ("Controls (random groups, smaller models)", "Final paired tests"):
        if label in files:
            with st.expander(label):
                st.markdown(files[label].read_text(encoding="utf-8"))
