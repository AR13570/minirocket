"""Fast transform (joint PPV): same features as the original MiniRocket transform, in less time."""
import pandas as pd
import streamlit as st

import common
import style
import transform_service as ts

dataset, seed = common.current_dataset()
common.page_header("views/fast_transform.py")
st.markdown("The original MiniRocket transform scans the convolution output **once per threshold** "
            "to count positive values. The joint-PPV transform counts **all thresholds of a kernel in "
            "one pass**. The features are bit-for-bit identical, so accuracy cannot change; only the "
            "time does.")


@st.cache_data(show_spinner=False)
def run_comparison(name, seed):
    X_train, _, X_test, _ = common.get_data(name)
    return ts.compare_transforms(X_train, X_test, seed=seed)


st.subheader(f"Live comparison on {dataset}")
with st.spinner("Timing both transforms (the first run also compiles them)..."):
    result = run_comparison(dataset, seed)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Original transform", f"{1000 * result['original_seconds']:.1f} ms")
c2.metric("Joint PPV transform", f"{1000 * result['joint_seconds']:.1f} ms",
          f"{(1 - result['ratio']) * 100:.0f}% less time" if result["ratio"] < 1 else
          f"{(result['ratio'] - 1) * 100:.0f}% more time", delta_color="normal" if result["ratio"] < 1 else "inverse")
c3.metric("Identical features", "yes ✓" if result["identical"] else "NO ✗")
c4.metric("Features per series", f"{result['features']:,}")
style.race_bars([
    ("Original transform", result["original_seconds"], f"{1000 * result['original_seconds']:.1f} ms", None),
    ("Joint PPV transform", result["joint_seconds"], f"{1000 * result['joint_seconds']:.1f} ms",
     (style.ROCKET, style.HEAT)),
])
st.caption(f"TEST set of {result['n_test']} series of length {result['length']}, median of 5 runs on this "
           "computer. Very small datasets take only milliseconds, so their timings are noisy.")

st.subheader("Benchmark: 113 UCR datasets × 3 seeds")
rows = ts.published_joint_ppv()
if not rows:
    st.info("The benchmark results file (results/joint_ppv_2026-10-26/summary.csv) was not found.")
else:
    st.dataframe(pd.DataFrame([{
        "features": f"{r['features']:,}", "accuracy % (both)": round(r["accuracy_percent"], 2),
        "identical output": "✓" if r["identical"] else "✗",
        "TRAIN transform: time saved": f"{r['train_reduction_percent']:.1f}%",
        "TEST transform: time saved": f"{r['test_reduction_percent']:.1f}%",
        "whole pipeline: time saved": f"{r['pipeline_reduction_percent']:.1f}%",
        "peak memory change": f"{r['memory_change_percent']:+.1f}%"} for r in rows]), hide_index=True)
    import matplotlib.pyplot as plt
    import numpy as np
    fig, ax = plt.subplots(figsize=(8, 3))
    positions = np.arange(len(rows))
    for i, (key, label, colour) in enumerate([("train_reduction_percent", "TRAIN transform", common.BLUE),
                                              ("test_reduction_percent", "TEST transform", common.ORANGE),
                                              ("pipeline_reduction_percent", "whole pipeline", common.GREY)]):
        bars = ax.bar(positions + (i - 1) * 0.27, [r[key] for r in rows], 0.25, label=label, color=colour)
        ax.bar_label(bars, fmt="%.1f%%", fontsize=7)
    ax.set_xticks(positions, [f"{r['features']:,} features" for r in rows])
    ax.set_ylabel("time saved (%)")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    common.show_figure(fig)
    st.markdown("The saving grows with the number of features, because more thresholds share one pass. "
                "Every one of the 1,017 paired comparisons produced identical features and predictions.")
