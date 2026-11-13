"""Reproduction of the paper: our MiniRocket accuracy vs. the published accuracy."""
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import common
import transform_service as ts

st.title("📄 Reproduction of the paper")
st.markdown("Before improving MiniRocket we re-ran it on the UCR archive with the published settings "
            "(the paper's first train/test split, 10,000 features, ridge classifier) and compared the "
            "accuracy with the results published by the authors, dataset by dataset.")
summary, rows = ts.published_reproduction()
if summary is None:
    common.feature_unavailable("reproduction results")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Our mean accuracy", f"{summary['ours_percent']:.2f}%")
c2.metric("Published (same split)", f"{summary['paper_percent']:.2f}%",
          f"{summary['ours_percent'] - summary['paper_percent']:+.2f} pp", delta_color="off")
c3.metric("Within 1 pp of the paper", f"{summary['within_1pp']} of {summary['datasets']}")
p = summary["wilcoxon_p"]
c4.metric("Systematic difference?", "no" if p is not None and p > 0.05 else "check",
          f"Wilcoxon p = {p:.2f}" if p is not None else None, delta_color="off")
st.caption(f"{summary['identical']} datasets match exactly; {summary['within_2_stds']} differ by at most two "
           "standard deviations of the paper's own run-to-run variation. The paper's headline "
           f"{summary['paper_30_resamples_percent']:.2f}% averages 30 different splits, so it is not the "
           "like-for-like number.")

fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter([100 * r["paper"] for r in rows], [100 * r["ours"] for r in rows], s=14, color=common.BLUE)
ax.plot([0, 100], [0, 100], color=common.GREY, linewidth=1)
ax.set_xlim(40, 101)
ax.set_ylim(40, 101)
ax.set_xlabel("published accuracy (%)")
ax.set_ylabel("our accuracy (%)")
ax.set_title("One dot per dataset (on the line = identical)")
fig.tight_layout()
left, _ = st.columns([3, 2])
with left:
    common.show_figure(fig)

st.subheader("Largest differences")
largest = sorted(rows, key=lambda r: -abs(r["difference_pp"]))[:10]
st.dataframe(pd.DataFrame([{"dataset": r["dataset"], "published %": round(100 * r["paper"], 2),
                            "ours %": round(100 * r["ours"], 2), "difference (pp)": round(r["difference_pp"], 2),
                            "in published std devs": round(r["difference_in_stds"], 2)} for r in largest]),
             hide_index=True)
