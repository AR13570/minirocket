"""Early classification: replay a series arriving in real time and see when the model decides."""
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

import common
import service

st.title("⏱️ Early classification")
st.markdown("Can the model decide before the series ends? One model per prefix length, plus a "
            "confidence threshold learned from the training data: the model answers as soon as it "
            "is confident enough.")
bundle = common.bundle_for_current()
meta = bundle.metadata

index = st.slider("Test series to replay", 0, len(bundle.X_test) - 1, 0, key="early_index")
x = bundle.X_test[index]
timeline = service.early_timeline(bundle, x)
steps = timeline["steps"]
lengths = [s["points_seen"] for s in steps]
seen = st.select_slider("Points received so far", options=lengths, value=lengths[0], key="early_seen")
current = next(s for s in steps if s["points_seen"] == seen)
decision = timeline["decision"]
decided = decision["points_seen"] <= seen

fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(np.arange(len(x)), x, color=common.GREY, linewidth=1.4, label="not yet received")
ax.plot(np.arange(seen), x[:seen], color="black", linewidth=1.8, label="received")
if decided:
    ax.axvline(decision["points_seen"] - 1, color=common.GREEN, linewidth=2.5,
               label=f"decided: class {decision['prediction']}")
ax.legend(loc="upper right", fontsize=8)
ax.set_xlabel("time step")
fig.tight_layout()
common.show_figure(fig)

a, b, c = st.columns(3)
a.metric("Current guess", current["prediction"])
threshold_text = "never stops here" if current["threshold"] is None else f"{current['threshold']:.2f}"
b.metric("Confidence (margin)", f"{current['margin']:.2f}", f"threshold {threshold_text}", delta_color="off")
if decided:
    c.success(f"Decision: class {decision['prediction']} after "
              f"{100 * decision['fraction_seen']:.0f}% of the series (true class {bundle.y_test[index]})")
else:
    c.warning("Not confident yet - waiting for more data")

with st.expander("Confidence at every prefix length"):
    st.dataframe([{"points seen": s["points_seen"], "% of series": round(100 * s["fraction_seen"]),
                   "guess": s["prediction"], "margin": round(s["margin"], 3),
                   "threshold": "never" if s["threshold"] is None else round(s["threshold"], 3),
                   "confident": s["confident"]} for s in steps])
em = meta["early_metrics"]
st.caption(f"On the whole test set: {100 * em['accuracy']:.1f}% accuracy using on average "
           f"{100 * em['earliness']:.0f}% of each series (harmonic mean {em['harmonic_mean']:.3f}).")
