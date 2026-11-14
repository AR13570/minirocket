"""Early classification: replay a series arriving in real time and see when the model decides."""
import time

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

import common
import service
import style

common.page_header("views/early.py")
st.markdown("Can the model decide before the series ends? One model per prefix length, plus a "
            "confidence threshold learned from the training data: the model answers as soon as it "
            "is confident enough. Press **Play** to watch a series arrive, or step through it by hand.")
bundle = common.bundle_for_current()
meta = bundle.metadata

index = st.slider("Test series to replay", 0, len(bundle.X_test) - 1, 0, key="early_index")
x = bundle.X_test[index]
timeline = service.early_timeline(bundle, x)
steps = timeline["steps"]
decision = timeline["decision"]
lengths = [s["points_seen"] for s in steps]


def stream_figure(seen):
    """The series with the part received so far highlighted, and the decision line once it is passed."""
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(np.arange(len(x)), x, color=common.GREY, linewidth=1.2, alpha=0.55, label="not yet received")
    ax.plot(np.arange(seen), x[:seen], color=common.ink(), linewidth=2.0, label="received")
    if seen < len(x):
        ax.scatter([seen - 1], [x[seen - 1]], color=common.ORANGE, s=42, zorder=5)       # the newest point
    if decision["points_seen"] <= seen:
        ax.axvline(decision["points_seen"] - 1, color=common.GREEN, linewidth=2.5,
                   label=f"decided: class {decision['prediction']}")
    ax.set_xlim(-1, len(x))
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("time step")
    fig.tight_layout()
    return fig


def step_at(seen):
    """The model's latest answer when `seen` points have arrived (None before the first prefix)."""
    reached = [s for s in steps if s["points_seen"] <= seen]
    return reached[-1] if reached else None


left, right = st.columns([1, 3])
play = left.button("▶ Play the stream", key="early_play")
seen = right.select_slider("Points received so far (manual)", options=lengths, value=lengths[0], key="early_seen")
plot_area = st.empty()
status_area = st.empty()

if play:
    # ---- live replay: about 45 frames from the first points to the full series ----
    announced = False
    for frame in np.unique(np.linspace(max(2, len(x) // 45), len(x), 45).astype(int)):
        figure = stream_figure(int(frame))
        plot_area.pyplot(figure, clear_figure=True)
        plt.close(figure)
        now = step_at(int(frame))
        if decision["points_seen"] <= frame:
            chip = style.status_chip(f"Decided: class {decision['prediction']} after "
                                     f"{100 * decision['fraction_seen']:.0f}% of the series", style.EARLY)
            if not announced:
                st.toast(f"Decision: class {decision['prediction']} at "
                         f"{100 * decision['fraction_seen']:.0f}% of the series", icon="✅")
                announced = True
        elif now is None:
            chip = style.status_chip("Listening: too few points for a first guess", style.MUTED)
        else:
            chip = style.status_chip(f"Current guess: class {now['prediction']}, confidence "
                                     f"{now['margin']:.2f} (not confident yet)", style.ROCKET)
        status_area.markdown(chip, unsafe_allow_html=True)
        time.sleep(0.05)
    seen = len(x)                                    # the summary below describes the full replay
else:
    figure = stream_figure(seen)
    plot_area.pyplot(figure, clear_figure=True)
    plt.close(figure)

current = step_at(seen)
decided = decision["points_seen"] <= seen
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
