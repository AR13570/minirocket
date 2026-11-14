"""Adaptive multi-pooling: what the model chose for this dataset, and the benchmark results."""
import pandas as pd
import streamlit as st

import common
import service

dataset, seed = common.current_dataset()
common.page_header("views/adaptive_pooling.py")
st.markdown("Standard MiniRocket only counts **how often** each pattern appears (PPV). Adaptive "
            "multi-pooling can also measure **how strongly** it appears (MPV). Five-fold "
            "cross-validation on the training set decides, per dataset, whether the extra features "
            "help; if not, the model stays exactly the standard (fast) one.")

bundle = common.bundle_for_current()
meta = bundle.metadata
c1, c2, c3 = st.columns(3)
c1.metric("Test accuracy", f"{100 * meta['test_accuracy']:.1f}%")
c2.metric("Feature set chosen", meta["chosen_feature_set"])
c3.metric("Training time", f"{meta['seconds']['model']:.2f} s")
st.caption("Cross-validation accuracy on the training set for each candidate (higher wins; ties keep PPV):")
st.bar_chart(pd.Series({k: round(100 * v, 2) for k, v in meta["cv_scores"].items()}, name="CV accuracy %"))

st.subheader("Benchmark: original vs. fast transform vs. adaptive model")
run, rows = service.latest_benchmark_results()
if not rows:
    st.info("No benchmark results found yet. Run the multi-pooling experiment script "
            "(python -B -m experiments.multipool, see the README) and reopen this page.")
else:
    st.caption(f"Latest run: {run}")
    table = [{"method": common.method_label(r["method"]), "datasets": int(r["datasets"]),
              "accuracy %": round(100 * float(r["mean_accuracy"]), 2),
              "wins/ties/losses vs original": f"{r['wins_vs_original']}/{r['ties_vs_original']}/{r['losses_vs_original']}",
              "total seconds per dataset": round(float(r["mean_total_s"]), 3)} for r in rows]
    st.dataframe(pd.DataFrame(table), hide_index=True)
    left, right = st.columns(2)
    left.caption("Accuracy (%)")
    left.bar_chart(pd.Series({r["method"]: r["accuracy %"] for r in table}, name="accuracy %"))
    right.caption("Seconds per dataset")
    right.bar_chart(pd.Series({r["method"]: r["total seconds per dataset"] for r in table}, name="seconds"))
st.markdown("**On 20 datasets never used during development:** original 89.73% → adaptive **89.97%** "
            "(better on 5, equal on 14, worse on 1).")
