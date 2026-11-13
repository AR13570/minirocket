"""Explain a prediction: heat map of the evidence over time (exact decomposition of the score)."""
import pandas as pd
import streamlit as st

import common
import service

st.title("🔍 Explain a prediction")
st.markdown("Why did the model decide that? The heat map spreads the model's score over the time "
            "steps: red parts push towards the class, blue parts push away, and the colours add up "
            "exactly to the score.")
bundle = common.bundle_for_current()

source = st.radio("Series to classify", ["Test series from the dataset", "Upload my own (CSV)"], horizontal=True)
true_label = None
series = None
if source == "Test series from the dataset":
    index = st.slider("Test series number", 0, len(bundle.X_test) - 1, 0, key="explain_index")
    series = bundle.X_test[index]
    true_label = str(bundle.y_test[index])
else:
    upload = st.file_uploader("CSV with one series (one row or one column of numbers)", type=["csv", "txt"])
    if upload is not None:
        try:
            series = service.parse_series_csv(upload.getvalue())
        except ValueError as error:
            st.error(str(error))

if series is not None:
    prediction = service.predict(bundle, series)
    target = st.selectbox("Explain evidence for class", ["(predicted class)"] + bundle.classes)
    explanation = service.explain(bundle, series, None if target == "(predicted class)" else target)
    x, _ = service.prepare_series(series, bundle.series_length)
    if prediction["note"]:
        st.info(prediction["note"])

    a, b, c = st.columns(3)
    a.metric("Prediction", prediction["predicted_class"])
    if true_label is not None:
        correct = true_label == prediction["predicted_class"]
        b.metric("True class", true_label, "correct" if correct else "wrong",
                 delta_color="normal" if correct else "inverse")
    c.metric("Confidence (margin)", f"{prediction['margin']:.2f}")

    common.show_figure(common.heat_figure(x, explanation["heat"],
                                          f"Evidence for class {explanation['explained_class']}"))
    st.write(explanation["summary"])
    with st.expander("Scores per class"):
        st.bar_chart(pd.Series(prediction["scores"], name="score"))
