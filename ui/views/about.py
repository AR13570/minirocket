"""About: the project in plain words, and the full model details for the chosen dataset."""
import streamlit as st

import common

common.page_header("About this project", "What MiniRocket is and what this project adds")
st.markdown("""
**MiniRocket** (Dempster, Schmidt and Webb, KDD 2021) classifies time series by sliding 84 fixed
pattern detectors over the series at several widths and counting how often each pattern appears
(PPV). A linear ridge classifier does the rest. It is one of the fastest accurate methods available.

This project reproduces MiniRocket and adds the following features, all available in this app:

| Feature | What it does | Page |
|---|---|---|
| Reproduction | re-runs MiniRocket on the UCR archive and compares with the published accuracy | Reproduction of the paper |
| Fast transform (joint PPV) | counts all thresholds of a kernel in one pass: identical features, less time | Fast transform |
| Group pruning | drops whole (dilation, kernel) groups, ranked on training data: smaller, faster models | Group pruning |
| Class-stratified thresholds | every class takes turns setting the thresholds | Class-stratified thresholds |
| Adaptive multi-pooling | adds pattern-strength features only where cross-validation shows they help | Adaptive multi-pooling |
| Explanations | splits the model's score exactly over the time steps (heat maps) | Explain a prediction, Model insights |
| Early classification | answers before the series ends once a learned confidence is reached | Early classification |

**Honest evaluation.** Every decision (pooling choice, pruned groups, confidence thresholds) uses the
training data only; the test data is used only to report results. Results on datasets never used
during development are reported separately.
""")
bundle = common.bundle_for_current()
with st.expander(f"Model details for {bundle.name}"):
    st.json(bundle.metadata)
