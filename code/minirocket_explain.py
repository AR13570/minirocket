"""
minirocket_explain.py
=====================

Author : Shreyas Shinde (Student ID 26104692)
Feature: Explainability for (adaptive multi-pooling) MiniRocket

WHAT IT ANSWERS
---------------
1. "Where in THIS series did the model look to make its prediction?"
   -> explain(x): a score for every time step (a heat map over the series).
2. "Which kernels / dilations / pooling features matter most overall?"
   -> global_importance()
3. "What temporal pattern separates class A from the other classes?"
   -> class_summaries(X, y)

HOW THE HEAT MAP IS COMPUTED (exact, not an approximation)
---------------------------------------------------------
The final classifier is linear (StandardScaler + Ridge). For class c:

    decision_c(x) = sum_f  w_cf * (F_f(x) - mean_f) / scale_f  +  b_c

Every MiniRocket feature F_f is itself a SUM over positions j of a convolution output C:

    PPV:  F = sum_j  [C_j > bias] / len(C)                  (each positive position adds 1/len(C))
    MPV:  F = sum_j  [C_j > bias] * (C_j - bias) / count     (each positive position adds its excess)

So the decision score can be split EXACTLY into one piece per position j. Each
position j of C comes from the kernel centred at a time step t and covering 9
time steps (t - 4d, ..., t, ..., t + 4d) for dilation d; we share the piece
equally between those 9 time steps. This gives a "raw map" raw_c(x)[t].

BASELINE: the classifier compares every feature with its average over the
training data (mean_f). We do the same per time step: we subtract the AVERAGE
raw map of the training series (the "background"). So heat[t] answers:
"compared with a typical training series, how much does THIS part of THIS
series push towards class c?" (same idea as SHAP with an average baseline).

Result: heat[t] for every time step t, and

    sum over t of heat[t]  ==  decision_c(x) - b_c        (checked by the tests)

Red (positive) time steps push the series TOWARDS class c; blue (negative)
time steps push it AWAY from class c.

The convolution outputs are recomputed exactly the way the original MiniRocket
code computes them (same float32 operations), so they match the model's features.
"""

from itertools import combinations

import numpy as np

# The 84 MiniRocket kernels: positions (out of 9) of the three weights equal to 2.
# Same order as the table in the original code (itertools.combinations(range(9), 3)).
KERNEL_POSITIONS = np.array(list(combinations(range(9), 3)), dtype=np.int32)


# ---------------------------------------------------------------------------
# 1. Which feature belongs to which kernel / dilation / bias?
# ---------------------------------------------------------------------------
def feature_table(representation_parameters):
    """
    For every original MiniRocket feature (in output order) return its kernel index,
    dilation, padding flag and bias. Same loop order as the transform:
        for each dilation -> for each of the 84 kernels -> for each bias
    """
    dilations, per_dilation, biases = representation_parameters
    kernels, dils, padded = [], [], []
    for dilation_index, dilation in enumerate(dilations):
        for kernel_index in range(84):
            uses_padding = ((dilation_index % 2) + kernel_index) % 2 == 0   # original rule
            for _ in range(per_dilation[dilation_index]):
                kernels.append(kernel_index)
                dils.append(int(dilation))
                padded.append(uses_padding)
    return {"kernel": np.array(kernels), "dilation": np.array(dils),
            "padded": np.array(padded), "bias": np.asarray(biases, dtype=np.float32)}


# ---------------------------------------------------------------------------
# 2. Convolution outputs of ONE series (same float32 steps as the original code)
# ---------------------------------------------------------------------------
def convolution_outputs(x, dilation):
    """
    Return (C_alpha, C_gamma) for one series and one dilation, built exactly like the
    original MiniRocket transform. Kernel k's output is then
        C_alpha + C_gamma[i0] + C_gamma[i1] + C_gamma[i2]   (i0, i1, i2 = KERNEL_POSITIONS[k]).
    """
    x = np.asarray(x, dtype=np.float32)
    L = len(x)
    padding = ((9 - 1) * dilation) // 2
    A = -x                       # weight -1
    G = x + x + x                # weight  2 is stored as 3x (because alpha + gamma = -1 + 3 = 2)
    C_alpha = A.copy()
    C_gamma = np.zeros((9, L), dtype=np.float32)
    C_gamma[4] = G
    start, end = dilation, L - padding
    for gamma_index in range(4):
        C_alpha[-end:] = C_alpha[-end:] + A[:end]
        C_gamma[gamma_index, -end:] = G[:end]
        end += dilation
    for gamma_index in range(5, 9):
        C_alpha[:-start] = C_alpha[:-start] + A[start:]
        C_gamma[gamma_index, :-start] = G[start:]
        start += dilation
    return C_alpha, C_gamma


# ---------------------------------------------------------------------------
# 3. The explainer
# ---------------------------------------------------------------------------
class MiniRocketExplainer:
    """
    Usage:
        model = AdaptivePoolingMiniRocket().fit(X_train, y_train)
        explainer = MiniRocketExplainer(model, X_train)   # training series = background
        result = explainer.explain(X_test[0])            # heat map for one series
        explainer.plot_explanation(X_test[0], "heatmap.png")
    """

    def __init__(self, model, X_background, max_background=200, random_state=0):
        self.model = model
        self.parameters = model.parameters_["raw"]
        self.table = feature_table(self.parameters)
        self.num_base = len(self.table["bias"])
        self.uses_mpv = "raw_MPV" in model.chosen_blocks_

        # Pull the linear model apart: scaler (mean, scale) and ridge (weights, intercept).
        scaler, ridge = model.classifier_[0], model.classifier_[-1]
        self.classes = ridge.classes_
        self.mean = scaler.mean_
        self.scale = scaler.scale_
        self.coef = np.atleast_2d(ridge.coef_)            # one row per score (1 row if 2 classes)
        self.intercept = np.atleast_1d(ridge.intercept_)
        self.effective = self.coef / self.scale            # w_f / scale_f, per score row

        # Background = average raw map of (up to max_background) training series.
        X_background = np.asarray(X_background, dtype=np.float32)
        if len(X_background) > max_background:
            pick = np.random.default_rng(random_state).choice(len(X_background), max_background, replace=False)
            X_background = X_background[pick]
        self.background = np.mean([self._raw_maps(x) for x in X_background], axis=0)   # (L x rows)

        # The classifier centres features with the mean of the FULL training set (scaler.mean_).
        # If the background is a sample, its average differs slightly; this small constant
        # (shared equally over time) keeps the heat map adding up exactly to the score.
        L = X_background.shape[1]
        self.correction = (self.background.sum(axis=0) - self.effective @ self.mean) / L   # (rows,)

    # ---- which score row belongs to which class (handles the 2-class case) ---
    def _row_and_sign(self, class_index):
        if len(self.classes) == 2:
            # Ridge stores ONE score row for two classes: score > 0 means classes[1].
            return 0, (1.0 if class_index == 1 else -1.0)
        return class_index, 1.0

    # ---- the heat map ------------------------------------------------------
    def explain(self, x, target_class=None):
        """
        Explain one series x (1D array).
        target_class: class label to explain (default: the predicted class).
        Returns a dict with the heat map and the numbers behind it.
        """
        x = np.asarray(x, dtype=np.float32)
        predicted = self.model.predict(x[None, :])[0]
        target = predicted if target_class is None else target_class
        class_index = int(np.where(self.classes == target)[0][0])
        row, sign = self._row_and_sign(class_index)

        # heat = (this series' raw map - average raw map of the training series) + tiny correction
        heat = sign * (self._raw_maps(x)[:, row] - self.background[:, row] + self.correction[row])
        intercept = sign * self.intercept[row]
        return {"heat": heat, "predicted_class": predicted, "explained_class": target,
                "decision_score": float(np.sum(heat) + intercept), "intercept": float(intercept)}

    def _raw_maps(self, x):
        """
        Raw map of one series for every score row: maps[t, r] = sum over features f of
        (w_rf / scale_f) * (the part of F_f(x) produced at time step t).
        Summing over t gives  sum_f (w_rf / scale_f) * F_f(x).
        """
        L = len(x)
        rows = self.effective.shape[0]
        w_ppv = self.effective[:, :self.num_base].T                                   # (features x rows)
        w_mpv = self.effective[:, self.num_base:2 * self.num_base].T if self.uses_mpv else None

        heat = np.zeros((L, rows), dtype=np.float64)
        dilations, per_dilation, biases = self.parameters
        feature = 0
        for dilation_index, dilation in enumerate(dilations):
            dilation = int(dilation)
            padding = ((9 - 1) * dilation) // 2
            C_alpha, C_gamma = convolution_outputs(x, dilation)
            k = int(per_dilation[dilation_index])

            for kernel_index in range(84):
                i0, i1, i2 = KERNEL_POSITIONS[kernel_index]
                C = C_alpha + C_gamma[i0] + C_gamma[i1] + C_gamma[i2]
                uses_padding = ((dilation_index % 2) + kernel_index) % 2 == 0
                if uses_padding:
                    centres = np.arange(L)                    # C[j] is centred at time j
                else:
                    C = C[padding:-padding]                   # only the valid part
                    centres = np.arange(padding, L - padding) # C[j] is centred at time j + padding

                b_k = biases[feature:feature + k]             # biases of this kernel/dilation
                positive = C[:, None] > b_k[None, :]          # (positions x biases), same test as the model
                n = len(C)

                # Piece of every score row contributed by each position j: (positions x rows)
                piece = positive @ (w_ppv[feature:feature + k] / n)                      # PPV part
                if w_mpv is not None:
                    count = positive.sum(axis=0)
                    safe = np.where(count > 0, count, 1)
                    excess = positive * (C[:, None].astype(np.float64) - b_k[None, :])
                    piece = piece + excess @ (w_mpv[feature:feature + k] / safe[:, None])  # MPV part

                # Share each position's piece equally between the 9 time steps the kernel covers.
                self._spread_over_receptive_field(heat, piece, centres, dilation, L)
                feature += k
        return heat

    @staticmethod
    def _spread_over_receptive_field(heat, piece, centres, dilation, L):
        """Add piece[j, :] to the (up to) 9 time steps covered by the kernel centred at centres[j]."""
        offsets = (np.arange(9) - 4) * dilation
        # How many of each position's 9 taps are real time steps (with zero padding some
        # taps fall outside the series); each real tap gets an equal share.
        real_taps = sum(((centres + o >= 0) & (centres + o < L)).astype(int) for o in offsets)
        for offset in offsets:
            taps = centres + offset
            inside = (taps >= 0) & (taps < L)
            np.add.at(heat, taps[inside], piece[inside] / real_taps[inside, None])

    # ---- global importance -------------------------------------------------
    def global_importance(self):
        """
        How much each part of MiniRocket matters to the classifier overall.
        Importance of a feature = size of its weights (over all classes), on standardised features.
        Returns sums grouped by dilation, by kernel and by pooling type, plus the top kernels.
        """
        size = np.sqrt(np.sum(self.coef ** 2, axis=0))            # one number per feature
        blocks = [("PPV", size[:self.num_base])]
        if self.uses_mpv:
            blocks.append(("MPV", size[self.num_base:2 * self.num_base]))
        per_feature = sum(block for _, block in blocks)            # PPV + MPV of the same kernel/bias

        by_dilation = {int(d): float(per_feature[self.table["dilation"] == d].sum())
                       for d in np.unique(self.table["dilation"])}
        by_kernel = np.array([per_feature[self.table["kernel"] == k].sum() for k in range(84)])
        by_pool = {name: float(block.sum()) for name, block in blocks}
        top = np.argsort(by_kernel)[::-1][:10]
        total = per_feature.sum()
        return {
            "by_dilation": by_dilation,
            "by_kernel": by_kernel,
            "by_pool": by_pool,
            "top_kernels": [{"kernel": int(k),
                             "weights": self.kernel_weights(int(k)).tolist(),
                             "share": float(by_kernel[k] / total)} for k in top],
        }

    @staticmethod
    def kernel_weights(kernel_index):
        """The 9 weights of a kernel: -1 everywhere, 2 at its three positions."""
        weights = -np.ones(9, dtype=int)
        weights[KERNEL_POSITIONS[kernel_index]] = 2
        return weights

    # ---- class-level summaries ----------------------------------------------
    def class_summaries(self, X, y, window_fraction=0.1):
        """
        For each class: average heat map over its examples (explaining that class), the
        time window with the strongest evidence, and how the class differs there from the
        other classes (mean level and slope).
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        L = X.shape[1]
        width = max(3, int(round(window_fraction * L)))
        summaries = []
        for cls in self.classes:
            members = X[y == cls]
            others = X[y != cls]
            heats = np.array([self.explain(x, target_class=cls)["heat"] for x in members])
            mean_heat = heats.mean(axis=0)

            # Strongest evidence window = highest moving-average of the mean heat.
            smooth = np.convolve(mean_heat, np.ones(width) / width, mode="valid")
            start = int(np.argmax(smooth))
            stop = start + width

            in_class = members[:, start:stop]
            in_other = others[:, start:stop]
            level_class, level_other = float(in_class.mean()), float(in_other.mean())
            slope_class = float(np.mean(np.diff(in_class, axis=1))) if width > 1 else 0.0
            slope_other = float(np.mean(np.diff(in_other, axis=1))) if width > 1 else 0.0

            text = (f"Class {cls}: strongest evidence at {100 * start / L:.0f}%-{100 * stop / L:.0f}% of the series "
                    f"(t = {start}-{stop - 1}). There the series is on average "
                    f"{'HIGHER' if level_class > level_other else 'LOWER'} than in other classes "
                    f"({level_class:.2f} vs {level_other:.2f}) and "
                    f"{'rising' if slope_class > 0 else 'falling'} (slope {slope_class:.3f} vs {slope_other:.3f}).")
            summaries.append({"class": cls, "examples": int(len(members)), "mean_heat": mean_heat,
                              "mean_series": members.mean(axis=0), "window": (start, stop),
                              "level_class": level_class, "level_other": level_other,
                              "slope_class": slope_class, "slope_other": slope_other, "text": text})
        return summaries

    # ---- plots ---------------------------------------------------------------
    def plot_explanation(self, x, save_path=None, true_class=None, title=None):
        """Series drawn on top of a red/blue heat strip; returns the matplotlib figure."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        result = self.explain(x)
        heat = result["heat"]
        limit = np.max(np.abs(heat)) or 1.0

        fig, ax = plt.subplots(figsize=(10, 3.6))
        # Heat strip behind the series: red = towards the predicted class, blue = against it.
        ax.imshow(heat[None, :], aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit,
                  extent=(0, len(x) - 1, float(np.min(x)) - 0.1, float(np.max(x)) + 0.1), alpha=0.85)
        ax.plot(np.arange(len(x)), x, color="black", linewidth=1.6)
        label = f"predicted: {result['predicted_class']}"
        if true_class is not None:
            label += f"   true: {true_class}"
        ax.set_title(title or f"Where the model looked ({label})")
        ax.set_xlabel("time step")
        ax.set_ylabel("value")
        mappable = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(-limit, limit))
        fig.colorbar(mappable, ax=ax, label="evidence for predicted class")
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
            plt.close(fig)
        return fig

    def plot_global_importance(self, save_path=None):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        info = self.global_importance()
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
        dils = list(info["by_dilation"])
        axes[0].bar([str(d) for d in dils], [info["by_dilation"][d] for d in dils], color="#4C72B0")
        axes[0].set_title("Importance by dilation (pattern width)")
        axes[0].set_xlabel("dilation")
        axes[0].tick_params(axis="x", rotation=90)
        top = info["top_kernels"]
        axes[1].barh([str(t["weights"]) for t in top][::-1], [100 * t["share"] for t in top][::-1], color="#DD8452")
        axes[1].set_title("Top 10 kernels (weights) - % of total importance")
        axes[1].tick_params(axis="y", labelsize=7)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
            plt.close(fig)
        return fig

    def plot_class_summaries(self, summaries, save_path=None):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(len(summaries), 1, figsize=(10, 2.6 * len(summaries)), squeeze=False)
        for ax, s in zip(axes[:, 0], summaries):
            heat = s["mean_heat"]
            limit = np.max(np.abs(heat)) or 1.0
            series = s["mean_series"]
            ax.imshow(heat[None, :], aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit,
                      extent=(0, len(series) - 1, float(series.min()) - 0.1, float(series.max()) + 0.1), alpha=0.85)
            ax.plot(series, color="black", linewidth=1.6, label="class mean")
            start, stop = s["window"]
            ax.axvspan(start, stop - 1, fill=False, edgecolor="green", linewidth=2, label="strongest evidence")
            ax.set_title(f"Class {s['class']} ({s['examples']} examples): average evidence map", fontsize=10)
            ax.legend(loc="upper right", fontsize=7)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
            plt.close(fig)
        return fig
