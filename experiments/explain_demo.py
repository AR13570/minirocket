"""
Explanation demo: train the final (adaptive) model on one dataset and explain it

Author : Shreyas Shinde (Student ID 26104692)

Run from the repository root:
    python -B -m experiments.explain_demo                                   # GunPoint (ships with aeon)
    python -B -m experiments.explain_demo --dataset ECG200 --data ./data/Univariate_ts/
    python -B -m experiments.explain_demo --dataset Lightning7 --examples 4

Saves in --output-dir (default results/local/explain_<dataset>/) :
    example_<i>.png        heat map over individual test series ("where the model looked")
    global_importance.png  which dilations and kernels matter most
    class_summaries.png    average evidence map per class + strongest window
    report.md              accuracy, chosen feature set, class-level text summaries
"""
import argparse
import time
from pathlib import Path

import numpy as np

from experiments.multipool import load_dataset
from minirocket_adaptive import AdaptivePoolingMiniRocket
from minirocket_explain import MiniRocketExplainer


def main():
    parser = argparse.ArgumentParser(description="Explain the final MiniRocket model on one dataset")
    parser.add_argument("--dataset", default="GunPoint")
    parser.add_argument("--data", default=None, help="local UCR archive folder (default: aeon download)")
    parser.add_argument("--examples", type=int, default=3, help="number of test series to explain")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    out = Path(args.output_dir) if args.output_dir else Path("results") / "local" / f"explain_{args.dataset}"
    out.mkdir(parents=True, exist_ok=True)

    # 1. Data and model
    X_train, y_train, X_test, y_test = load_dataset(args.dataset, args.data)
    print(f"{args.dataset}: train {X_train.shape}, test {X_test.shape}")
    t = time.perf_counter()
    model = AdaptivePoolingMiniRocket(random_state=args.seed).fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(f"Final model: accuracy {100 * accuracy:.2f}%, feature set {model.chosen_feature_set_} "
          f"({time.perf_counter() - t:.2f}s)")

    # 2. Explainer (training series = background)
    t = time.perf_counter()
    explainer = MiniRocketExplainer(model, X_train)
    print(f"Explainer ready ({time.perf_counter() - t:.2f}s)")

    # 3. Individual explanations: one test series per class first, then more if requested
    picks = []
    for cls in model.classifier_.classes_:
        idx = np.where(y_test == cls)[0]
        if len(idx):
            picks.append(int(idx[0]))
    picks = (picks + [i for i in range(len(X_test)) if i not in picks])[:args.examples]
    for number, i in enumerate(picks, start=1):
        path = out / f"example_{number}.png"
        explainer.plot_explanation(X_test[i], save_path=path, true_class=y_test[i])
        print(f"  saved {path.name} (test series {i}, true class {y_test[i]})")

    # 4. Global importance
    info = explainer.global_importance()
    explainer.plot_global_importance(save_path=out / "global_importance.png")

    # 5. Class-level summaries (explaining each class on its own training examples)
    summaries = explainer.class_summaries(X_train, y_train)
    explainer.plot_class_summaries(summaries, save_path=out / "class_summaries.png")

    # 6. Written report
    lines = [f"# Explainability report: {args.dataset}", "",
             f"* Test accuracy: **{100 * accuracy:.2f}%**",
             f"* Feature set chosen from training data: **{model.chosen_feature_set_}**",
             f"* Training examples: {len(X_train)}, test examples: {len(X_test)}, series length: {X_train.shape[1]}", "",
             "## What separates the classes", ""]
    lines += [f"* {s['text']}" for s in summaries]
    lines += ["", "## Most important dilations (pattern widths)", ""]
    for d, value in sorted(info["by_dilation"].items(), key=lambda kv: -kv[1])[:5]:
        lines.append(f"* dilation {d}: kernel spans {8 * d + 1} time steps, "
                     f"{100 * value / sum(info['by_dilation'].values()):.1f}% of total importance")
    lines += ["", "## Most important kernels", ""]
    for k in info["top_kernels"][:5]:
        lines.append(f"* kernel {k['kernel']} weights {k['weights']}: {100 * k['share']:.1f}%")
    if len(info["by_pool"]) > 1:
        total = sum(info["by_pool"].values())
        lines += ["", "## Pooling features", ""] + \
                 [f"* {name}: {100 * v / total:.1f}% of importance" for name, v in info["by_pool"].items()]
    (out / "report.md").write_text("\n".join(lines) + "\n")

    print("\nClass summaries:")
    for s in summaries:
        print("  " + s["text"])
    print(f"\nSaved everything to {out}")


if __name__ == "__main__":
    main()
