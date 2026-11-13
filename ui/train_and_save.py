"""
train_and_save.py - build the model artifacts the UI and API load (the "training pipeline")

Author : Shreyas Shinde (Student ID 26104692)

Run from the repository root:
    python ui/train_and_save.py                       # all demo datasets
    python ui/train_and_save.py --datasets GunPoint ECG200

For each dataset it trains the three modules, measures them on the test set and saves
    ui/artifacts/<dataset>_seed<seed>.joblib   the model bundle
    ui/artifacts/<dataset>_seed<seed>.json     readable metadata (accuracy, versions, date, ...)
After this, the UI and API start instantly (no training while a user waits).
In deployment this step runs once while building the Docker image (see DEPLOYMENT.md).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import service

import argparse
import time


def main():
    parser = argparse.ArgumentParser(description="Train and save model artifacts for the UI/API")
    parser.add_argument("--datasets", nargs="+", default=list(service.DEMO_DATASETS))
    parser.add_argument("--data-dir", default=None, help="folder with UCR .ts files (default: aeon download)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default=None, help="default: ui/artifacts")
    args = parser.parse_args()

    for name in args.datasets:
        t = time.perf_counter()
        try:
            bundle = service.build_bundle(name, data_dir=args.data_dir, seed=args.seed)
        except Exception as error:
            print(f"{name:<18} FAILED: {error}")
            continue
        path = service.save_bundle(bundle, args.output_dir)
        m = bundle.metadata
        print(f"{name:<18} accuracy {100 * m['test_accuracy']:6.2f}%  features {m['chosen_feature_set']:<8} "
              f"early {100 * m['early_metrics']['accuracy']:6.2f}% using {100 * m['early_metrics']['earliness']:4.0f}%  "
              f"({time.perf_counter() - t:.1f}s) -> {path.name}")


if __name__ == "__main__":
    main()
