"""
warm_up.py - compile every feature's numba code once, so no page has a slow first visit

Author : Shreyas Shinde (Student ID 26104692)

Run from the project root (used by the Dockerfile; optional locally):
    python ui/warm_up.py [--dataset GunPoint]

Covers the pages that train_and_save.py does not: Fast transform (original + joint PPV),
Group pruning and Class-stratified thresholds. GunPoint ships with aeon, so no internet is needed.
"""
import argparse
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import service                       # noqa: E402  (paths and caches first)
import transform_service as ts       # noqa: E402



def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="GunPoint")
    args = parser.parse_args(argv)
    start = perf_counter()
    X_train, y_train, X_test, y_test = service.load_dataset(args.dataset)
    result = ts.compare_transforms(X_train, X_test, repetitions=1)
    print(f"fast transform: identical features = {result['identical']}")
    if (service.PROJECT_ROOT / "code" / "minirocket_pruned.py").exists():
        import pruning_service as ps
        for bias, scorer in (("original", "ridge"), ("stratified", "filter")):
            run = ps.run_pruning(X_train, y_train, X_test, y_test, keep_fractions=(0.5,),
                                 scorer=scorer, bias_strategy=bias, repetitions=1)
            print(f"group pruning ({bias}, {scorer}): all exact = {all(r['exact'] for r in run.rows)}")
    else:
        print("group pruning module not installed: skipped")
    print(f"warm-up finished in {perf_counter() - start:.1f} s")


if __name__ == "__main__":
    main()
