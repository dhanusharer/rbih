"""
Master Pipeline Orchestrator
Usage:
    python run_pipeline.py                    # full pipeline
    python run_pipeline.py --phase 1          # static features
    python run_pipeline.py --phase 2          # transaction features
    python run_pipeline.py --phase 3          # merge features
    python run_pipeline.py --phase 4          # train models
    python run_pipeline.py --phase 5          # tune hyperparams (optional)
    python run_pipeline.py --phase 6          # generate submission
    python run_pipeline.py --phase 7          # evaluation report
    python run_pipeline.py --skip-txn         # skip phase 2 if already cached
"""
from __future__ import annotations
import argparse, time
from pathlib import Path
from src.utils.config import CFG
from src.utils.logger import get_logger

log = get_logger()

PHASES = {
    1: ("Static features",       lambda: __import__("src.features.static_features", fromlist=["run"]).run()),
    2: ("Transaction features",  lambda: __import__("src.features.txn_features",    fromlist=["run"]).run()),
    21:("Graph & network features", lambda: __import__("src.features.graph_features", fromlist=["run"]).run()),
    22:("Precise temporal windows",  lambda: __import__("src.features.temporal_windows", fromlist=["run"]).run()),
    3: ("Merge features",        lambda: __import__("src.features.merge_features",   fromlist=["run"]).run()),
    4: ("Train models",          lambda: __import__("src.models.train",              fromlist=["train"]).train()),
    5: ("Hyperparameter tuning", lambda: __import__("src.models.tune",               fromlist=["run"]).run()),
    6: ("Generate submission",   lambda: __import__("src.models.predict",            fromlist=["run"]).run()),
    7: ("Evaluation report",     lambda: __import__("src.models.evaluate",           fromlist=["run"]).run()),
}


def main():
    parser = argparse.ArgumentParser(description="AML Mule Hunter Pipeline")
    parser.add_argument("--phase", type=int, default=None)
    parser.add_argument("--skip-txn",  action="store_true")
    parser.add_argument("--skip-tune", action="store_true", default=True)
    args = parser.parse_args()

    start = time.time()

    if args.phase:
        if args.phase not in PHASES:
            log.error(f"Unknown phase {args.phase}. Choose 1–7.")
            return
        name, fn = PHASES[args.phase]
        log.info(f"Running Phase {args.phase}: {name}")
        fn()
    else:
        skip = set()
        if args.skip_txn and Path(CFG.paths.txn_features).exists():
            log.info("Transaction features cached — skipping Phase 2.")
            skip.add(2)
        if args.skip_tune:
            skip.add(5)

        for num, (name, fn) in PHASES.items():
            if num in skip:
                log.info(f"Skipping Phase {num}: {name}")
                continue
            t0 = time.time()
            log.info(f"\n{'='*60}\nPHASE {num}: {name.upper()}\n{'='*60}")
            fn()
            log.info(f"Phase {num} done in {(time.time()-t0)/60:.1f} min")

    log.info(f"\n✅ Complete in {(time.time()-start)/60:.1f} min")
    log.info(f"   Submission: {CFG.paths.submissions}/")
    log.info(f"   Models:     {CFG.paths.models}/")
    log.info(f"   Reports:    {CFG.paths.reports}/")


if __name__ == "__main__":
    main()