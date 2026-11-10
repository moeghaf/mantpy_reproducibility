"""``mantpy-liver``: rerun the Schistosoma-infected mouse liver experiments from the command line.

Stages
------
prepare    download the 24 ROIs, normalise them, extract patches, remove background, compute coverage
search     label-free search of patch size, encoder settings and K (NSGA-II; resumable; any number of workers)
freeze     choose the encoder and K from the finished search by the stability rule
tune       within-fold tuning of the classifier (TPE; one held-out mouse per task)
final      fit the selected settings: encoder seeds, classifiers, integrated gradients, logistic
           regression, permutation test, tuned classifiers
report     descriptive statistics, scores and the comparison with the paper
reproduce  prepare + final + report with the settings selected in the paper
all        prepare + search + freeze + tune + final + report, using this run's own selections
tasks      print the independent tasks of 'final' and 'tune' (for cluster job arrays)
"""

from __future__ import annotations

import argparse
import os
import sys

STAGES = ["prepare", "search", "freeze", "tune", "final", "report", "reproduce", "all", "tasks"]
PARALLEL_STAGES = ("search", "tune", "final")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mantpy-liver", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--results", default="results/liver", help="results directory (shared by all workers)")
    parser.add_argument("--cache-dir", default=None, help="dataset cache (default: MANTPY_CACHE or ~/.cache/mantpy)")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                        help="device of the encoder and of the tuned classifiers; the fixed classifier always runs on the CPU")
    parser.add_argument("--threads", type=int, default=16,
                        help="CPU threads of each classifier fit, also written to OMP_NUM_THREADS and the other thread "
                             "variables of every stage (the paper's runs used 16; results change slightly with this number)")
    parser.add_argument("--workers", type=int, default=1,
                        help="search, tune and final: processes to run in parallel on this machine")
    parser.add_argument("--selection", default=None, choices=["paper", "search"],
                        help="settings used by tune/final/report: the paper's (default; 'all' uses 'search') or this "
                             "directory's own selections")
    parser.add_argument("--partition", default=None, choices=["paper", "refit"],
                        help="clusters analysed downstream: the partition deposited with the paper (default with "
                             "--selection paper) or the one refitted in this directory (always with --smoke)")
    parser.add_argument("--n-encoders", type=int, default=None,
                        help="search and freeze: new encoders proposed by NSGA-II before the search ends "
                             "(default 60; 2 with --smoke)")
    parser.add_argument("--n-trials", type=int, default=None,
                        help="tune: configurations per held-out mouse (default 80; 3 with --smoke)")
    parser.add_argument("--time-limit-hours", type=float, default=None,
                        help="search: stop after this many hours; the command then exits with code 3 and running it "
                             "again continues the search")
    parser.add_argument("--worker-index", type=int, default=0,
                        help="search: number of this worker when several run at once (each proposes different settings)")
    parser.add_argument("--task-index", type=int, default=None, help="final or tune: run only this task (cluster job arrays)")
    parser.add_argument("--smoke", action="store_true",
                        help="quick installation check: a crop of each ROI and tiny training budgets; "
                             "results are not the paper's")
    parser.add_argument("--worker-slice", default=None, help=argparse.SUPPRESS)
    return parser


def _spawn_workers(args, argv) -> int:
    """Run ``--workers`` copies of this command; the result store keeps them from duplicating work."""
    from ..runtime import run_workers

    base = [a for i, a in enumerate(argv) if a != "--workers" and (i == 0 or argv[i - 1] != "--workers")]
    return run_workers([[sys.executable, "-m", "mantpy_repro.liver", *base, "--workers", "1",
                         "--worker-slice", f"{index}/{args.workers}"] for index in range(args.workers)])


def _thread_environment(threads: int) -> None:
    """Set the thread variables to ``--threads``. NumPy and PyTorch read them when they are loaded."""
    os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(threads)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = _parser().parse_args(argv)
    _thread_environment(args.threads)      # before the imports below, which load NumPy

    from ..runtime import UNFINISHED, exit_on_sigterm
    from .config import N_MICE
    from .final import final_tasks, task_ranges

    if args.stage == "tasks":
        ranges = task_ranges(args.smoke)
        print(f"final: {len(final_tasks(args.smoke))} tasks (use --task-index 0..{len(final_tasks(args.smoke)) - 1})")
        for kind, (first, last) in ranges.items():
            print(f"  {first:>3}-{last:<3} {kind}")
        print(f"tune: {N_MICE} tasks, one held-out mouse each (use --task-index 0..{N_MICE - 1})")
        return 0
    if args.task_index is not None and args.stage not in ("final", "tune"):
        raise SystemExit("--task-index applies to the 'final' and 'tune' stages only.")
    if args.task_index is not None:
        count = len(final_tasks(args.smoke)) if args.stage == "final" else N_MICE
        if not 0 <= args.task_index < count:
            raise SystemExit(f"--task-index {args.task_index} is outside 0..{count - 1}, the tasks of '{args.stage}'"
                             f"{' with --smoke' if args.smoke else ''} (see 'mantpy-liver tasks').")
    if args.workers > 1:
        if args.stage not in PARALLEL_STAGES:
            raise SystemExit("--workers applies to the 'search', 'tune' and 'final' stages; run them separately")
        return _spawn_workers(args, argv)

    from .core import Context, log
    from .final import run_task, run_tune
    from .freeze import freeze
    from .report import write_report
    from .search import run_search

    stages = {"reproduce": ["prepare", "final", "report"],
              "all": ["prepare", "search", "freeze", "tune", "final", "report"]}.get(args.stage, [args.stage])
    selection = args.selection or ("search" if args.stage == "all" else "paper")
    if args.partition == "paper" and (selection != "paper" or args.smoke):
        raise SystemExit("--partition paper needs --selection paper and cannot be combined with --smoke.")
    ctx = Context(args.results, device=args.device, cache_dir=args.cache_dir, threads=args.threads, smoke=args.smoke)
    part, parts = (int(v) for v in args.worker_slice.split("/")) if args.worker_slice else (0, 1)

    for stage in stages:
        if stage == "prepare":
            log(f"prepared {ctx.prepared.summary()}")
        elif stage == "search":
            with exit_on_sigterm():
                finished = run_search(ctx, args.n_encoders, worker_index=part if args.worker_slice else args.worker_index,
                                      time_limit_hours=args.time_limit_hours)
            if not finished:
                return UNFINISHED
        elif stage == "freeze":
            freeze(ctx, args.n_encoders)
        elif stage == "tune":
            folds = list(range(N_MICE)) if args.task_index is None else [args.task_index]
            for fold in folds[part::parts]:
                run_tune(ctx, fold, selection, args.partition, args.n_trials)
        elif stage == "final":
            tasks = final_tasks(args.smoke)
            if args.task_index is not None:
                tasks = [tasks[args.task_index]]
            for task in tasks[part::parts]:
                run_task(ctx, task, selection, args.partition)
        elif stage == "report":
            if write_report(ctx, selection, args.partition):
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
