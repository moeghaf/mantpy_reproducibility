"""``mantpy-intestine``: rerun the small-intestine collagen IV experiments from the command line.

Stages
------
prepare    download the two images, extract ECM patches, compute the label-free support
search     label-free search of the 28 settings: refit the configurations evaluated in the paper
           (default) or run a new NSGA-II search (--search-mode nsga); resumable, any number of workers
freeze     select one configuration per image from the finished search
final      fit the selected configuration: five seeds, graph controls, GIN ablation, baselines, choice of K
report     score against the manual annotation and compare with the paper
reproduce  prepare + final + report with the settings selected in the paper
all        prepare + search + freeze + final + report, using the search's own selection
tasks      print the number of final tasks (for cluster job arrays)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

from .config import IMAGES

STAGES = ["prepare", "search", "freeze", "final", "report", "reproduce", "all", "tasks"]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mantpy-intestine", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--image", nargs="+", choices=IMAGES, default=list(IMAGES), help="images to process (default: both)")
    parser.add_argument("--results", default="results/intestine", help="results directory (shared by all workers)")
    parser.add_argument("--cache-dir", default=None, help="dataset cache (default: MANTPY_CACHE or ~/.cache/mantpy)")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--workers", type=int, default=1,
                        help="search and final stages: processes to run in parallel on this machine")
    parser.add_argument("--shard", default="0/1", metavar="I/N",
                        help="search and final stages: do part I of N (cluster job arrays: --shard $SLURM_ARRAY_TASK_ID/N)")
    parser.add_argument("--selection", default=None, choices=["paper", "search"],
                        help="settings used by final and report: the paper's (default) or this directory's search result")
    parser.add_argument("--search-mode", default="paper", choices=["paper", "nsga"],
                        help="search: refit the paper's configurations (default) or run a new NSGA-II search")
    parser.add_argument("--n-trials", type=int, default=200, help="search, nsga mode: fitted configurations per image")
    parser.add_argument("--max-edges", type=float, default=2.5e6,
                        help="search, nsga mode: skip graphs with more estimated directed edges (the paper's limit)")
    parser.add_argument("--time-limit-hours", type=float, default=None, help="search: stop cleanly after this long")
    parser.add_argument("--task-index", type=int, default=None, help="final: run only this task of one image")
    parser.add_argument("--cnn-init", default="paper", choices=["paper", "once", "rebuilt"],
                        help="random start of the CNN; 'paper' matches the paper's runs seed by seed (default)")
    parser.add_argument("--smoke", action="store_true",
                        help="quick installation check: a crop of each image and tiny training budgets; "
                             "results are not the paper's")
    return parser


def _parse_shard(text: str) -> tuple[int, int]:
    try:
        part, parts = (int(value) for value in text.split("/"))
    except ValueError:
        raise SystemExit(f"--shard must look like 0/4, got {text!r}") from None
    if not 0 <= part < parts:
        raise SystemExit(f"--shard {text}: the part must be between 0 and {parts - 1}")
    return part, parts


def worker_command(argv: list[str], index: int, workers: int) -> list[str]:
    """Command line of one of ``--workers`` processes: the same command for one shard."""
    base, skip = [], False
    for argument in argv:
        if skip:
            skip = False
        elif argument in ("--workers", "--shard"):
            skip = True
        elif not argument.startswith(("--workers=", "--shard=")):
            base.append(argument)
    return [sys.executable, "-m", "mantpy_repro.intestine", *base, "--workers", "1", "--shard", f"{index}/{workers}"]


def _spawn_workers(argv: list[str], workers: int) -> int:
    processes = [subprocess.Popen(worker_command(argv, index, workers), env=os.environ.copy())
                 for index in range(workers)]
    codes = [process.wait() for process in processes]
    for index, code in enumerate(codes):
        if code != 0:
            print(f"worker {index} exited with code {code}", file=sys.stderr)
    return 0 if all(code == 0 for code in codes) else 1


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = _parser().parse_args(argv)
    shard = _parse_shard(args.shard)

    from .final import final_tasks

    if args.stage == "tasks":
        for image in args.image:
            count = len(final_tasks(image))
            print(f"{image}: {count} final tasks (--image {image} --task-index 0..{count - 1})")
        return 0
    if args.task_index is not None and len(args.image) != 1:
        raise SystemExit("--task-index needs exactly one --image")
    if args.workers > 1:
        if args.stage not in ("search", "final"):
            raise SystemExit("--workers applies to the 'search' and 'final' stages; run them separately")
        return _spawn_workers(argv, args.workers)

    from .core import Context, log
    from .final import run_task
    from .freeze import freeze
    from .report import paper_reference, write_report
    from .search import run_nsga_search, run_paper_search

    ctx = Context(args.results, device=args.device, cache_dir=args.cache_dir, cnn_init=args.cnn_init,
                  smoke=args.smoke)
    stages = {"reproduce": ["prepare", "final", "report"],
              "all": ["prepare", "search", "freeze", "final", "report"]}.get(args.stage, [args.stage])
    selection = args.selection or ("search" if args.stage == "all" else "paper")

    for stage in stages:
        if stage == "prepare":
            for image in args.image:
                summary = ctx.prepared(image).summary()
                log(f"prepared {summary}")
                expected = paper_reference()["images"][image]["n_patches"]
                if not args.smoke and summary["n_nodes"][32] != expected:
                    raise SystemExit(f"{image}: {summary['n_nodes'][32]} patches of 32 px, but the paper has "
                                     f"{expected}. Check the Mantpy version and the dataset.")
        elif stage == "search":
            finished = True
            for image in args.image:
                if args.search_mode == "paper":
                    finished &= run_paper_search(ctx, image, shard=shard, time_limit_hours=args.time_limit_hours,
                                                 limit=args.n_trials if args.smoke else None)
                else:
                    finished &= run_nsga_search(ctx, image, args.n_trials, max_edges=args.max_edges,
                                                worker=shard[0], time_limit_hours=args.time_limit_hours)
            if not finished:
                log("search not finished; rerun the same command to continue")
                return 3
        elif stage == "freeze":
            for image in args.image:
                freeze(ctx, image)
        elif stage == "final":
            for image in args.image:
                tasks = final_tasks(image)
                tasks = [tasks[args.task_index]] if args.task_index is not None else tasks[shard[0]::shard[1]]
                for task in tasks:
                    run_task(ctx, task, selection)
        elif stage == "report":
            status = write_report(ctx, args.image, selection)
            if status:
                return status
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
