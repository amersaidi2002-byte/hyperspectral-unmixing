"""Command-line entry points for fitting, benchmarking and local inspection."""

import argparse
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from .algorithms import unmix
from .data import download_data
from .experiment import METHODS, read_config, run_benchmark, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser(
        "download", help="Download and verify pinned reference data."
    )
    download.add_argument("--data", type=Path, default=Path("data/raw"))
    benchmark = commands.add_parser("benchmark", help="Run the fixed comparison.")
    benchmark.add_argument("--config", type=Path, default=Path("configs/benchmark.json"))
    benchmark.add_argument("--data", type=Path, default=Path("data/raw"))
    benchmark.add_argument("--output", type=Path, default=Path("runs/benchmark"))
    report = commands.add_parser("report", help="Generate measured figures and report.")
    report.add_argument("--results", type=Path, default=Path("results"))
    report.add_argument("--data", type=Path, default=Path("data/raw"))
    report.add_argument("--output", type=Path, default=Path("docs"))
    explore = commands.add_parser("explore", help="Inspect saved fits on localhost.")
    explore.add_argument("--results", type=Path, default=Path("results"))
    explore.add_argument("--data", type=Path, default=Path("data/raw"))
    explore.add_argument("--port", type=int, default=7861)
    fit = commands.add_parser("fit", help="Unmix an unlabeled H x W x B NumPy cube.")
    fit.add_argument("--input", required=True, type=Path)
    fit.add_argument("--components", required=True, type=int)
    fit.add_argument("--method", choices=METHODS, default="simplex-nmf")
    fit.add_argument("--seed", type=int, default=0)
    fit.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "download":
            download_data(args.data)
            print(f"Verified four files in {args.data}")
        elif args.command == "benchmark":
            run_benchmark(read_config(args.config), args.data, args.output)
        elif args.command == "report":
            from .report import build_report

            build_report(args.results, args.data, args.output)
        elif args.command == "explore":
            from .explorer import serve

            serve(args.results, args.data, args.port)
        elif args.command == "fit":
            if args.output.suffix != ".npz":
                raise ValueError("Use an .npz output filename.")
            cube = np.load(args.input, allow_pickle=False)
            with threadpool_limits(limits=1):
                fitted = unmix(cube, args.components, args.method, args.seed)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                args.output,
                endmembers=fitted.endmembers,
                abundances=fitted.abundances.reshape(*cube.shape[:2], -1),
            )
            write_json(args.output.with_suffix(".json"), fitted.diagnostics)
            print(json.dumps(fitted.diagnostics, indent=2))
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
