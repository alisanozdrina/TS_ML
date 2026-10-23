#!/usr/bin/env python3
"""Select NUR simulation files for Slurm array processing."""

from __future__ import annotations

import argparse
import glob
import os
import random
from pathlib import Path
from typing import Iterable, List

import yaml


def parse_energies(values: Iterable[str] | None) -> List[str]:
    if not values:
        return []
    energies: List[str] = []
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if not item:
                continue
            energies.append(item if item.startswith("lgE") else f"lgE{item}")
    return energies


def collect_files(sim_root: Path, energies: List[str]) -> List[Path]:
    if energies:
        files = []
        for energy in energies:
            files.extend(sorted((sim_root / energy).glob("*.nur")))
        return sorted(files)
    return sorted(sim_root.glob("lgE*/*.nur"))


def load_config_input_patterns(config_path: Path) -> List[str]:
    with config_path.open("r") as f:
        config = yaml.safe_load(f) or {}
    patterns = config.get("input_files")
    if patterns is None:
        patterns = config.get("input_glob")
    if patterns is None:
        raise ValueError(f"{config_path} does not define input_files or input_glob")
    if isinstance(patterns, str):
        patterns = [patterns]
    return [os.path.expandvars(str(pattern)) for pattern in patterns]


def collect_pattern_files(patterns: Iterable[str]) -> List[Path]:
    files: List[Path] = []
    for pattern in patterns:
        matches = glob.glob(pattern) if any(c in pattern for c in "*?[]") else [pattern]
        files.extend(Path(match).expanduser().resolve() for match in matches)
    return sorted(path for path in dict.fromkeys(files) if path.is_file() and path.suffix == ".nur")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/build_ml_sim.example.yaml"),
        help="YAML config to read input_files/input_glob from.",
    )
    parser.add_argument(
        "--sim-root",
        type=Path,
        default=None,
        help="Override config input patterns and collect lgE*/*.nur under this root.",
    )
    parser.add_argument(
        "--energy",
        action="append",
        help="Energy directory to include, e.g. 18.5, lgE18.5, or comma-separated values.",
    )
    parser.add_argument("--n-files", type=int, default=None, help="Maximum number of files to write")
    parser.add_argument("--random", action="store_true", help="Randomly sample files before truncating")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("slurm/lists/nc_cr_proxy_nur.txt"),
        help="Output list path.",
    )
    args = parser.parse_args()

    if args.sim_root is not None or args.energy:
        sim_root = args.sim_root or Path("/fs/ess/PAS2608/alisa/rno_g/deepCRsearch/data/nc_cr_proxy")
        files = collect_files(sim_root.expanduser(), parse_energies(args.energy))
    else:
        files = collect_pattern_files(load_config_input_patterns(args.config))
    if args.random:
        rng = random.Random(args.seed)
        rng.shuffle(files)
    if args.n_files is not None:
        files = files[:args.n_files]
    if not files:
        raise FileNotFoundError("no simulation .nur files matched the requested selection")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(str(path) for path in files) + "\n")
    print(f"Wrote {len(files)} simulation files to {args.output}", flush=True)


if __name__ == "__main__":
    main()
