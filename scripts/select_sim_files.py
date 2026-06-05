#!/usr/bin/env python3
"""Select NUR simulation files for Slurm array processing."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Iterable, List


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sim-root",
        type=Path,
        default=Path("/fs/ess/PAS2608/alisa/rno_g/deepCRsearch/data/nc_cr_proxy"),
    )
    parser.add_argument(
        "--energy",
        action="append",
        help="Energy directory to include, e.g. 18.5, lgE18.5, or comma-separated values.",
    )
    parser.add_argument("--n-files", type=int, default=None, help="Maximum number of files to write")
    parser.add_argument("--random", action="store_true", help="Randomly sample files before truncating")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    files = collect_files(args.sim_root, parse_energies(args.energy))
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
