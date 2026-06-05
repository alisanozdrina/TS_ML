#!/usr/bin/env python3
"""Build a CSV event manifest for ML training from HDF5 shards."""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
from pathlib import Path
from typing import Iterable, List

import h5py
import numpy as np


SPLIT_NAMES = ("train", "val", "test")


def expand_inputs(patterns: Iterable[str]) -> List[Path]:
    paths: List[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern)) if any(c in pattern for c in "*?[]") else [pattern]
        matches = [Path(match) for match in matches]
        paths.extend(path for path in matches if path.is_file())
    unique = sorted(dict.fromkeys(path.resolve() for path in paths))
    if not unique:
        raise FileNotFoundError("no input HDF5 files matched")
    return unique


def stable_fraction(value: str, seed: int) -> float:
    digest = hashlib.sha1(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def choose_split(group_value: str, train_fraction: float, val_fraction: float, seed: int) -> str:
    frac = stable_fraction(group_value, seed)
    if frac < train_fraction:
        return "train"
    if frac < train_fraction + val_fraction:
        return "val"
    return "test"


def scalar_attr(h5: h5py.File, name: str, default):
    value = h5.attrs.get(name, default)
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def read_vector_or_default(h5: h5py.File, name: str, n_events: int, default, dtype):
    if name in h5:
        return np.asarray(h5[name][:])
    return np.full((n_events,), default, dtype=dtype)


def split_group_value(h5: h5py.File, path: Path, event_index: int, group_by: str) -> str:
    if group_by == "file":
        return str(path)
    if group_by == "run" and "run" in h5:
        return f"run:{int(h5['run'][event_index])}"
    if group_by == "source_run" and "run" in h5 and "source_type" in h5:
        return f"source:{int(h5['source_type'][event_index])}:run:{int(h5['run'][event_index])}"
    return str(path)


def write_manifest(
    paths: List[Path],
    output: Path,
    train_fraction: float,
    val_fraction: float,
    seed: int,
    group_by: str,
    relative_to: Path | None,
    default_label: int,
    default_source_type: int,
) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    with output.open("w", newline="") as fout:
        writer = csv.DictWriter(
            fout,
            fieldnames=[
                "file_path",
                "event_index",
                "station",
                "run",
                "event_number",
                "timestamp",
                "trigger_type",
                "label",
                "source_type",
                "split",
                "weight",
            ],
        )
        writer.writeheader()
        for path in paths:
            with h5py.File(path, "r") as h5:
                n_events = int(h5["waveforms"].shape[0])
                station = int(scalar_attr(h5, "station", -1))
                labels = read_vector_or_default(
                    h5, "label", n_events, scalar_attr(h5, "label", default_label), "i1"
                )
                source_types = read_vector_or_default(
                    h5,
                    "source_type",
                    n_events,
                    scalar_attr(h5, "source_type", default_source_type),
                    "i1",
                )
                weights = read_vector_or_default(h5, "weight", n_events, 1.0, "f4")
                runs = read_vector_or_default(h5, "run", n_events, scalar_attr(h5, "run", -1), "i4")
                event_numbers = read_vector_or_default(h5, "event_number", n_events, -1, "i4")
                timestamps = read_vector_or_default(h5, "timestamp", n_events, np.nan, "f8")
                trigger_types = read_vector_or_default(h5, "trigger_type", n_events, -1, "i1")
                file_path = path
                if relative_to is not None:
                    try:
                        file_path = path.relative_to(relative_to)
                    except ValueError:
                        pass

                for event_index in range(n_events):
                    split = choose_split(
                        split_group_value(h5, path, event_index, group_by),
                        train_fraction,
                        val_fraction,
                        seed,
                    )
                    writer.writerow(
                        {
                            "file_path": str(file_path),
                            "event_index": event_index,
                            "station": station,
                            "run": int(runs[event_index]),
                            "event_number": int(event_numbers[event_index]),
                            "timestamp": float(timestamps[event_index]),
                            "trigger_type": int(trigger_types[event_index]),
                            "label": int(labels[event_index]),
                            "source_type": int(source_types[event_index]),
                            "split": split,
                            "weight": float(weights[event_index]),
                        }
                    )
                    rows_written += 1
    return rows_written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="Input HDF5 files or glob patterns")
    parser.add_argument("--output", required=True, type=Path, help="Output CSV manifest")
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument(
        "--group-by",
        choices=("run", "file", "source_run"),
        default="source_run",
        help="Unit kept together when assigning splits.",
    )
    parser.add_argument(
        "--relative-to",
        type=Path,
        default=None,
        help="Store file paths relative to this directory when possible.",
    )
    parser.add_argument("--default-label", type=int, default=0)
    parser.add_argument("--default-source-type", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.train_fraction <= 0 or args.val_fraction < 0:
        raise ValueError("split fractions must be non-negative and train must be positive")
    if args.train_fraction + args.val_fraction >= 1:
        raise ValueError("train_fraction + val_fraction must be less than 1")
    paths = expand_inputs(args.inputs)
    relative_to = args.relative_to.resolve() if args.relative_to is not None else None
    rows = write_manifest(
        paths,
        args.output,
        args.train_fraction,
        args.val_fraction,
        args.seed,
        args.group_by,
        relative_to,
        args.default_label,
        args.default_source_type,
    )
    print(f"Wrote {rows} events from {len(paths)} files to {args.output}", flush=True)


if __name__ == "__main__":
    main()
