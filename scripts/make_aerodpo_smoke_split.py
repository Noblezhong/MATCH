#!/usr/bin/env python3
"""Write a three-scene AeroDPO smoke split from the canonical seen split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SMOKE_MAPS = ("Carla_Town01", "NYCEnvironmentMegapa", "TropicalIsland")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(
            "/HDD2/TravelUAV_dataset/TravelUAV_data/data/uav_dataset/seen_valset.json"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = json.loads(args.source.read_text(encoding="utf-8"))
    selected = {}
    for row in rows:
        map_name = str(row["json"]).split("/", 1)[0]
        if map_name in SMOKE_MAPS and map_name not in selected:
            selected[map_name] = row
        if len(selected) == len(SMOKE_MAPS):
            break
    missing = [name for name in SMOKE_MAPS if name not in selected]
    if missing:
        raise ValueError(f"smoke scenes absent from source split: {missing}")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing split: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps([selected[name] for name in SMOKE_MAPS], indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "maps": list(SMOKE_MAPS)}))


if __name__ == "__main__":
    main()
