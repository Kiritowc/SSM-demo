#!/usr/bin/env python3
"""Concatenate YOLO training manifest text files (one ``./dataset/...`` path per line)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List


def merge_files(dest: Path, sources: List[Path]) -> int:
    lines: list[str] = []
    for src in sources:
        raw = Path(src).read_text(encoding="utf-8").splitlines()
        for line in raw:
            line = line.strip()
            if line:
                lines.append(line)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge train.txt / val.txt style manifests.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("inputs", nargs="+", type=Path)
    args = parser.parse_args()
    n = merge_files(args.output, list(args.inputs))
    print("wrote %s (%d paths)" % (args.output.resolve(), n))


if __name__ == "__main__":
    main()
