"""Argparse helpers for Python 3.8 (BooleanOptionalAction added in 3.9)."""

from __future__ import annotations

import argparse


def ensure_boolean_optional_action() -> None:
    """Register argparse.BooleanOptionalAction when running on Python 3.8."""
    if hasattr(argparse, "BooleanOptionalAction"):
        return

    class BooleanOptionalAction(argparse.Action):
        def __init__(self, option_strings, dest, default=None, **kwargs):
            if default is None:
                default = False
            opts: list[str] = []
            for opt in option_strings:
                if opt.startswith("--no-"):
                    continue
                opts.append(opt)
                opts.append(f"--no-{opt[2:]}")
            super().__init__(opts, dest, nargs=0, default=default, **kwargs)

        def __call__(self, parser, namespace, values, option_string=None):
            if option_string is not None and option_string.startswith("--no-"):
                setattr(namespace, self.dest, False)
            else:
                setattr(namespace, self.dest, True)

    argparse.BooleanOptionalAction = BooleanOptionalAction
