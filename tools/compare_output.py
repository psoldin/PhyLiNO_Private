#!/usr/bin/env python3
"""Compare two LLHFit Output.json files, ignoring timing fields.

Usage: compare_output.py A.json B.json [abs_tolerance]
Exit 0 if equal (within tolerance), 1 otherwise.
"""
import json
import sys

IGNORED = {"fitTime", "fitDuration"}
TOL = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
_failed = False


def clean(obj):
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items() if k not in IGNORED}
    if isinstance(obj, list):
        return [clean(v) for v in obj]
    return obj


def report(msg):
    global _failed
    _failed = True
    print(msg)


def diff(a, b, path=""):
    if isinstance(a, bool) or isinstance(b, bool):
        if a != b:
            report(f"{path}: {a!r} vs {b!r}")
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if a != b and abs(a - b) > TOL:
            report(f"{path}: {a!r} vs {b!r} (delta {abs(a - b):.3e})")
    elif type(a) is not type(b):
        report(f"{path}: type {type(a).__name__} vs {type(b).__name__}")
    elif isinstance(a, dict):
        for k in sorted(a.keys() | b.keys()):
            if k not in a:
                report(f"{path}.{k}: only in right file")
            elif k not in b:
                report(f"{path}.{k}: only in left file")
            else:
                diff(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list):
        if len(a) != len(b):
            report(f"{path}: length {len(a)} vs {len(b)}")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                diff(x, y, f"{path}[{i}]")
    elif a != b:
        report(f"{path}: {a!r} vs {b!r}")


with open(sys.argv[1]) as f:
    left = clean(json.load(f))
with open(sys.argv[2]) as f:
    right = clean(json.load(f))

diff(left, right)
if not _failed:
    print("IDENTICAL (ignoring timing fields)" + (f" within tolerance {TOL}" if TOL else ""))
sys.exit(1 if _failed else 0)
