"""
Group-level summary statistics for the manuscript's CADC tables, computed
from the per-drive and Mondrian-sweep artifacts rather than transcribed by
hand -- the earlier draft's headline range (7.5x) and MDE (0.13) were both
hand-asserted and neither survived review. This script is the one place
those summary numbers are computed; the manuscript tables cite its output.

Usage:
    python summarize_cadc_tables.py \
        --per-drive ../manuscript/per_drive_miscoverage_cadc.json \
        --out ../manuscript/table_summary_cadc.json
"""
import argparse
import json

import numpy as np


def group_stats(drives: dict, role: str) -> dict:
    values = np.array([d["mean_miscoverage"] for d in drives.values() if d["role"] == role])
    return {
        "role": role,
        "n_drives": int(len(values)),
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)) if len(values) > 1 else float("nan"),
        "min": float(values.min()),
        "max": float(values.max()),
        "ratio_max_over_min": float(values.max() / values.min()),
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--per-drive", required=True)
    p.add_argument("--out", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.per_drive) as f:
        data = json.load(f)
    drives = data["drives"]

    calibration = group_stats(drives, "calibration")
    nominal = group_stats(drives, "nominal")
    degraded = group_stats(drives, "degraded")

    all_bare_values = np.array([
        d["mean_miscoverage"] for d in drives.values() if d["role"] in ("calibration", "nominal")
    ])
    all_bare = {
        "n_drives": int(len(all_bare_values)),
        "min": float(all_bare_values.min()),
        "max": float(all_bare_values.max()),
        "ratio_max_over_min": float(all_bare_values.max() / all_bare_values.min()),
    }

    for label, g in (("calibration", calibration), ("nominal", nominal), ("degraded", degraded)):
        print(f"{label:>12}: n={g['n_drives']}, mean={g['mean']:.4f} (sd={g['sd']:.4f}), "
              f"range=[{g['min']:.4f}, {g['max']:.4f}], ratio={g['ratio_max_over_min']:.2f}x")
    print(f"{'all bare':>12}: n={all_bare['n_drives']}, "
          f"range=[{all_bare['min']:.4f}, {all_bare['max']:.4f}], "
          f"ratio={all_bare['ratio_max_over_min']:.2f}x")

    diff = degraded["mean"] - nominal["mean"]
    print(f"\ndegraded_mean - nominal_mean = {diff:+.4f}")
    print(f"nominal sd = {nominal['sd']:.4f}")

    out = {
        "note": "Group summary statistics computed from per_drive_miscoverage_cadc.json "
                "(the drive-disjoint, correctly-split checkpoint).",
        "source_json": args.per_drive,
        "calibration": calibration,
        "nominal": nominal,
        "degraded": degraded,
        "all_bare_drives": all_bare,
        "degraded_minus_nominal_mean": diff,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
