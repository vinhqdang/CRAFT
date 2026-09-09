"""
Minimum detectable effect (MDE) for the headline weather-effect estimate.

The prior manuscript draft asserted an MDE of "roughly 0.13, about twice the
point estimate" with no computation behind it and a between-session sd
(0.06) borrowed from a different experiment (the in-sample per-drive
miscoverage table, not the drive-disjoint effect estimate). Neither survived
review: the value did not reproduce from its own stated inputs under any
standard power formula, and the sd was measured on the wrong data.

This computes the MDE directly from the bootstrap standard error already
produced by `two_level_effect` (via the two-level percentile CI), so the
number is tied to the same artifact the headline effect comes from rather
than to a borrowed variance estimate.

Standard two-sided-alpha / power formula for a normally-approximated
estimator:

    MDE = (z_{1-alpha/2} + z_{power}) * SE

We approximate SE from the reported 95% percentile CI half-width under a
normal approximation (SE = half-width / 1.96); this is the same
approximation implicit in reporting a symmetric-looking "95% CI" at all, and
is stated as an approximation rather than an exact resample of the power
itself.

Usage:
    python compute_mde.py --effect-json ../manuscript/split_effect_split_ep5_fixed.json \
        --out ../manuscript/mde_cadc.json
"""
import argparse
import json

Z_975 = 1.959963985
Z_80 = 0.841621234
Z_SUM = Z_975 + Z_80  # ~2.8016, two-sided alpha=0.05, 80% power


def mde_from_ci(ci_low: float, ci_high: float) -> dict:
    half_width = (ci_high - ci_low) / 2.0
    se = half_width / Z_975
    mde = Z_SUM * se
    return {"ci_half_width": half_width, "se_normal_approx": se, "mde_80pct_power": mde}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--effect-json", required=True)
    p.add_argument("--out", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.effect_json) as f:
        data = json.load(f)

    results = {}
    for arm_label, rows in data["results"].items():
        for row in rows:
            if row["calibration_mode"] != "mondrian" or row["gap"] != 0:
                continue
            m = mde_from_ci(row["ci_low"], row["ci_high"])
            m["point_estimate"] = row["effect"]
            m["mde_over_point_estimate"] = (
                m["mde_80pct_power"] / abs(row["effect"]) if row["effect"] else float("nan")
            )
            results[arm_label] = m
            print(f"{arm_label} (mondrian, gap=0): "
                  f"point estimate {row['effect']:+.4f}, "
                  f"SE~{m['se_normal_approx']:.4f}, "
                  f"MDE(80% power)={m['mde_80pct_power']:.4f} "
                  f"({m['mde_over_point_estimate']:.2f}x the point estimate)")

    out = {
        "note": "MDE computed from the two-level bootstrap CI half-width under a "
                "normal approximation (SE = half-width / z_0.975); "
                "MDE = (z_0.975 + z_0.80) * SE.",
        "source_json": args.effect_json,
        "z_975": Z_975,
        "z_80": Z_80,
        "results": results,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
