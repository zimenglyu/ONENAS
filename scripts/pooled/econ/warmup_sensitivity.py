#!/usr/bin/env python3
"""Is ONE-NAS's weak 2020 a warm-up artifact or the regime?

The registered geometry already burns in through 2019 plus 50 warm-up
generations (PRIMARY.md), and the stitched streams start 2019-01-03 while
scoring starts 2020-01-01 -- so the scored population is not cold.  The
deployment question is still fair to ask: would you trade this thing in its
first scored year?

Two measurements, both re-scored from saved predictions:

  1. PER-YEAR INCLUDING 2019, the unscored burn-in year.  A warm-up ramp
     predicts 2019 < 2020 < 2021 as the population matures.  A regime
     effect predicts 2020 is a dip in an otherwise flat profile.
  2. BY SCORING START.  Every arm scored over [start..2024] for start in
     2020/2021/2022, so any burn-in is applied to ONE-NAS and the
     baselines alike.  Reported as net, annualised net, and Sharpe,
     because spans of different length are not comparable on raw net.

Run:  python3 warmup_sensitivity.py
"""
import math
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "baselines"))
sys.path.insert(0, os.path.dirname(HERE))

import scoring                        # noqa: E402
import score_stream as ss             # noqa: E402
from panel import Panel               # noqa: E402
from rebook import load_preds         # noqa: E402

PANELS = "/Users/jonathanchang/.claude/jobs/a28206de/tmp/panels_core7"
SETS = ["set1", "set2", "set3", "set4"]
SEEDS = list(range(42, 52))
TOP_K, HOLD = 10, 10
FROM_ALL, TO = "2019-01-01", "2024-12-31"
YEARS = [str(y) for y in range(2019, 2025)]
STARTS = ["2020", "2021", "2022"]


def isl8_dir(sd):
    return "onenas_c7e" if sd <= 44 else ("probe_s4547" if sd <= 47
                                          else "probe_s4851")


ARMS = [
    ("ONE-NAS (40 isl.)",
     lambda s, sd: f"probe_ISL40/{s}_seed{sd}/ensemble_stitched_predictions.csv"),
    ("ONE-NAS (8 isl.)",
     lambda s, sd: f"{isl8_dir(sd)}/{s}_seed{sd}/ensemble_stitched_predictions.csv"),
    ("Online LSTM",
     lambda s, sd: f"results_econ/lstm/{s}_core7_seed{sd}/predictions.csv"),
    ("Online GRU",
     lambda s, sd: f"results_econ/gru/{s}_core7_seed{sd}/predictions.csv"),
    ("Periodic LSTM",
     lambda s, sd: f"results_econ/periodic_lstm_monthly/{s}_core7_seed{sd}/predictions.csv"),
]


def sharpe(v):
    v = np.asarray(v, dtype=float)
    sd = v.std(ddof=1)
    return float(v.mean() / sd * math.sqrt(252)) if sd > 0 else float("nan")


def main():
    PAN = {s: Panel(os.path.join(PANELS, s), "RET_CS") for s in SETS}

    # daily series per (arm, panel, seed) over the widest span available
    daily = {}
    for label, pathfn in ARMS:
        for s in SETS:
            for sd in SEEDS:
                p = os.path.join(HERE, pathfn(s, sd))
                if not os.path.exists(p):
                    continue
                preds, prows = load_preds(p, PAN[s], FROM_ALL, TO)
                prows = [r for r in prows if r > 0]
                if not prows:
                    continue
                days = scoring.build_days(PAN[s], preds, prows)
                book = ss.run_book(days, 2, PAN[s].prc, PAN[s].tc, TOP_K,
                                   book="sleeves", hold_days=HOLD)
                daily[(label, s, sd)] = list(zip([d[1] for d in days],
                                                 book["daily_ret"]))
        print(f"  loaded {label}", flush=True)

    arms = [a[0] for a in ARMS]

    # ---- 1. per-year, including the 2019 burn-in year
    print("\n=== PER-YEAR NET %, INCLUDING THE 2019 BURN-IN YEAR ===")
    print("(2019 is normally NOT scored; shown to expose a warm-up ramp)")
    print(f"{'Arm':22s}" + "".join(f"{y:>9s}" for y in YEARS))
    for arm in arms:
        cells = []
        for y in YEARS:
            v = []
            for (a, s, sd), ser in daily.items():
                if a != arm:
                    continue
                tot = sum(r for d, r in ser if d.startswith(y))
                if any(d.startswith(y) for d, _ in ser):
                    v.append(100 * tot)
            cells.append(float(np.mean(v)) if v else float("nan"))
        line = f"{arm:22s}" + "".join(
            (f"{c:>+9.1f}" if c == c else f"{'--':>9s}") for c in cells)
        print(line, flush=True)

    # ---- 2. by scoring start, same treatment for every arm
    print("\n=== BY SCORING START: net / annualised / Sharpe ===")
    for start in STARTS:
        print(f"\n-- scored {start}-01-01 .. 2024-12-31 --")
        print(f"{'Arm':22s}{'Net%':>9s}{'Ann.%':>8s}{'Sharpe':>8s}"
              f"{'vs ONE-NAS40':>14s}")
        ref_by_cell = {}
        for (a, s, sd), ser in daily.items():
            if a != arms[0]:
                continue
            sub = [r for d, r in ser if d >= start]
            ref_by_cell[(s, sd)] = 100 * sum(sub)
        for arm in arms:
            nets, shs, deltas = [], [], []
            for (a, s, sd), ser in daily.items():
                if a != arm:
                    continue
                sub = [r for d, r in ser if d >= start]
                if len(sub) < 30:
                    continue
                n = 100 * sum(sub)
                nets.append(n)
                shs.append(sharpe(sub))
                if (s, sd) in ref_by_cell and arm != arms[0]:
                    deltas.append(ref_by_cell[(s, sd)] - n)
            if not nets:
                continue
            yrs = (2025 - int(start))
            d = np.asarray(deltas, dtype=float)
            if len(d) > 1:
                t = d.mean() / (d.std(ddof=1) / math.sqrt(len(d)))
                dtxt = f"{d.mean():>+7.1f} ({t:>4.1f})"
            else:
                dtxt = "---"
            print(f"{arm:22s}{np.mean(nets):>+9.1f}"
                  f"{np.mean(nets) / yrs:>8.1f}{np.mean(shs):>8.2f}"
                  f"{dtxt:>14s}", flush=True)


if __name__ == "__main__":
    main()
