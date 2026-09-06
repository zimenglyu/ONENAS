#!/usr/bin/env python3
"""Cost sensitivity + paired significance on 2022-2024, for the results-only draft.

The working draft (paper/results.tex) reports the 2022-2024 window at the
ten seeds ONE-NAS was run at.  These two exhibits are computed on exactly
that window and that seed set, so they drop into the draft without changing
any number already in it.

Run:  python3 draft_additions.py
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
FROM, TO = "2022-01-01", "2024-12-31"
MULTS = [1.0, 2.0, 3.0, 5.0]


def isl8(sd):
    return "onenas_c7e" if sd <= 44 else ("probe_s4547" if sd <= 47
                                          else "probe_s4851")


ARMS = [
    ("ONE-NAS (40 isl.)",
     lambda s, sd: f"probe_ISL40/{s}_seed{sd}/ensemble_stitched_predictions.csv"),
    ("ONE-NAS (8 isl.)",
     lambda s, sd: f"{isl8(sd)}/{s}_seed{sd}/ensemble_stitched_predictions.csv"),
    ("Online LSTM",
     lambda s, sd: f"results_econ/lstm/{s}_core7_seed{sd}/predictions.csv"),
    ("Online GRU",
     lambda s, sd: f"results_econ/gru/{s}_core7_seed{sd}/predictions.csv"),
    ("Periodic LSTM (monthly)",
     lambda s, sd: f"results_econ/periodic_lstm_monthly/{s}_core7_seed{sd}/predictions.csv"),
    ("Online AR",
     lambda s, sd: f"results_econ/ar/{s}_core7_seed{sd}/predictions.csv"),
]


def paired_t(d):
    d = np.asarray([x for x in d if x == x], dtype=float)
    if len(d) < 2:
        return float("nan"), float("nan"), len(d)
    sd = d.std(ddof=1)
    if sd == 0:
        return float(d.mean()), float("inf"), len(d)
    return float(d.mean()), float(d.mean() / (sd / math.sqrt(len(d)))), len(d)


def main():
    PAN = {s: Panel(os.path.join(PANELS, s), "RET_CS") for s in SETS}
    streams = {}
    for label, fn in ARMS:
        for s in SETS:
            for sd in SEEDS:
                p = os.path.join(HERE, fn(s, sd))
                if not os.path.exists(p):
                    continue
                preds, rows = load_preds(p, PAN[s], FROM, TO)
                rows = [r for r in rows if r > 0]
                if rows:
                    streams[(label, s, sd)] = scoring.build_days(
                        PAN[s], preds, rows)
        print(f"  loaded {label}", flush=True)

    # Each arm is booked on its own calendar, matching how the draft's
    # existing window table was produced.  Intersecting calendars across
    # arms shifts the baseline cells by a few tenths, which would put two
    # different values for the same quantity in one document.
    net = defaultdict(dict)          # (arm, mult) -> {(panel, seed): net}
    for (a, s, sd), days in streams.items():
        for m in MULTS:
            b = ss.run_book(days, 2, PAN[s].prc, PAN[s].tc, TOP_K,
                            book="sleeves", hold_days=HOLD, cost_mult=m)
            net[(a, m)][(s, sd)] = 100.0 * float(np.sum(b["daily_ret"]))

    arms = [a for a, _ in ARMS]
    ref = arms[0]

    print("\n=== PAIRED SIGNIFICANCE vs ONE-NAS (40 isl.), 2022-2024 ===")
    print(f"{'Comparison':32s}{'dNet':>9s}{'t':>8s}   n")
    for a in arms[1:]:
        d = [net[(ref, 1.0)][k] - net[(a, 1.0)][k]
             for k in net[(ref, 1.0)] if k in net[(a, 1.0)]]
        m, t, n = paired_t(d)
        print(f"{'vs ' + a:32s}{m:>+9.1f}{t:>8.2f}   {n}")

    print("\n=== COST SENSITIVITY, net % 2022-2024 ===")
    print(f"{'Arm':26s}" + "".join(f"{('x%g' % m):>9s}" for m in MULTS)
          + f"{'B/E':>8s}")
    for a in arms:
        cells = [float(np.mean(list(net[(a, m)].values()))) for m in MULTS]
        n1, n2 = cells[0], cells[1]
        be = 1.0 + n1 / (n1 - n2) if (n1 - n2) > 0 else float("nan")
        line = f"{a:26s}" + "".join(f"{c:>+9.1f}" for c in cells)
        print(line + (f"{be:>8.1f}" if be == be else f"{'--':>8s}"))


if __name__ == "__main__":
    main()
