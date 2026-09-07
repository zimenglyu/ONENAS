#!/usr/bin/env python3
"""Recompute the paper's exhibits with 20- and 60-island arms in place of 8.

Author request: every table and plot reports 20 and 60 islands instead of
8. This covers the three exhibits that need only the ensemble streams --
the window table, the paired-significance table and the cost table. The
single-vs-ensemble table additionally needs global-best predictions and is
handled separately once those stitch.

Everything is on the draft's convention: 2022-2024, seeds 42-51, each arm
booked on its own calendar (matching how the existing window table was
produced), so the 1x cost column reproduces the window table exactly.

    python3 swap8_to_2060.py
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
YEARS = ["2022", "2023", "2024"]
MULTS = [1.0, 2.0, 3.0, 5.0]
STITCHED = "ensemble_stitched_predictions.csv"

ARMS = [
    ("ONE-NAS (40 isl.)",
     lambda s, sd: f"probe_ISL40/{s}_seed{sd}/{STITCHED}"),
    ("ONE-NAS (20 isl.)",
     lambda s, sd: f"probe_ISL20/{s}_seed{sd}/{STITCHED}"),
    ("ONE-NAS (60 isl.)",
     lambda s, sd: f"islands_sweep/islands_60/{s}_seed{sd}/{STITCHED}"),
    ("Online LSTM",
     lambda s, sd: f"results_econ/lstm/{s}_core7_seed{sd}/predictions.csv"),
    ("Online GRU",
     lambda s, sd: f"results_econ/gru/{s}_core7_seed{sd}/predictions.csv"),
    ("Periodic LSTM (monthly)",
     lambda s, sd: f"results_econ/periodic_lstm_monthly/{s}_core7_seed{sd}/predictions.csv"),
    ("Online AR",
     lambda s, sd: f"results_econ/ar/{s}_core7_seed{sd}/predictions.csv"),
]


def sharpe(v):
    v = np.asarray(v, dtype=float)
    sd = v.std(ddof=1)
    return float(v.mean() / sd * math.sqrt(252)) if sd > 0 else float("nan")


def mdd(v):
    eq = np.cumsum(np.asarray(v, dtype=float))
    return float(100 * np.max(np.maximum.accumulate(eq) - eq))


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
        got = 0
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
                    got += 1
        print(f"  {label:26s} {got} runs", flush=True)

    net = defaultdict(dict)
    peryear = defaultdict(lambda: defaultdict(dict))
    risk = defaultdict(list)
    for (a, s, sd), days in streams.items():
        dates = [d[1] for d in days]
        for m in MULTS:
            b = ss.run_book(days, 2, PAN[s].prc, PAN[s].tc, TOP_K,
                            book="sleeves", hold_days=HOLD, cost_mult=m)
            dr = b["daily_ret"]
            net[(a, m)][(s, sd)] = 100.0 * float(np.sum(dr))
            if m == 1.0:
                risk[a].append((sharpe(dr), mdd(dr),
                                float(np.mean(b["traded"])) / ss.GROSS_NOTIONAL))
        # Per-year cells are SEPARATE book runs restarted at each year
        # boundary, which is the convention the existing window table
        # states ("books restarted at each window boundary"); slicing one
        # continuous book by year gives different cells that sum to the
        # same total.
        for y in YEARS:
            sub = [d for d in days if d[1][:4] == y]
            if not sub:
                continue
            by_ = ss.run_book(sub, 2, PAN[s].prc, PAN[s].tc, TOP_K,
                              book="sleeves", hold_days=HOLD)
            peryear[a][y][(s, sd)] = 100.0 * float(np.sum(by_["daily_ret"]))

    arms = [a for a, _ in ARMS if any(k[0] == a for k in streams)]

    def mse(v):
        v = np.asarray([x for x in v if x == x], dtype=float)
        if len(v) == 0:
            return float("nan"), float("nan")
        return float(v.mean()), (float(v.std(ddof=1) / math.sqrt(len(v)))
                                 if len(v) > 1 else 0.0)

    def by_seed(arm, cellmap):
        """Average the four panels within each seed, then report over seeds.

        This is the existing window table's convention: the panels are a
        diversification axis, not independent replicates, so the +-SE is
        the spread over the 10 seeds of the panel-averaged book, not the
        spread over all 40 runs. Taking SE over 40 runs instead roughly
        doubles it and would not be comparable to the rows kept from the
        current table.
        """
        per_seed = defaultdict(list)
        for (s, sd), v in cellmap.items():
            per_seed[sd].append(v)
        return [float(np.mean(v)) for v in per_seed.values() if v]

    print("\n=== WINDOW TABLE: net % by year, 2022-2024 ===")
    print(f"{'Arm':26s}" + "".join(f"{y:>14s}" for y in YEARS)
          + f"{'2022-24':>14s}   n")
    for a in arms:
        line = f"{a:26s}"
        for y in YEARS:
            m, e = mse(by_seed(a, peryear[a][y]))
            line += f"{m:>+9.1f}±{e:<4.1f}" if m == m else f"{'--':>14s}"
        m, e = mse(by_seed(a, net[(a, 1.0)]))
        print(line + f"{m:>+9.1f}±{e:<4.1f}   {len(net[(a, 1.0)])}",
              flush=True)

    print("\n=== RISK (2022-2024) ===")
    print(f"{'Arm':26s}{'Sharpe':>9s}{'MDD%':>8s}{'Turn':>8s}")
    for a in arms:
        sh = [x[0] for x in risk[a]]
        md = [x[1] for x in risk[a]]
        tu = [x[2] for x in risk[a]]
        print(f"{a:26s}{np.mean(sh):>9.2f}{np.mean(md):>8.1f}"
              f"{np.mean(tu):>8.3f}", flush=True)

    ref = "ONE-NAS (40 isl.)"
    print(f"\n=== PAIRED vs {ref}, 2022-2024 ===")
    print(f"{'Comparison':32s}{'dNet':>9s}{'t':>8s}   n")
    for a in arms:
        if a == ref:
            continue
        d = [net[(ref, 1.0)][k] - net[(a, 1.0)][k]
             for k in net[(ref, 1.0)] if k in net[(a, 1.0)]]
        m, t, n = paired_t(d)
        print(f"{'vs ' + a:32s}{m:>+9.1f}{t:>8.2f}   {n}", flush=True)

    print("\n=== COST SENSITIVITY, net % 2022-2024 ===")
    print(f"{'Arm':26s}" + "".join(f"{('x%g' % m):>9s}" for m in MULTS)
          + f"{'B/E':>8s}")
    for a in arms:
        cells = [float(np.mean(list(net[(a, m)].values()))) for m in MULTS]
        n1, n2 = cells[0], cells[1]
        be = 1.0 + n1 / (n1 - n2) if (n1 - n2) > 0 else float("nan")
        line = f"{a:26s}" + "".join(f"{c:>+9.1f}" for c in cells)
        print(line + (f"{be:>8.1f}" if be == be else f"{'--':>8s}"),
              flush=True)


if __name__ == "__main__":
    main()
