#!/usr/bin/env python3
"""Do the baseline rows change when every available seed is used?

The paired tests against ONE-NAS must use matched (panel, seed) cells, and
ONE-NAS exists only at seeds 42-51, so the pairing is necessarily limited to
those ten.  The baselines' own MEANS are a different question: online LSTM
was run at 48 seeds and GRU at 40, so reporting them from the ten matched
seeds throws away most of the evidence about where the baseline actually
sits, and a subsample that happens to run cold or hot would misstate it.

This scores every available run of each baseline and compares:
  matched  seeds 42-51 only, as used for the paired tests
  full     every seed present on disk

If the two agree the paired-seed subsample is representative and the table
rows stand.  If they diverge the table should quote the full-seed mean and
keep the pairing separate.

Run:  python3 baseline_full_seeds.py
"""
import math
import os
import re
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
MATCHED = set(range(42, 52))
TOP_K, HOLD = 10, 10
FROM, TO = "2020-01-01", "2024-12-31"

ARMS = ["lstm", "gru", "periodic_lstm_monthly", "ar"]
YEARS = [str(y) for y in range(2020, 2025)]


def sharpe(v):
    v = np.asarray(v, dtype=float)
    sd = v.std(ddof=1)
    return float(v.mean() / sd * math.sqrt(252)) if sd > 0 else float("nan")


def mdd(v):
    eq = np.cumsum(np.asarray(v, dtype=float))
    return float(100 * np.max(np.maximum.accumulate(eq) - eq))


def main():
    PAN = {s: Panel(os.path.join(PANELS, s), "RET_CS") for s in SETS}
    rows = defaultdict(list)

    for arm in ARMS:
        d = os.path.join(HERE, "results_econ", arm)
        if not os.path.isdir(d):
            print(f"  missing {d}", flush=True)
            continue
        for run in sorted(os.listdir(d)):
            m = re.match(r"(set\d)_core7_seed(\d+)$", run)
            if not m:
                continue
            s, sd = m.group(1), int(m.group(2))
            p = os.path.join(d, run, "predictions.csv")
            if s not in PAN or not os.path.exists(p):
                continue
            preds, prows = load_preds(p, PAN[s], FROM, TO)
            prows = [r for r in prows if r > 0]
            if not prows:
                continue
            days = scoring.build_days(PAN[s], preds, prows)
            book = ss.run_book(days, 2, PAN[s].prc, PAN[s].tc, TOP_K,
                               book="sleeves", hold_days=HOLD)
            dr = book["daily_ret"]
            rec = {"panel": s, "seed": sd,
                   "net": 100.0 * float(np.sum(dr)),
                   "sharpe": sharpe(dr), "mdd": mdd(dr),
                   "turnover": (float(np.mean(book["traded"]))
                                / ss.GROSS_NOTIONAL)}
            per_year = defaultdict(list)
            for dt, r in zip([d[1] for d in days], dr):
                per_year[dt[:4]].append(r)
            for y in YEARS:
                rec[f"net_{y}"] = (100.0 * float(np.sum(per_year[y]))
                                   if per_year.get(y) else float("nan"))
            rows[arm].append(rec)
        print(f"  {arm}: {len(rows[arm])} runs, "
              f"{len({r['seed'] for r in rows[arm]})} seeds", flush=True)

    def agg(recs, field):
        v = np.asarray([r[field] for r in recs if r[field] == r[field]],
                       dtype=float)
        if len(v) == 0:
            return float("nan"), float("nan"), 0
        se = v.std(ddof=1) / math.sqrt(len(v)) if len(v) > 1 else 0.0
        return float(v.mean()), float(se), len(v)

    print(f"\n{'Arm':22s}{'subset':>9s}{'n':>5s}{'seeds':>7s}"
          f"{'Net%':>9s}{'±SE':>7s}{'Sharpe':>8s}{'±SE':>7s}"
          f"{'MDD%':>8s}{'Turn':>8s}")
    for arm in ARMS:
        if not rows[arm]:
            continue
        for label, recs in (
                ("matched", [r for r in rows[arm] if r["seed"] in MATCHED]),
                ("full", rows[arm])):
            if not recs:
                continue
            n_seeds = len({r["seed"] for r in recs})
            nm, nse, n = agg(recs, "net")
            sm, sse, _ = agg(recs, "sharpe")
            dm, _, _ = agg(recs, "mdd")
            tm, _, _ = agg(recs, "turnover")
            print(f"{arm:22s}{label:>9s}{n:>5d}{n_seeds:>7d}"
                  f"{nm:>+9.1f}{nse:>7.1f}{sm:>8.2f}{sse:>7.2f}"
                  f"{dm:>8.1f}{tm:>8.3f}", flush=True)

    print("\nONE-NAS exists only at seeds 42-51, so paired tests stay on the\n"
          "matched subset regardless of what the baselines' full-seed means\n"
          "turn out to be.")

    # per-year cells at full replication, for the window table
    print(f"\n=== PER-YEAR NET %, FULL SEEDS (mean +- SE) ===")
    print(f"{'Arm':22s}" + "".join(f"{y:>14s}" for y in YEARS)
          + f"{'2020-24':>14s}   n")
    for arm in ARMS:
        if not rows[arm]:
            continue
        line = f"{arm:22s}"
        for y in YEARS:
            m, se, _ = agg(rows[arm], f"net_{y}")
            line += f"{m:>+9.1f}±{se:<4.1f}" if m == m else f"{'--':>14s}"
        m, se, n = agg(rows[arm], "net")
        line += f"{m:>+9.1f}±{se:<4.1f}   {n}"
        print(line, flush=True)


if __name__ == "__main__":
    main()
