#!/usr/bin/env python3
"""Four paper exhibits, all from already-saved predictions (no new training).

  A. WINDOW EXTENSION.  Table 1 currently reports 2022-2024, a window
     inherited from EXAMM's evaluation span.  EXAMM is out of the paper, so
     that restriction now has no stated reason and reads as window-picking.
     The registered evaluation span is 2020-2024; this rebuilds the same
     per-year table across all five years.

  B. HEADLINE SIGNIFICANCE.  Table 1 carries +-SE but no test, while the
     paper's headline claim is ONE-NAS vs the online baselines.  Paired
     over the (panel, seed) cells the arms share, which is the same pairing
     Table 2 already uses for single-vs-ensemble.

  C. RISK COLUMNS.  MDD and turnover are computed by every scorer already
     and simply were not tabulated next to net.

  D. COST SENSITIVITY.  Every headline number is net of realistic per-name
     costs (TC/|PRC|); a reviewer will ask what happens when costs are
     worse than that.  Sweeps a multiplier over the realistic cost model
     (preserving its cross-sectional shape) and reports the break-even
     multiple where net return reaches zero.

Run:
    python3 paper_additions.py [--panels-dir DIR] [--out-dir DIR]
"""
import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "baselines"))
sys.path.insert(0, os.path.dirname(HERE))

import scoring                         # noqa: E402
import score_stream as ss              # noqa: E402
from panel import Panel                # noqa: E402
from rebook import load_preds          # noqa: E402

SETS = ["set1", "set2", "set3", "set4"]
SEEDS = tuple(range(42, 52))
HOLD = 10
TOP_K = 10
FROM, TO = "2020-01-01", "2024-12-31"
YEARS = ["2020", "2021", "2022", "2023", "2024"]
COST_MULTS = [1.0, 1.5, 2.0, 3.0, 5.0]


def isl8_dir(sd):
    return "onenas_c7e" if sd <= 44 else ("probe_s4547" if sd <= 47
                                          else "probe_s4851")


# label -> path builder, relative to HERE
ARMS = [
    ("ONE-NAS (40 isl.)",
     lambda s, sd: f"probe_ISL40/{s}_seed{sd}/ensemble_stitched_predictions.csv"),
    ("ONE-NAS (8 isl.)",
     lambda s, sd: f"{isl8_dir(sd)}/{s}_seed{sd}/ensemble_stitched_predictions.csv"),
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


def paired_t(deltas):
    """Paired t over matched cells; returns (mean, t, n)."""
    d = np.asarray([x for x in deltas if x == x], dtype=float)
    if len(d) < 2:
        return float("nan"), float("nan"), len(d)
    sd = d.std(ddof=1)
    if sd == 0:
        return float(d.mean()), float("inf"), len(d)
    return float(d.mean()), float(d.mean() / (sd / math.sqrt(len(d)))), len(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels-dir",
                    default="/Users/jonathanchang/.claude/jobs/a28206de/tmp/panels_core7")
    ap.add_argument("--out-dir", default=os.path.join(HERE, "results_econ"))
    args = ap.parse_args()

    print("loading panels ...", flush=True)
    PAN = {s: Panel(os.path.join(args.panels_dir, f"{s}_core7"), "RET_CS")
           for s in SETS}

    # ---- pass 1: load every arm's stream, find the shared calendar.
    # Arms end on different dates (ONE-NAS streams stop mid-December, the
    # baselines run to the 31st); comparing them on their own calendars
    # would give different arms different numbers of trading days.
    RAW = {}
    for label, pathfn in ARMS:
        for s in SETS:
            for sd in SEEDS:
                p = os.path.join(HERE, pathfn(s, sd))
                if not os.path.exists(p):
                    continue
                try:
                    preds, prows = load_preds(p, PAN[s], FROM, TO)
                    # row 0 has no naive column; build_days needs row-1 >= 0
                    prows = [r for r in prows if r > 0]
                    days = scoring.build_days(PAN[s], preds, prows)
                except Exception as e:                      # noqa: BLE001
                    print(f"  skip {label} {s} {sd}: {e}", flush=True)
                    continue
                if days:
                    RAW[(label, s, sd)] = days

    if not RAW:
        raise SystemExit("no prediction streams loaded -- check --panels-dir")

    by_set = defaultdict(list)
    for (label, s, sd), days in RAW.items():
        by_set[s].append({d[1] for d in days})
    shared = {s: set.intersection(*v) for s, v in by_set.items()}
    for s in SETS:
        print(f"  {s}: shared calendar {len(shared.get(s, []))} days",
              flush=True)

    arms_present = sorted({k[0] for k in RAW}, key=lambda L: [a[0] for a in ARMS].index(L))

    # ---- pass 2: score every (arm, panel, seed) at each cost multiple.
    rows = []
    for (label, s, sd), days in sorted(RAW.items()):
        days = [d for d in days if d[1] in shared[s]]
        if not days:
            continue
        for cm in COST_MULTS:
            book = ss.run_book(days, 2, PAN[s].prc, PAN[s].tc, TOP_K,
                               book="sleeves", hold_days=HOLD, cost_mult=cm)
            dr = book["daily_ret"]
            dates = [d[1] for d in days]
            rec = {"arm": label, "panel": s, "seed": sd, "cost_mult": cm,
                   "n_days": len(dr),
                   "net_pct": 100.0 * float(np.sum(dr)),
                   "sharpe": sharpe(dr), "mdd_pct": mdd(dr),
                   "turnover": float(np.mean(book["traded"])) / ss.GROSS_NOTIONAL,
                   "cost_pct": 100.0 * float(np.sum(book["cost"])) / ss.CAPITAL}
            # per-year net, for exhibit A (only needed at realistic costs)
            if cm == 1.0:
                per_year = defaultdict(list)
                for dt, r in zip(dates, dr):
                    per_year[dt[:4]].append(r)
                for y in YEARS:
                    rec[f"net_{y}"] = (100.0 * float(np.sum(per_year[y]))
                                       if per_year.get(y) else float("nan"))
                    rec[f"sharpe_{y}"] = (sharpe(per_year[y])
                                          if len(per_year.get(y, [])) > 1
                                          else float("nan"))
            rows.append(rec)
        print(f"  scored {label:26s} {s} seed{sd}", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    raw_csv = os.path.join(args.out_dir, "paper_additions_raw.csv")
    keys = sorted({k for r in rows for k in r})
    with open(raw_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {raw_csv}  ({len(rows)} rows)", flush=True)

    base = [r for r in rows if r["cost_mult"] == 1.0]

    def agg(arm, field):
        v = [r[field] for r in base if r["arm"] == arm and r[field] == r[field]]
        if not v:
            return float("nan"), float("nan"), 0
        a = np.asarray(v, dtype=float)
        se = a.std(ddof=1) / math.sqrt(len(a)) if len(a) > 1 else 0.0
        return float(a.mean()), float(se), len(a)

    out = {}

    # ---- A: per-year table, 2020-2024
    print("\n=== A. WINDOW EXTENSION: net % by year (mean +- SE) ===",
          flush=True)
    print(f"{'Arm':28s}" + "".join(f"{y:>14s}" for y in YEARS)
          + f"{'2020-24':>14s}   n", flush=True)
    tabA = {}
    for arm in arms_present:
        cells = []
        for y in YEARS:
            m, se, n = agg(arm, f"net_{y}")
            cells.append((m, se, n))
        tot, tot_se, n = agg(arm, "net_pct")
        tabA[arm] = {"years": cells, "total": (tot, tot_se, n)}
        line = f"{arm:28s}"
        for m, se, _ in cells:
            line += f"{m:>+9.1f}±{se:<4.1f}" if m == m else f"{'--':>14s}"
        line += f"{tot:>+9.1f}±{tot_se:<4.1f}" if tot == tot else f"{'--':>14s}"
        line += f"   {n}"
        print(line, flush=True)
    out["A_yearly"] = tabA

    # ---- B: paired significance vs each baseline
    print("\n=== B. HEADLINE SIGNIFICANCE: ONE-NAS (40 isl.) vs baselines ===",
          flush=True)
    print("(paired over shared (panel,seed) cells, 2020-2024, realistic costs)",
          flush=True)
    ref = "ONE-NAS (40 isl.)"
    idx = {(r["arm"], r["panel"], r["seed"]): r for r in base}
    print(f"{'Comparison':34s}{'dNet':>9s}{'t':>8s}{'dSharpe':>10s}{'t':>8s}   n",
          flush=True)
    tabB = {}
    for arm in arms_present:
        if arm == ref:
            continue
        dn, dsh = [], []
        for s in SETS:
            for sd in SEEDS:
                a = idx.get((ref, s, sd))
                b = idx.get((arm, s, sd))
                if a and b:
                    dn.append(a["net_pct"] - b["net_pct"])
                    dsh.append(a["sharpe"] - b["sharpe"])
        m1, t1, n1 = paired_t(dn)
        m2, t2, _ = paired_t(dsh)
        tabB[arm] = {"dnet": m1, "dnet_t": t1, "dsharpe": m2,
                     "dsharpe_t": t2, "n": n1}
        print(f"{'vs ' + arm:34s}{m1:>+9.1f}{t1:>8.2f}{m2:>+10.2f}{t2:>8.2f}   {n1}",
              flush=True)
    out["B_paired"] = tabB

    # ---- C: risk columns
    print("\n=== C. RISK COLUMNS (2020-2024, realistic costs) ===", flush=True)
    print(f"{'Arm':28s}{'Net%':>10s}{'Sharpe':>9s}{'MDD%':>9s}{'Turnover':>10s}   n",
          flush=True)
    tabC = {}
    for arm in arms_present:
        net, net_se, n = agg(arm, "net_pct")
        sh, sh_se, _ = agg(arm, "sharpe")
        md, md_se, _ = agg(arm, "mdd_pct")
        tu, tu_se, _ = agg(arm, "turnover")
        tabC[arm] = {"net": net, "net_se": net_se, "sharpe": sh,
                     "sharpe_se": sh_se, "mdd": md, "mdd_se": md_se,
                     "turnover": tu, "turnover_se": tu_se, "n": n}
        print(f"{arm:28s}{net:>+10.1f}{sh:>9.2f}{md:>9.1f}{tu:>10.3f}   {n}",
              flush=True)
    out["C_risk"] = tabC

    # ---- D: cost sensitivity
    print("\n=== D. COST SENSITIVITY: net % vs cost multiple ===", flush=True)
    print(f"{'Arm':28s}" + "".join(f"{('x%g' % m):>11s}" for m in COST_MULTS)
          + f"{'breakeven':>11s}", flush=True)
    tabD = {}
    for arm in arms_present:
        cells = []
        for cm in COST_MULTS:
            v = [r["net_pct"] for r in rows
                 if r["arm"] == arm and r["cost_mult"] == cm]
            cells.append(float(np.mean(v)) if v else float("nan"))
        # break-even multiple: net(x) is ~linear in the multiplier, so solve
        # net = net1 - (x-1)*cost1 = 0 using the measured cost drag.
        c1 = [r for r in rows if r["arm"] == arm and r["cost_mult"] == 1.0]
        c2 = [r for r in rows if r["arm"] == arm and r["cost_mult"] == 2.0]
        be = float("nan")
        if c1 and c2:
            n1 = float(np.mean([r["net_pct"] for r in c1]))
            n2 = float(np.mean([r["net_pct"] for r in c2]))
            drag = n1 - n2                      # cost of one extra multiple
            if drag > 0:
                be = 1.0 + n1 / drag
        tabD[arm] = {"by_mult": dict(zip(map(str, COST_MULTS), cells)),
                     "breakeven_mult": be}
        line = f"{arm:28s}" + "".join(
            (f"{c:>+11.1f}" if c == c else f"{'--':>11s}") for c in cells)
        line += f"{be:>11.2f}" if be == be else f"{'--':>11s}"
        print(line, flush=True)
    out["D_cost"] = tabD

    js = os.path.join(args.out_dir, "paper_additions.json")
    with open(js, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote {js}", flush=True)


if __name__ == "__main__":
    main()
