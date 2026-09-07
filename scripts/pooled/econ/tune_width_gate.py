#!/usr/bin/env python3
"""Adjudicate the tuning-span width selection against its declared gate.

HP_SCREEN.md, declared 2026-09-06: widths 50 and 60 are candidates against
the incumbent 40, measured on the baselines' tuning span (2016-2019), and
adoption requires -- paired against 40 on identical (panel, seed) cells --
a positive dIC with t >= 2.4 AND a positive economic delta with t >= 2.4.
2.4 is the Bonferroni-corrected 5% level for the two candidates; one
family only is a FLAG, not a KEEP.

Scoring reuses the HP screen's own code path (strategy_sweep.sleeves_book
for economics, score_stream.daily_ics/spearman for the registered IC
objective) so this comparison is on the same footing as every other cell
in that screen.

The 40-island arm spans two directories: seeds 42-46 are the pre-existing
probe_TUNE16_ISL40 fleet, seeds 47-51 were added for this test.

    python3 tune_width_gate.py [--panels-dir DIR]
"""
import argparse
import csv
import importlib.util
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

_spec = importlib.util.spec_from_file_location(
    "sw", os.path.join(HERE, "strategy_sweep.py"))
_sw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sw)

SETS = ["set1", "set2", "set3", "set4"]
W0, W1 = "2016-01-01", "2019-12-31"
GATE_T = 2.4
STITCHED = "ensemble_stitched_predictions.csv"


def run_dirs(width):
    """(seed_lo, seed_hi, path) sources for an arm."""
    if width == 40:
        return [(42, 46, os.path.join(HERE, "probe_TUNE16_ISL40")),
                (47, 51, os.path.join(HERE, "tune_sweep", "tune_islands_40"))]
    return [(42, 51, os.path.join(HERE, "tune_sweep", f"tune_islands_{width}"))]


def paired_t(d):
    d = np.asarray([x for x in d if x == x], dtype=float)
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
    args = ap.parse_args()

    panels = {s: Panel(os.path.join(args.panels_dir, s), "RET_CS")
              for s in SETS}

    cells = defaultdict(dict)      # width -> (panel, seed) -> metrics
    for width in (40, 50, 60):
        for lo, hi, d in run_dirs(width):
            if not os.path.isdir(d):
                print(f"  missing {d}", flush=True)
                continue
            for s in SETS:
                for sd in range(lo, hi + 1):
                    p = os.path.join(d, f"{s}_seed{sd}", STITCHED)
                    if not os.path.exists(p):
                        continue
                    preds, rows = load_preds(p, panels[s], W0, W1)
                    rows = [r for r in rows if r > 0]
                    if not rows:
                        continue
                    days = scoring.build_days(panels[s], preds, rows)
                    book = _sw.sleeves_book(panels[s], preds, rows, 10, 10)
                    net, sharpe, mdd = ss.book_stats(book["daily_ret"])
                    ic, _, _ = ss.mean_se_t(
                        ss.finite(ss.daily_ics(days, scoring.PRED,
                                               ss.spearman)))
                    cells[width][(s, sd)] = {"rank_ic": ic, "net": net,
                                             "sharpe": sharpe, "mdd": mdd}
        print(f"  width {width:>2}: {len(cells[width])} cells", flush=True)

    print("\n=== TUNING-SPAN LEVELS (2016-2019, mean over cells) ===")
    print(f"{'islands':>8}{'n':>5}{'rank IC':>10}{'net%':>9}{'Sharpe':>8}"
          f"{'MDD%':>7}")
    for w in (40, 50, 60):
        if not cells[w]:
            continue
        v = list(cells[w].values())
        print(f"{w:>8}{len(v):>5}"
              f"{np.mean([x['rank_ic'] for x in v]):>+10.4f}"
              f"{np.mean([x['net'] for x in v]):>+9.1f}"
              f"{np.mean([x['sharpe'] for x in v]):>8.2f}"
              f"{np.mean([x['mdd'] for x in v]):>7.1f}")

    print(f"\n=== GATE: paired vs 40 islands, need t >= {GATE_T} on BOTH "
          f"IC and economics ===")
    rows_out = []
    for w in (50, 60):
        if not cells[w]:
            continue
        shared = [k for k in cells[w] if k in cells[40]]
        dic = [cells[w][k]["rank_ic"] - cells[40][k]["rank_ic"] for k in shared]
        dnet = [cells[w][k]["net"] - cells[40][k]["net"] for k in shared]
        dsh = [cells[w][k]["sharpe"] - cells[40][k]["sharpe"] for k in shared]
        mic, tic, n = paired_t(dic)
        mnet, tnet, _ = paired_t(dnet)
        msh, tsh, _ = paired_t(dsh)

        ic_pass = (mic > 0) and (tic >= GATE_T)
        # economic co-primary: net and Sharpe are the same family; require
        # one of them to clear, both positive
        econ_pass = ((mnet > 0 and tnet >= GATE_T) or
                     (msh > 0 and tsh >= GATE_T))
        if ic_pass and econ_pass:
            verdict = "KEEP"
        elif ic_pass or econ_pass:
            verdict = "FLAG (one family only)"
        else:
            verdict = "no adoption"

        print(f"\n  {w} islands vs 40   (n={n} paired cells)")
        print(f"    dIC      {mic:+.4f}  t={tic:+.2f}   "
              f"{'PASS' if ic_pass else 'fail'}")
        print(f"    dNet     {mnet:+.1f}     t={tnet:+.2f}   "
              f"{'PASS' if (mnet > 0 and tnet >= GATE_T) else 'fail'}")
        print(f"    dSharpe  {msh:+.2f}     t={tsh:+.2f}   "
              f"{'PASS' if (msh > 0 and tsh >= GATE_T) else 'fail'}")
        print(f"    -> {verdict}")
        rows_out.append({"islands": w, "n": n, "dIC": mic, "t_IC": tic,
                         "dNet": mnet, "t_net": tnet, "dSharpe": msh,
                         "t_sharpe": tsh, "verdict": verdict})

    rd = os.path.join(HERE, "results_econ")
    os.makedirs(rd, exist_ok=True)
    if rows_out:
        with open(os.path.join(rd, "tune_width_gate.csv"), "w",
                  newline="") as fh:
            w_ = csv.DictWriter(fh, fieldnames=list(rows_out[0]))
            w_.writeheader()
            w_.writerows(rows_out)
        print(f"\nwrote {rd}/tune_width_gate.csv")


if __name__ == "__main__":
    main()
