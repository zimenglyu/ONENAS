#!/usr/bin/env python3
"""Single-champion vs ensemble at 20 and 60 islands, for the M1 table.

Both sides come from the SAME runs -- the global-best genome and the
island-champion rank-mean ensemble are two scoring-time rules over one
population -- so the comparison is within-run and the t is paired over
(panel, seed) cells.

    python3 m1_2060.py
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
STITCHED = "ensemble_stitched_predictions.csv"

# width -> (ensemble path builder, single-champion path builder)
FLEETS = {
    20: (lambda s, sd: f"probe_ISL20/{s}_seed{sd}/{STITCHED}",
         lambda s, sd: f"globalbest/C7_ISL20/{s}_seed{sd}/{STITCHED}"),
    40: (lambda s, sd: f"probe_ISL40/{s}_seed{sd}/{STITCHED}",
         None),
    60: (lambda s, sd: f"islands_sweep/islands_60/{s}_seed{sd}/{STITCHED}",
         lambda s, sd: f"globalbest/islands_60/{s}_seed{sd}/{STITCHED}"),
}
WINDOWS = [("2022--24", "2022-01-01", "2024-12-31"),
           ("2020--24", "2020-01-01", "2024-12-31")]


def sharpe(v):
    v = np.asarray(v, dtype=float)
    sd = v.std(ddof=1)
    return float(v.mean() / sd * math.sqrt(252)) if sd > 0 else float("nan")


def paired_t(d):
    d = np.asarray([x for x in d if x == x], dtype=float)
    if len(d) < 2:
        return float("nan"), float("nan"), len(d)
    sd = d.std(ddof=1)
    if sd == 0:
        return float(d.mean()), float("inf"), len(d)
    return float(d.mean()), float(d.mean() / (sd / math.sqrt(len(d)))), len(d)


def score(path, panel, w0, w1):
    if not os.path.exists(path):
        return None
    preds, rows = load_preds(path, panel, w0, w1)
    rows = [r for r in rows if r > 0]
    if not rows:
        return None
    days = scoring.build_days(panel, preds, rows)
    b = ss.run_book(days, 2, panel.prc, panel.tc, TOP_K, book="sleeves",
                    hold_days=HOLD)
    dr = b["daily_ret"]
    return 100.0 * float(np.sum(dr)), sharpe(dr)


def main():
    PAN = {s: Panel(os.path.join(PANELS, s), "RET_CS") for s in SETS}
    print(f"{'Fleet':>6} {'Window':>9} {'Single':>16} {'Ensemble':>16}"
          f" {'dNet (t)':>17} {'dSharpe (t)':>16}")
    for width, (ens_fn, sgl_fn) in FLEETS.items():
        if sgl_fn is None:
            continue
        for wname, w0, w1 in WINDOWS:
            es, ss_ = [], []
            # Deltas are averaged over the four panels WITHIN each seed and
            # the t is taken over the 10 seeds, which is the convention the
            # published 40-island rows use (verified: it reproduces their
            # dNet +23.0 at t=18.2, where per-cell pairing gives t=10.8).
            dn_by_seed = defaultdict(list)
            dsh_by_seed = defaultdict(list)
            for s in SETS:
                for sd in SEEDS:
                    e = score(os.path.join(HERE, ens_fn(s, sd)), PAN[s], w0, w1)
                    g = score(os.path.join(HERE, sgl_fn(s, sd)), PAN[s], w0, w1)
                    if e is None or g is None:
                        continue
                    es.append(e); ss_.append(g)
                    dn_by_seed[sd].append(e[0] - g[0])
                    dsh_by_seed[sd].append(e[1] - g[1])
            dn = [float(np.mean(v)) for v in dn_by_seed.values()]
            dsh = [float(np.mean(v)) for v in dsh_by_seed.values()]
            if not es:
                print(f"{width:>6} {wname:>9}   (no data)")
                continue
            # panel-averaged per seed, matching the window table's convention
            mn_e = float(np.mean([x[0] for x in es]))
            mn_s = float(np.mean([x[0] for x in ss_]))
            sh_e = float(np.mean([x[1] for x in es]))
            sh_s = float(np.mean([x[1] for x in ss_]))
            m1, t1, n = paired_t(dn)
            m2, t2, _ = paired_t(dsh)
            print(f"{width:>6} {wname:>9} "
                  f"{mn_s:>+8.1f}/{sh_s:>5.2f} {mn_e:>+8.1f}/{sh_e:>5.2f} "
                  f"{m1:>+9.1f} ({t1:>4.1f}) {m2:>+8.2f} ({t2:>4.1f})  n={n}",
                  flush=True)


if __name__ == "__main__":
    main()
