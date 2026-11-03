#!/usr/bin/env python3
# =========================================================================================
# (C) (or copyright) 2026. Triad National Security, LLC. All rights reserved.
#
# This program was produced under U.S. Government contract 89233218CNA000001 for Los
# Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
# for the U.S. Department of Energy/National Nuclear Security Administration. All rights
# in the program are reserved by Triad National Security, LLC, and the U.S. Department
# of Energy/National Nuclear Security Administration. The Government is granted for
# itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
# license in this material to reproduce, prepare derivative works, distribute copies to
# the public, perform publicly and display publicly, and to permit others to do so.
# =========================================================================================

# This file was made in part with generative AI.

"""Analyze the output of <parthenon/comms_trace>.

Usage: comms_trace_analyze.py [prefix] [--slow-factor F] [--top N] [--csv-out PATH]

Reads <prefix>.clock.csv, <prefix>.steps.<rank>.csv and <prefix>.exchanges.<rank>.csv,
puts all time stamps on rank 0's clock, matches every traced send with its receive, and
prints
  - per-cycle step time statistics and which cycles are slow,
  - message latency statistics split by on-node/off-node and message size,
  - whether waiting is caused by late senders or by slow transfers,
  - for the slowest cycles, the chain of exchanges leading to the last rank to finish.
"""

import argparse
import glob
import sys

try:
    import numpy as np
    import pandas as pd
except ImportError:
    sys.exit("comms_trace_analyze.py requires numpy and pandas")

TCOLS = [f"t{i}" for i in range(6)]
KEY = ["src", "dst", "btype", "lane", "partition_tag", "cycle", "build"]


def load(prefix):
    clock = pd.read_csv(f"{prefix}.clock.csv")
    steps = pd.concat(
        [pd.read_csv(f) for f in sorted(glob.glob(f"{prefix}.steps.*.csv"))],
        ignore_index=True,
    )
    ex_files = sorted(glob.glob(f"{prefix}.exchanges.*.csv"))
    ex = pd.concat([pd.read_csv(f) for f in ex_files], ignore_index=True)
    if clock["dropped_records"].sum() > 0:
        print(
            "WARNING: some exchange records were dropped, increase max_records:",
            dict(zip(clock["rank"], clock["dropped_records"])),
        )
    return clock, steps, ex


def global_time(clock):
    """Return a function mapping (rank array, local time array) to rank 0 time, linearly
    interpolating the clock offset between the two synchronizations to remove drift."""
    c = clock.set_index("rank")

    def f(rank, t):
        cc = c.loc[rank]
        l0, o0 = cc["sync0_local"].values, cc["sync0_offset"].values
        l1, o1 = cc["sync1_local"].values, cc["sync1_offset"].values
        dl = np.where(np.abs(l1 - l0) > 0, l1 - l0, 1.0)
        off = o0 + (o1 - o0) * (t - l0) / dl
        return np.where(t >= 0, t + off, np.nan)

    return f


def pct(x, q):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    return np.percentile(x, q) if len(x) else np.nan


def fmt_ms(x):
    return f"{1e3 * x:9.3f}" if np.isfinite(x) else f"{'-':>9s}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix", nargs="?", default="comms_trace")
    ap.add_argument(
        "--slow-factor",
        type=float,
        default=1.25,
        help="a cycle is slow if its step time exceeds this times the median",
    )
    ap.add_argument("--top", type=int, default=5, help="number of slow cycles to dissect")
    ap.add_argument("--csv-out", default=None, help="write the matched exchanges here")
    args = ap.parse_args()

    clock, steps, ex = load(args.prefix)
    gt = global_time(clock)
    node = dict(zip(clock["rank"], clock["node"]))
    nranks = len(clock)
    print(
        f"{nranks} ranks on {clock['node'].nunique()} nodes, "
        f"{steps['cycle'].nunique()} traced cycles, {len(ex)} exchange records"
    )
    print(
        "max clock sync round trip: "
        f"{1e6 * clock[['sync0_rtt', 'sync1_rtt']].max().max():.1f} us, max drift "
        f"between syncs: {1e6 * (clock['sync1_offset'] - clock['sync0_offset']).abs().max():.1f} us"
    )

    # ---------------------------------------------------------------- steps
    for col in ["t_start", "t_step_done", "t_end"]:
        steps[col + "_g"] = gt(steps["rank"].values, steps[col].values)
    steps["step"] = steps["t_step_done"] - steps["t_start"]
    steps["spin"] = steps["task_incomplete"]
    cyc = steps.groupby("cycle").agg(
        step_max=("step", "max"),
        step_min=("step", "min"),
        ared_min=("allreduce", "min"),
        ared_max=("allreduce", "max"),
        spin_max=("spin", "max"),
        fence_max=("fence", "max"),
        start_skew=("t_start_g", lambda x: x.max() - x.min()),
    )
    # The wall time of a cycle as seen globally
    cyc["wall"] = (
        steps.groupby("cycle")["t_end_g"].max()
        - steps.groupby("cycle")["t_start_g"].min()
    )
    last_rank = steps.loc[steps.groupby("cycle")["t_step_done_g"].idxmax()]
    cyc["last_rank"] = last_rank.set_index("cycle")["rank"]
    med = cyc["step_max"].median()
    cyc["slow"] = cyc["step_max"] > args.slow_factor * med
    print(
        f"\nStep time (max over ranks): median {1e3 * med:.3f} ms, min "
        f"{1e3 * cyc['step_max'].min():.3f} ms, p90 {1e3 * pct(cyc['step_max'], 90):.3f} "
        f"ms, max {1e3 * cyc['step_max'].max():.3f} ms"
    )
    print(
        f"{cyc['slow'].sum()} of {len(cyc)} cycles are slow (> {args.slow_factor} x median)"
    )
    print(
        "Fraction of the best cycle's step time: "
        f"mean {cyc['step_max'].min() / cyc['step_max'].mean():.3f}"
    )
    print("Number of slow cycles in which each rank finished its step last:")
    if cyc["slow"].any():
        print(cyc.loc[cyc["slow"], "last_rank"].value_counts().to_string())

    per_rank = steps.groupby("rank").agg(
        step=("step", "mean"),
        spin=("spin", "mean"),
        busy=("task_complete", "mean"),
        fence=("fence", "mean"),
        allreduce=("allreduce", "mean"),
        max_recv_wait=("max_recv_wait", "mean"),
    )
    per_rank["node"] = per_rank.index.map(node)
    print("\nPer-rank means (ms):")
    print((per_rank.drop(columns="node") * 1e3).round(3).assign(node=per_rank["node"]))

    # ---------------------------------------------------------------- exchanges
    ex["node_self"] = ex["rank"].map(node)
    for c in TCOLS:
        ex[c] = gt(ex["rank"].values, ex[c].values)
    sends = ex[ex["is_send"] == 1].copy()
    recvs = ex[ex["is_send"] == 0].copy()
    sends["src"], sends["dst"] = sends["rank"], sends["peer"]
    recvs["src"], recvs["dst"] = recvs["peer"], recvs["rank"]
    # Match the n-th send of a cycle on a channel with the n-th receive
    sends = sends.sort_values(["exchange"])
    recvs = recvs.sort_values(["exchange"])
    for d in (sends, recvs):
        d["nth"] = d.groupby(KEY).cumcount()
    s = sends[KEY + ["nth", "in_step", "bytes", "nbuf"] + TCOLS].rename(
        columns={c: "s_" + c for c in TCOLS}
    )
    r = recvs[KEY + ["nth", "npolls", "max_poll_gap"] + TCOLS].rename(
        columns={c: "r_" + c for c in TCOLS}
    )
    m = s.merge(r, on=KEY + ["nth"], how="inner")
    print(
        f"\nMatched {len(m)} exchanges ({len(sends)} sends, {len(recvs)} receives traced)"
    )
    n_out = (m["in_step"] == 0).sum()
    if n_out > 0:
        print(
            f"Ignoring {n_out} exchanges outside of Step() (e.g. during remeshing or load "
            "balancing)"
        )
        m = m[m["in_step"] == 1].copy()
    m["off_node"] = m["src"].map(node) != m["dst"].map(node)
    m["kib"] = m["bytes"] / 1024.0
    # Time from data Isend to the receiver seeing the data envelope / completing
    m["envelope_lat"] = m["r_t2"] - m["s_t2"]
    m["transfer"] = m["r_t4"] - m["s_t2"]
    # Time from data envelope to completed receive (rendezvous / bandwidth part)
    m["recv_complete"] = m["r_t4"] - m["r_t2"]
    # Time the receiver spent ready before the sender posted the message
    m["recv_wait"] = m["r_t4"] - m["r_t0"]
    m["late_sender"] = (m["s_t2"] - m["r_t0"]).clip(lower=0).clip(upper=m["recv_wait"])
    m["pack"] = m["s_t2"] - m["s_t0"]
    m["unpack_launch"] = m["r_t5"] - m["r_t4"]
    m["status_vs_data"] = m["r_t3"] - m["r_t4"]

    bins = [0, 4, 16, 64, 256, 1024, 4096, 1 << 30]
    labels = ["<=4K", "<=16K", "<=64K", "<=256K", "<=1M", "<=4M", ">4M"]
    m["size_bin"] = pd.cut(m["kib"], bins=bins, labels=labels, include_lowest=True)

    def table(g):
        rows = []
        for k, d in g:
            if len(d) == 0:
                continue
            rows.append(
                [
                    *(k if isinstance(k, tuple) else (k,)),
                    len(d),
                    fmt_ms(pct(d["transfer"], 50)),
                    fmt_ms(pct(d["transfer"], 90)),
                    fmt_ms(pct(d["transfer"], 99)),
                    fmt_ms(d["transfer"].max()),
                    fmt_ms(pct(d["envelope_lat"], 50)),
                    fmt_ms(pct(d["recv_complete"], 90)),
                    fmt_ms(pct(d["late_sender"], 90)),
                    fmt_ms(pct(d["max_poll_gap"], 90)),
                ]
            )
        return rows

    print(
        "\nTransfer = data received (receiver) - data Isend posted (sender), in ms."
        "\nenv50 = median time to data envelope at receiver; rc90 = p90 envelope -> "
        "complete; late90 = p90 time receiver waited for sender to post; gap90 = p90 of "
        "the largest receiver poll gap"
    )
    hdr = (
        f"{'off':>5s} {'btype':>12s} {'size':>7s} {'n':>7s} {'xfer50':>9s} {'xfer90':>9s} "
        f"{'xfer99':>9s} {'xfermax':>9s} {'env50':>9s} {'rc90':>9s} {'late90':>9s} "
        f"{'gap90':>9s}"
    )
    print(hdr)
    for row in table(m.groupby(["off_node", "btype", "size_bin"], observed=True)):
        print(
            f"{str(row[0]):>5s} {row[1]:>12s} {str(row[2]):>7s} {row[3]:7d} "
            + " ".join(row[4:])
        )

    # Attribution of receive waits
    w = m.dropna(subset=["recv_wait"])
    tot_wait = w["recv_wait"].sum()
    if tot_wait > 0:
        print(
            f"\nOf the total receive wait ({1e3 * tot_wait:.1f} ms summed over all "
            f"receives): {100 * w['late_sender'].sum() / tot_wait:.1f}% before the "
            "sender posted, the rest in flight. Off-node share of in-flight time: "
            f"{100 * (w['recv_wait'] - w['late_sender'])[w['off_node']].sum() / max((w['recv_wait'] - w['late_sender']).sum(), 1e-30):.1f}%"
        )

    # Correlate slow cycles with off-node transfer times
    per_cyc = m.groupby(["cycle", "off_node"])["transfer"].max().unstack()
    per_cyc.columns = [f"max_xfer_{'off' if c else 'on'}" for c in per_cyc.columns]
    cyc = cyc.join(per_cyc)
    if "max_xfer_off" in cyc and cyc["max_xfer_off"].notna().sum() > 2:
        cc = np.corrcoef(cyc["step_max"], cyc["max_xfer_off"].fillna(0))[0, 1]
        print(f"Correlation of step time with max off-node transfer time: {cc:.3f}")
        print(
            "Median max off-node transfer (ms): fast cycles "
            f"{1e3 * cyc.loc[~cyc['slow'], 'max_xfer_off'].median():.3f}, slow cycles "
            f"{1e3 * cyc.loc[cyc['slow'], 'max_xfer_off'].median():.3f}"
        )

    print("\nPer-cycle summary (ms):")
    show = cyc.copy()
    for c in show.columns:
        if c not in ("last_rank", "slow"):
            show[c] = (1e3 * show[c]).round(3)
    with pd.option_context("display.max_rows", 200, "display.width", 200):
        print(show.to_string(max_rows=200))

    # Dissect the slowest cycles
    worst = cyc.sort_values("step_max", ascending=False).head(args.top)
    for c, row in worst.iterrows():
        lr = int(row["last_rank"])
        print(
            f"\n--- cycle {c}: step_max {1e3 * row['step_max']:.3f} ms, last rank {lr} "
            f"(node {node[lr]}) ---"
        )
        mc = m[(m["cycle"] == c) & (m["dst"] == lr)].sort_values(
            "recv_wait", ascending=False
        )
        cols = [
            "src",
            "btype",
            "partition_tag",
            "kib",
            "off_node",
            "recv_wait",
            "late_sender",
            "transfer",
            "envelope_lat",
            "recv_complete",
            "npolls",
            "max_poll_gap",
        ]
        out = mc[cols].head(8).copy()
        for k in [
            "recv_wait",
            "late_sender",
            "transfer",
            "envelope_lat",
            "recv_complete",
            "max_poll_gap",
        ]:
            out[k] = (1e3 * out[k]).round(3)
        print("Receives on the last rank with the longest waits (ms):")
        print(out.to_string(index=False))
        st = steps[steps["cycle"] == c].set_index("rank")
        t0 = st["t_start_g"].min()
        print("Per-rank step start/done relative to the earliest start (ms):")
        print(
            pd.DataFrame(
                {
                    "node": st.index.map(node),
                    "start": (1e3 * (st["t_start_g"] - t0)).round(3),
                    "done": (1e3 * (st["t_step_done_g"] - t0)).round(3),
                    "spin": (1e3 * st["spin"]).round(3),
                    "fence": (1e3 * st["fence"]).round(3),
                }
            ).to_string()
        )

    if args.csv_out:
        m.to_csv(args.csv_out, index=False)
        print(f"\nWrote matched exchanges to {args.csv_out}")


if __name__ == "__main__":
    main()
