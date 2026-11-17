#!/usr/bin/env python3
"""Rank every L2 kernel of the exp1 circuits by walk time, one table per initial state (orig, wide).

walk = time spent walking Feynman paths: the separate walk-only pass, capped by the real walk + insert time
(as in analyze.split()). It is the part of a kernel's time that compiling can remove; hash insertion and
per-kernel fixed costs stay.
L2 key: gate names, parameters rounded to 1e-6, qubits renamed a, b, c, ... in order of first use.
Kernels with more than MAX_GATES gates show only their first MAX_GATES gates.

Usage: python3 scripts/rank_l2_walk.py [out.md]   (default: tables/kernel_ranking_l2_walk.md)
"""
import json, os, sys
from collections import Counter

MAX_GATES = 20
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "tables", "kernel_ranking_l2_walk.md")


def load(*p):
    f = os.path.join(ROOT, "raw", *p)
    return [json.loads(l) for l in open(f)] if os.path.exists(f) else []


def l2(k):
    m = {}  # qubit -> label, in order of first use
    lab = lambda q: m.setdefault(q, chr(97 + len(m)) if len(m) < 26 else f"q{len(m)}")
    par = lambda ps: "[" + ",".join(str(round(p, 6) + 0.0) for p in ps) + "]" if ps else ""
    return [f"{n}{par(ps)}({','.join(lab(q) for q in qs)})" for n, ps, qs in zip(k["names"], k["params"], k["qubits"])]


def fmt(t):
    return f"{t:.2f} s" if t >= 1 else f"{t * 1e3:.1f} ms" if t >= 1e-3 else f"{t * 1e6:.0f} µs"


agg = {}  # key -> {show, len, k, circ, orig: [time, walk], wide: [time, walk]}
for c in sorted(f[:-6] for f in os.listdir(os.path.join(ROOT, "raw", "static"))):
    static = load("static", c + ".jsonl")
    keys = []
    for k in static:
        g = l2(k)
        key = " · ".join(g)
        show = key if len(g) <= MAX_GATES else " · ".join(g[:MAX_GATES]) + f" · … (+{len(g) - MAX_GATES} gates)"
        agg.setdefault(key, {"show": show, "len": k["len"], "k": k["k"], "circ": Counter()})["circ"][c] += 1
        keys.append(key)
    for v in ("orig", "wide"):
        for r in load("runtime", f"{c}.{v}.p4.jsonl"):
            assert r["gates"] == static[r["kernel"]]["gates"], (c, v, r["kernel"])
            s = agg[keys[r["kernel"]]].setdefault(v, [0.0, 0.0])
            s[0] += r["t_total"]
            s[1] += min(r["t_walk"], r["t_walk_and_insert"]) if r["t_walk"] is not None else 0.0

out = ["# L2 kernels ranked by walk time", "",
       "Columns: **time #** = rank by total time in the same run; **kernels** = kernels with this key; "
       "**len/k** = gates / distinct qubits; **walk** = time walking Feynman paths (what compilation can remove); "
       "**share of walk (cum.)** = share of the table's total walk and running sum; **time** = total time; "
       "**walk / time** = walk as a share of the key's own time. orig and wide are separate runs, never added."]
for v, name in (("orig", "|0...0>"), ("wide", "65536 random basis states")):
    rows = [key for key in agg if v in agg[key]]
    time_rank = {key: i for i, key in enumerate(sorted(rows, key=lambda key: -agg[key][v][0]), 1)}
    walk_total = sum(agg[key][v][1] for key in rows)
    time_total = sum(agg[key][v][0] for key in rows)
    out += [f"\n## {v} ({name}, 4 threads): {len(rows)} keys, walk {fmt(walk_total)} of total {fmt(time_total)}\n",
            "| # | time # | kernels | circuits | len/k | walk | share of walk (cum.) | time | walk / time | gates |",
            "|---" * 10 + "|"]
    cum = 0.0
    for i, key in enumerate(sorted(rows, key=lambda key: -agg[key][v][1]), 1):
        a = agg[key]
        t, w = a[v]
        cum += w
        circ = ", ".join(f"{c}:{n}" for c, n in a["circ"].items())
        out.append(f"| {i} | {time_rank[key]} | {sum(a['circ'].values())} | {circ} | {a['len']}/{a['k']} | {fmt(w)} | "
                   f"{100 * w / walk_total:.1f}% ({100 * cum / walk_total:.1f}%) | {fmt(t)} | {100 * w / t:.0f}% | `{a['show']}` |")
    out.append(f"\n{len(agg) - len(rows)} keys have no {v} run.")

os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
open(OUT, "w").write("\n".join(out) + "\n")
print(f"{len(agg)} L2 keys over {sum(sum(a['circ'].values()) for a in agg.values())} kernels -> {OUT}")
