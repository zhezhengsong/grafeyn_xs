#!/usr/bin/env python3
"""Rank every L2 kernel of the exp1 circuits by occurrence count and by total time, listing all gates.

L2 key: gate names, parameters rounded to 1e-6, qubits renamed a, b, c, ... in order of first use.
Kernels with the same key are pooled over all circuits. Times come from the 4-thread runs; the orig
(|0...0>) and wide (65536 random basis states) runs are ranked separately, never added.
walk = min(walk-only pass, walk + insert), as in analyze.split().

Usage: python3 scripts/rank_l2.py [out.md]   (default: tables/kernel_ranking_l2.md)
"""
import json, os, sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "tables", "kernel_ranking_l2.md")


def load(*p):
    f = os.path.join(ROOT, "raw", *p)
    return [json.loads(l) for l in open(f)] if os.path.exists(f) else []


def l2(k):
    m = {}  # qubit -> label, in order of first use
    lab = lambda q: m.setdefault(q, chr(97 + len(m)) if len(m) < 26 else f"q{len(m)}")
    par = lambda ps: "[" + ",".join(str(round(p, 6) + 0.0) for p in ps) + "]" if ps else ""
    return " · ".join(f"{n}{par(ps)}({','.join(lab(q) for q in qs)})"
                      for n, ps, qs in zip(k["names"], k["params"], k["qubits"]))


def fmt(t):
    return f"{t:.2f} s" if t >= 1 else f"{t * 1e3:.1f} ms" if t >= 1e-3 else f"{t * 1e6:.0f} µs"


agg = {}  # key -> {len, k, n, circ, orig: [time, walk], wide: [time, walk]}
for c in sorted(f[:-6] for f in os.listdir(os.path.join(ROOT, "raw", "static"))):
    static = load("static", c + ".jsonl")
    keys = [l2(k) for k in static]
    for k, key in zip(static, keys):
        a = agg.setdefault(key, {"len": k["len"], "k": k["k"], "n": 0, "circ": Counter()})
        a["n"] += 1
        a["circ"][c] += 1
    for v in ("orig", "wide"):
        for r in load("runtime", f"{c}.{v}.p4.jsonl"):
            assert r["gates"] == static[r["kernel"]]["gates"], (c, v, r["kernel"])
            s = agg[keys[r["kernel"]]].setdefault(v, [0.0, 0.0])
            s[0] += r["t_total"]
            s[1] += min(r["t_walk"], r["t_walk_and_insert"]) if r["t_walk"] is not None else 0.0

out = ["# L2 kernel ranking, all exp1 circuits", "",
       "Columns: **kernels** = kernels with this key; **len/k** = gates / distinct qubits; **share (cum.)** = share "
       "of the table's total and running sum; **walk** = time spent walking Feynman paths (what compilation can "
       "remove), as a share of the key's own time. Keys are pooled over circuits; orig and wide are never added."]


def table(title, rows, v=None):
    val = (lambda a: a[v][0]) if v else (lambda a: a["n"])
    total = sum(val(agg[key]) for key in rows)
    out.append(f"\n## {title}: {len(rows)} keys, total {fmt(total) if v else total}\n")
    out.append("| # | kernels | circuits | len/k | " + ("time | share (cum.) | walk |" if v else "share (cum.) |") + " gates |")
    out.append("|---" * (8 if v else 6) + "|")
    cum = 0
    for i, key in enumerate(sorted(rows, key=lambda key: -val(agg[key])), 1):
        a = agg[key]
        x = val(a)
        cum += x
        share = f"{100 * x / total:.1f}% ({100 * cum / total:.1f}%)"
        circ = ", ".join(f"{c}:{n}" for c, n in a["circ"].items())
        mid = f"{fmt(x)} | {share} | {100 * a[v][1] / x:.0f}%" if v else share
        out.append(f"| {i} | {a['n']} | {circ} | {a['len']}/{a['k']} | {mid} | `{key}` |")


table("By occurrence count (all keys)", list(agg))
for v, name in (("orig", "|0...0>"), ("wide", "65536 random basis states")):
    rows = [key for key in agg if v in agg[key]]
    table(f"By total time, {v} ({name}, 4 threads)", rows, v)
    out.append(f"\n{len(agg) - len(rows)} keys have no {v} run; they are listed in the count table.")

os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
open(OUT, "w").write("\n".join(out) + "\n")
print(f"{len(agg)} L2 keys over {sum(a['n'] for a in agg.values())} kernels -> {OUT}")
