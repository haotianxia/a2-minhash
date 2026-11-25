import csv
import itertools
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
import numpy as np

from minhash import LSH, MinHash, jaccard, shingles, similarity
from policy import TaskReuse

SURFACE, GRID, AXIS = "#fcfcfb", "#e1e0d9", "#c3c2b7"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
COLORS = {"baseline": "#a8a7a1", "task": "#2a78d6", "content": "#eb6834", "both": "#1baf7a"}
NAMES = {"baseline": "baseline", "task": "task-level reuse", "content": "content-level reuse", "both": "both"}
ORDER = ["tickets", "bugs", "docs", "citations", "hosts", "configs", "code", "logs"]
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "assignment2", "datasets")

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2, "axes.titlesize": 9.5,
    "axes.titlecolor": INK, "axes.titleweight": "regular", "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 8.5, "lines.linewidth": 1.6, "lines.markersize": 5.5,
})


def save(fig, out, name):
    os.makedirs(os.path.join(out, "plots"), exist_ok=True)
    fig.savefig(os.path.join(out, "plots", name), dpi=200, bbox_inches="tight")
    plt.close(fig)


def per_task(t, key):
    return t[key] / t["n"]


def fig_minhash(out):
    mh = MinHash()
    est, exact = [], []
    for name in ("tickets", "bugs", "docs", "hosts"):
        pol = TaskReuse(os.path.join(DATA, name), calibrate=False)
        records = [json.loads(line) for line in open(os.path.join(DATA, name, "env", "history.jsonl"))][:90]
        sets = [shingles(pol.text(r)) for r in records]
        sigs = [mh.signature(s) for s in sets]
        for i, j in itertools.combinations(range(len(records)), 2):
            exact.append(jaccard(sets[i], sets[j]))
            est.append(similarity(sigs[i], sigs[j]))
    est, exact = np.array(est), np.array(exact)
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.2, 3.6))
    a.scatter(exact, est, s=7, color=COLORS["task"], alpha=0.35, linewidths=0)
    a.plot([0, 1], [0, 1], color=MUTED, lw=1)
    a.set(xlim=(0, 1), ylim=(0, 1), xlabel="exact Jaccard of 3-shingles", ylabel="MinHash estimate (128 hashes)",
          title="Estimate vs. exact Jaccard, %d record pairs" % len(exact))
    j = np.linspace(0, 1, 201)
    rng = np.random.default_rng(1115)
    for (bands, rows), key in (((64, 2), "task"), ((32, 4), "content")):
        lsh = LSH(bands, rows)
        b.plot(j, lsh.probability(j), color=COLORS[key], label="%d bands x %d rows" % (bands, rows))
        grid = np.round(np.arange(0.05, 1.0, 0.1), 2)
        hits = [np.mean([np.any(np.all((sig_a == sig_b).reshape(bands, rows), axis=1))
                         for sig_a, sig_b in pairs_at(rng, mh, x)]) for x in grid]
        b.scatter(grid, hits, color=COLORS[key], s=22, edgecolors=SURFACE, linewidths=1.2, zorder=3)
    b.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Jaccard similarity",
          ylabel="probability of becoming a candidate", title="LSH: theory (line) and measured (dots)")
    b.legend(loc="lower right")
    fig.tight_layout()
    save(fig, out, "fig1_minhash.png")
    bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
    rows = []
    for lo, hi in bins:
        m = (exact >= lo) & (exact < hi)
        if m.any():
            err = est[m] - exact[m]
            rows.append([lo, hi, int(m.sum()), round(float(err.mean()), 4), round(float(err.std()), 4),
                         round(float(np.sqrt(exact[m] * (1 - exact[m]) / 128).mean()), 4)])
    with open(os.path.join(out, "minhash_error.csv"), "w", newline="") as handle:
        csv.writer(handle).writerows([["lo", "hi", "pairs", "bias", "sd", "theory_sd"]] + rows)


def pairs_at(rng, mh, j, n=120, size=400):
    out = []
    shared = int(round(2 * j * size / (1 + j)))
    for _ in range(n):
        base = rng.integers(0, 1 << 40, size=2 * size - shared)
        a = {str(x) for x in base[:size]}
        b = {str(x) for x in base[:shared]} | {str(x) for x in base[size:]}
        out.append((mh.signature(a), mh.signature(b)))
    return out


def fig_tradeoff(out, table):
    names = present(table)
    fig, axes = plt.subplots(2, 4, figsize=(10, 5), sharey=False)
    for ax in axes.flat[len(names):]:
        ax.set_visible(False)
    for ax, name in zip(axes.flat, names):
        base = table[(name, "baseline")]
        for config in ("baseline", "task", "content", "both"):
            t = table.get((name, config))
            if t is None:
                continue
            ax.scatter(per_task(t, "input_tokens") / 1000, 100 * t["passed"] / t["n"], s=46, color=COLORS[config],
                       edgecolors=SURFACE, linewidths=1.6, zorder=3, label=NAMES[config])
        ax.set_title(name)
        ax.set_xlim(left=0, right=max(per_task(table[(name, c)], "input_tokens") for c in COLORS
                                      if (name, c) in table) / 1000 * 1.15)
        lo = min(100 * table[(name, c)]["passed"] / table[(name, c)]["n"] for c in COLORS if (name, c) in table)
        ax.set_ylim(max(0, lo - 12), 102)
        ax.axhline(100 * base["passed"] / base["n"], color=GRID, lw=1, zorder=1)
    for ax in axes[1]:
        ax.set_xlabel("input tokens per task (K)")
    for ax in axes[:, 0]:
        ax.set_ylabel("accuracy (%)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, out, "fig2_tradeoff.png")


def fig_threshold(out):
    rows = list(csv.DictReader(open(os.path.join(out, "sweep.csv"))))
    names = [n for n in ORDER if any(r["dataset"] == n for r in rows)]
    fig, axes = plt.subplots(2, 4, figsize=(10, 5), sharex=True)
    for ax in axes.flat[len(names):]:
        ax.set_visible(False)
    for ax, name in zip(axes.flat, names):
        for mode, key in (("policy", "task"), ("similarity", "content")):
            sel = [r for r in rows if r["dataset"] == name and r["mode"] == mode]
            taus = [float(r["tau"]) for r in sel]
            good = [int(r["reused"]) - int(r["wrong"]) for r in sel]
            bad = [int(r["wrong"]) for r in sel]
            label = "validated policy" if mode == "policy" else "similarity only"
            ax.plot(taus, good, color=COLORS[key], marker="o", markeredgecolor=SURFACE, label=label + ": correct reuses")
            ax.plot(taus, bad, color=COLORS[key], marker="x", lw=1.0, label=label + ": wrong reuses")
        cal = [r for r in rows if r["dataset"] == name and r["mode"] == "calibrated"][0]
        ax.axvline(float(cal["tau"]), color=MUTED, lw=1)
        ax.text(float(cal["tau"]), ax.get_ylim()[1] * 0.97, " calibrated %g" % round(float(cal["tau"]), 3),
                color=INK2, fontsize=7.5, va="top")
        ax.set_title(name)
    for ax in axes[1]:
        ax.set_xlabel("similarity threshold")
    for ax in axes[:, 0]:
        ax.set_ylabel("tasks")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, out, "fig4_threshold.png")


def fig_datasets(out, table):
    metrics = [("input_tokens", "input tokens per task (K)", 1000), ("tool_calls", "tool calls per task", 1),
               ("requests", "model requests per task", 1), ("seconds", "completion time per task (s)", 1),
               ("passed", "accuracy (%)", 0.01), ("minhash_seconds", "MinHash overhead per task (ms)", 0.001)]
    configs = ["baseline", "task", "content", "both"]
    names = present(table)
    fig, axes = plt.subplots(3, 2, figsize=(10, 8.4))
    x = np.arange(len(names))
    width = 0.2
    for ax, (key, label, scale) in zip(axes.flat, metrics):
        for k, config in enumerate(configs):
            values = [per_task(table[(n, config)], key) / scale if (n, config) in table else 0 for n in names]
            ax.bar(x + (k - 1.5) * width, values, width * 0.9, color=COLORS[config], label=NAMES[config])
        ax.set_xticks(x, names)
        ax.set_ylabel(label)
        ax.grid(axis="x", visible=False)
    axes.flat[4].set_ylim(0, 105)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, out, "fig3_datasets.png")


def place_labels(ax, points):
    ax.figure.canvas.draw()
    scale = ax.figure.dpi / 72
    pixels = [ax.transData.transform((x, y)) for x, y, _ in points]
    boxes = [(px - 4 * scale, py - 4 * scale, px + 4 * scale, py + 4 * scale) for px, py in pixels]
    for (x, y, name), (px, py) in sorted(zip(points, pixels), key=lambda item: -item[0][1]):
        w, h = len(name) * 4.3 * scale, 9 * scale
        for dx, dy, ha in ((5, 2, "left"), (5, -11, "left"), (-5, 2, "right"), (-5, -11, "right"), (5, 12, "left")):
            x0 = px + dx * scale - (w if ha == "right" else 0)
            box = (x0, py + dy * scale, x0 + w, py + dy * scale + h)
            inside = ax.bbox.x0 <= box[0] and box[2] <= ax.bbox.x1 and ax.bbox.y0 <= box[1] and box[3] <= ax.bbox.y1
            if inside and not any(box[0] < b[2] and b[0] < box[2] and box[1] < b[3] and b[1] < box[3] for b in boxes):
                break
        boxes.append(box)
        ax.annotate(name, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, fontsize=7.5, color=INK2)


def fig_overhead(out, table):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6), sharex=True, sharey=True)
    placed = []
    for ax, config in zip(axes, ("task", "content", "both")):
        points = []
        for name in present(table):
            t = table.get((name, config))
            if t is None:
                continue
            points.append((1000 * per_task(t, "minhash_seconds"),
                           per_task(table[(name, "baseline")], "seconds") - per_task(t, "seconds"), name))
        ax.scatter([p[0] for p in points], [p[1] for p in points], s=46, color=COLORS[config], edgecolors=SURFACE,
                   linewidths=1.6, zorder=3)
        ax.axhline(0, color=AXIS, lw=1)
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator([0.1, 0.3, 1, 3, 10, 30, 100, 300]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "%g" % v))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set(title=NAMES[config], xlabel="MinHash overhead per task (ms)")
        placed.append((ax, points))
    axes[0].set_ylabel("completion time saved per task (s)")
    fig.tight_layout()
    for ax, points in placed:
        place_labels(ax, points)
    save(fig, out, "fig5_overhead.png")


def paired(rows, names):
    by = {(r["config"], r["dataset"], r["tid"]): r for r in rows}
    lines = ["", "Table D. Paired view: the same task ids in a configuration and in the baseline.", "",
             "| dataset | configuration | reused (history / earlier task) | baseline accuracy on reused | "
             "baseline tokens of reused (K) | share of baseline tokens | agent tasks | accuracy config / baseline | "
             "tokens config / baseline (K) | median tokens config / baseline (K) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name in names:
        for config in ("task", "content", "both"):
            tids = [r["tid"] for r in rows if r["config"] == config and r["dataset"] == name
                    and ("baseline", name, r["tid"]) in by]
            reused = [t for t in tids if by[(config, name, t)]["reused"]]
            agent = [t for t in tids if not by[(config, name, t)]["reused"]]

            def total(cfg, ts, key):
                return sum(by[(cfg, name, t)][key] for t in ts)

            def median(cfg, ts):
                return float(np.median([by[(cfg, name, t)]["input_tokens"] for t in ts])) if ts else 0.0

            history = sum(by[(config, name, t)]["match"].startswith("H-") for t in reused)
            lines.append("| %s | %s | %d (%d / %d) | %d/%d | %.0f | %.0f%% | %d | %d / %d | %.0f / %.0f | %.1f / %.1f |" % (
                name, NAMES[config], len(reused), history, len(reused) - history, total("baseline", reused, "passed"),
                len(reused), total("baseline", reused, "input_tokens") / 1000,
                100 * total("baseline", reused, "input_tokens") / total("baseline", tids, "input_tokens"), len(agent),
                total(config, agent, "passed"), total("baseline", agent, "passed"), total(config, agent, "input_tokens") / 1000,
                total("baseline", agent, "input_tokens") / 1000, median(config, agent) / 1000,
                median("baseline", agent) / 1000))
    return lines


def make_all(out, rows):
    import kickoff
    table = kickoff.summarize(rows)
    text = tables(out, table) + "\n" + "\n".join(paired(rows, present(table)))
    with open(os.path.join(out, "tables.md"), "w") as handle:
        handle.write(text + "\n")
    print(text)
    fig_minhash(out)
    fig_tradeoff(out, table)
    fig_datasets(out, table)
    fig_overhead(out, table)
    if os.path.exists(os.path.join(out, "sweep.csv")):
        fig_threshold(out)


def present(table):
    return [n for n in ORDER if (n, "baseline") in table]


def pct(new, old):
    return "%+.0f%%" % (100.0 * (new - old) / old) if old else "-"


def tables(out, table):
    names, configs = present(table), ["baseline", "task", "content", "both"]
    lines = ["Table A. All %d datasets together." % len(names), "",
             "| configuration | accuracy | reused (wrong) | tool calls | model requests | input tokens | completion time | MinHash overhead |",
             "|---|---|---|---|---|---|---|---|"]
    total = {}
    for config in configs:
        t = {k: sum(table[(n, config)][k] for n in names if (n, config) in table)
             for k in ("n", "passed", "reused", "wrong_reuse", "tool_calls", "requests", "input_tokens", "seconds",
                       "minhash_seconds")}
        total[config] = t
        b = total["baseline"]
        lines.append("| %s | %d/%d (%.1f%%) | %d (%d) | %d (%s) | %d (%s) | %.1fM (%s) | %.0f min (%s) | %.2f s |" % (
            NAMES[config], t["passed"], t["n"], 100.0 * t["passed"] / t["n"], t["reused"], t["wrong_reuse"],
            t["tool_calls"], pct(t["tool_calls"], b["tool_calls"]), t["requests"], pct(t["requests"], b["requests"]),
            t["input_tokens"] / 1e6, pct(t["input_tokens"], b["input_tokens"]), t["seconds"] / 60,
            pct(t["seconds"], b["seconds"]), t["minhash_seconds"]))
    lines += ["", "Table B. Per dataset, per task averages (accuracy over all tasks).", "",
              "| dataset | configuration | accuracy | reused | tool calls | requests | input tokens (K) | time (s) | overhead (ms) |",
              "|---|---|---|---|---|---|---|---|---|"]
    for n in names:
        for config in configs:
            t = table.get((n, config))
            if t is None:
                continue
            lines.append("| %s | %s | %d/%d | %d | %.1f | %.1f | %.1f | %.1f | %.1f |" % (
                n, NAMES[config], t["passed"], t["n"], t["reused"], per_task(t, "tool_calls"), per_task(t, "requests"),
                per_task(t, "input_tokens") / 1000, per_task(t, "seconds"), 1000 * per_task(t, "minhash_seconds")))
    lines += ["", "Table C. Content layer inside the agent (content-level reuse only).", "",
              "| dataset | tool outputs | parts folded | output chars in | output chars out | kept |",
              "|---|---|---|---|---|---|"]
    for n in names:
        t = table.get((n, "content"))
        if t:
            lines.append("| %s | %d | %d | %d | %d | %.1f%% |" % (n, t["outputs"], t["folded"], t["chars_in"],
                                                                 t["chars_out"], 100.0 * t["chars_out"] / max(1, t["chars_in"])))
    with open(os.path.join(out, "tables.md"), "w") as handle:
        handle.write("\n".join(lines) + "\n")
    return "\n".join(lines)
