import argparse
import csv
import glob
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from spartacus import Harness

from assignment2.datasets.common import norm_code
from content import WRAPPED, ContentReuse, call_label
from minhash import similarity
from policy import TaskReuse, dataset_name, load_jsonl

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "assignment2", "datasets")
MODEL = "gemini-3.5-flash-lite"
PROMPT = "Solve the request in task.json: follow INSTRUCTIONS.md and write answer.json in this directory."
DATASETS = "tickets,bugs,docs,citations,hosts,configs,code,logs"
CORPUS = "hosts,configs,code,logs"
CONFIGS = {"baseline": (False, False), "task": (True, False), "content": (False, True), "both": (True, True)}
STAGED = ("INSTRUCTIONS.md", "history.jsonl", "db.json")
COUNTS = ("tool_calls", "requests", "input_tokens", "output_tokens")
TAUS = [round(0.1 * i, 1) for i in range(1, 11)]
LOCK = threading.Lock()


def data_dir(name):
    return name if os.path.isdir(name) else os.path.join(DATA, name)


def read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def tasks_of(data):
    folder = os.path.join(data, "env", "incoming")
    return sorted((read_json(os.path.join(folder, f)) for f in os.listdir(folder)), key=lambda t: t["seq"])


def stage(data, task, work):
    env = os.path.join(data, "env")
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    for name in STAGED:
        if os.path.exists(os.path.join(env, name)):
            shutil.copy(os.path.join(env, name), work)
    shutil.copy(os.path.join(env, "incoming", task["tid"] + ".json"), os.path.join(work, "task.json"))
    if "scenario" in task:
        shutil.copytree(scenario_dir(data, task), work, dirs_exist_ok=True)


def scenario_dir(data, task):
    scenarios = os.path.join(data, "env", "scenarios")
    return os.path.join(scenarios, next(s for s in os.listdir(scenarios) if norm_code(s) == norm_code(task["scenario"])))


def unstage(work):
    for name in os.listdir(work):
        if name not in ("answer.json", ".spartacus"):
            path = os.path.join(work, name)
            shutil.rmtree(path) if os.path.isdir(path) else os.remove(path)


def verify(data, tid, path):
    done = subprocess.run([sys.executable, os.path.join(data, "verify.py"), "--tid", tid, "--answer", path],
                          capture_output=True, text=True)
    return done.returncode == 0


def counter(stats):
    def on_event(kind, payload):
        if kind == "assistant":
            stats["requests"] += 1
            stats["input_tokens"] += payload["usage"]["input"]
            stats["output_tokens"] += payload["usage"]["output"]
        elif kind == "tool_start":
            stats["tool_calls"] += 1
    return on_event


def run_agent(data, task, use_content, work):
    for attempt in range(3):
        stage(data, task, work)
        stats = dict.fromkeys(COUNTS, 0)
        layer = ContentReuse() if use_content else None
        agent = Harness(work, model=MODEL, on_event=counter(stats), extra_tools=layer.tools(work) if layer else None)
        start = time.perf_counter()
        try:
            agent.run(PROMPT)
            stats["error"] = ""
        except Exception as error:
            stats["error"] = "%s: %s" % (type(error).__name__, str(error)[:300])
        stats["agent_seconds"] = time.perf_counter() - start
        if layer:
            stats.update({"c_" + k: v for k, v in layer.stats.items()})
        if not stats["error"]:
            break
        time.sleep(30 * (attempt + 1))
    stats["attempts"] = attempt + 1
    return stats


def run_task(data, task, config, reuse, work):
    use_task, use_content = CONFIGS[config]
    row = {"tid": task["tid"], "seq": task["seq"], "reused": False, "error": ""}
    row.update(dict.fromkeys(COUNTS, 0))
    before = reuse.seconds if reuse else 0.0
    start = time.perf_counter()
    hit = reuse.lookup(task) if reuse else None
    row["lookup_seconds"] = time.perf_counter() - start
    answer_path = os.path.join(work, "answer.json")
    if hit:
        os.makedirs(work, exist_ok=True)
        with open(answer_path, "w", encoding="utf-8") as handle:
            json.dump(hit["answer"], handle)
        row.update(reused=True, match=hit["match"], sim=hit["sim"], tau=hit["tau"], agent_seconds=0.0)
    else:
        row.update(run_agent(data, task, use_content, work))
        if reuse:
            row["tau"] = reuse.tau
    row["seconds"] = row["lookup_seconds"] + row["agent_seconds"]
    row["minhash_seconds"] = (reuse.seconds - before if reuse else 0.0) + row.get("c_seconds", 0.0)
    try:
        row["answer"] = read_json(answer_path)
    except (OSError, ValueError):
        row["answer"] = None
    row["passed"] = verify(data, task["tid"], answer_path)
    if os.path.isdir(work):
        unstage(work)
    return row


def run_stream(config, data, out, limit=None):
    name = dataset_name(data)
    path = os.path.join(out, "results", config, name + ".jsonl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    done = {r["tid"]: r for r in load_jsonl(path)} if os.path.exists(path) else {}
    start = time.perf_counter()
    reuse = TaskReuse(data) if CONFIGS[config][0] else None
    build = time.perf_counter() - start
    for task in tasks_of(data)[:limit]:
        row = done.get(task["tid"])
        fresh = row is None
        if fresh:
            row = run_task(data, task, config, reuse, os.path.join(out, "runs", config, name, task["tid"]))
            row.update(dataset=name, config=config, build_seconds=build)
            if row["error"]:
                print("%-8s %-9s %s stopped: %s" % (config, name, task["tid"], row["error"][:120]), flush=True)
                return
        if reuse and row["passed"] and isinstance(row["answer"], dict):
            before = reuse.seconds
            reuse.add(task["tid"], task, row["answer"])
            row.setdefault("index_seconds", reuse.seconds - before)
        if fresh:
            with LOCK:
                with open(path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row) + "\n")
                print("%-8s %-9s %s %s %s tok=%d calls=%d %.1fs %s" % (
                    config, name, task["tid"], "PASS" if row["passed"] else "FAIL", "reuse" if row["reused"] else "agent",
                    row["input_tokens"], row["tool_calls"], row["seconds"], row["error"][:80]), flush=True)


def cmd_run(args):
    streams = [(c, data_dir(d)) for d in args.datasets.split(",") for c in args.configs.split(",")]
    with ThreadPoolExecutor(args.workers) as pool:
        for _ in pool.map(lambda s: run_stream(s[0], s[1], args.out, args.limit), streams):
            pass


def results(out, limit=None):
    rows = []
    for config in CONFIGS:
        folder = os.path.join(out, "results", config)
        for name in sorted(os.listdir(folder)) if os.path.isdir(folder) else []:
            rows += [r for r in load_jsonl(os.path.join(folder, name)) if limit is None or r["seq"] <= limit]
    return rows


def summarize(rows):
    table = {}
    for r in rows:
        t = table.setdefault((r["dataset"], r["config"]), {"n": 0, "passed": 0, "reused": 0, "wrong_reuse": 0,
                                                            "tool_calls": 0, "requests": 0, "input_tokens": 0,
                                                            "output_tokens": 0, "seconds": 0.0, "minhash_seconds": 0.0,
                                                            "folded": 0, "outputs": 0, "chars_in": 0, "chars_out": 0,
                                                            "errors": 0})
        t["n"] += 1
        t["passed"] += r["passed"]
        t["reused"] += r["reused"]
        t["wrong_reuse"] += r["reused"] and not r["passed"]
        for k in COUNTS + ("seconds", "minhash_seconds"):
            t[k] += r[k]
        t["seconds"] += r.get("index_seconds", 0.0)
        t["minhash_seconds"] += r.get("index_seconds", 0.0)
        t["build_seconds"] = max(t.get("build_seconds", 0.0), r["build_seconds"])
        for k in ("folded", "outputs", "chars_in", "chars_out"):
            t[k] += r.get("c_" + k, 0)
        t["errors"] += bool(r["error"])
    for t in table.values():
        t["seconds"] += t["build_seconds"]
        t["minhash_seconds"] += t["build_seconds"]
    return table


def cmd_summary(args):
    table = summarize(results(args.out, args.limit))
    path = os.path.join(args.out, "summary.csv")
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", "config"] + list(next(iter(table.values()))))
        for (name, config), t in sorted(table.items()):
            writer.writerow([name, config] + list(t.values()))
    print("wrote", path)
    for (name, config), t in sorted(table.items()):
        print("%-9s %-8s acc %3d/%-3d reuse %2d (wrong %d) calls %5d req %5d tok %9d time %7.1fs mh %.3fs fold %d/%d err %d" % (
            name, config, t["passed"], t["n"], t["reused"], t["wrong_reuse"], t["tool_calls"], t["requests"],
            t["input_tokens"], t["seconds"], t["minhash_seconds"], t["folded"], t["outputs"], t["errors"]))


def nearest(pol, task):
    sig = pol.mh.text(pol.text(task))
    return max(((similarity(sig, pol.sig[r]), r) for r in pol.lsh.query(sig)), default=None)


def cmd_sweep(args):
    rows = results(args.out, args.limit)
    base = {(r["dataset"], r["tid"]): r for r in rows if r["config"] == "baseline"}
    cache = {}

    def check(data, tid, answer):
        key = (data, tid, json.dumps(answer, sort_keys=True))
        if key not in cache:
            path = os.path.join(args.out, "sweep_answer_%d.json" % threading.get_ident())
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(answer, handle)
            cache[key] = verify(data, tid, path)
            os.remove(path)
        return cache[key]

    out = []
    for name in args.datasets.split(","):
        data = data_dir(name)
        tasks = tasks_of(data)[:args.limit]
        for mode in ("policy", "similarity", "calibrated"):
            for tau in TAUS if mode != "calibrated" else [None]:
                pol = TaskReuse(data, floor=tau or 0.3, calibrate=mode == "calibrated")
                reused = wrong = 0
                for task in tasks:
                    if mode == "similarity":
                        best = nearest(pol, task)
                        hit = {"answer": pol.answer[best[1]]} if best and best[0] >= tau else None
                    else:
                        hit = pol.lookup(task)
                    outcome = base.get((dataset_name(data), task["tid"]))
                    if hit:
                        ok = check(data, task["tid"], hit["answer"])
                        reused += 1
                        wrong += not ok
                        answer = hit["answer"]
                    else:
                        ok = bool(outcome and outcome["passed"])
                        answer = outcome["answer"] if outcome else None
                    if ok and isinstance(answer, dict):
                        pol.add(task["tid"], task, answer)
                out.append({"dataset": dataset_name(data), "mode": mode, "tau": tau if tau else round(pol.tau, 3),
                            "tasks": len(tasks), "reused": reused, "wrong": wrong})
                print(out[-1])
    with open(os.path.join(args.out, "sweep.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out[0]))
        writer.writeheader()
        writer.writerows(out)


BATCH = 'for f in $(find . -type f | sort); do echo "=== $f"; cat "$f"; done'


def replay(name, max_ratio, batched):
    data = data_dir(name)
    stats = {"files": 0, "folded": 0, "chars_in": 0, "chars_out": 0, "seconds": 0.0}
    scenarios = os.path.join(data, "env", "scenarios")
    for sid in sorted(os.listdir(scenarios)):
        root = os.path.join(scenarios, sid)
        layer = ContentReuse(max_ratio=max_ratio)
        tools = {t.name: t for t in layer.tools(root)}
        files = sorted(os.path.relpath(os.path.join(b, f), root) for b, _, fs in os.walk(root) for f in fs)
        if batched:
            tools["bash"].run(command=BATCH)
        else:
            for f in files:
                tools["read_file"].run(path=f)
        stats["files"] += len(files)
        for k in ("folded", "chars_in", "chars_out", "seconds"):
            stats[k] += layer.stats[k]
    return stats


def cmd_replay(args):
    out = []
    for name in args.datasets.split(","):
        for batched in (False, True):
            for ratio in (0.5, 0.6, 0.75, 0.9, 1.0):
                s = replay(name, ratio, batched)
                out.append({"dataset": name, "pattern": "one bash call" if batched else "one read_file per file",
                            "max_ratio": ratio, "files": s["files"], "folded": s["folded"], "chars_in": s["chars_in"],
                            "chars_out": s["chars_out"], "kept": round(s["chars_out"] / s["chars_in"], 3),
                            "ms_per_file": round(1000 * s["seconds"] / s["files"], 3)})
                print(out[-1])
    with open(os.path.join(args.out, "replay.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out[0]))
        writer.writeheader()
        writer.writerows(out)


def replay_trajectory(path, root):
    layer = ContentReuse()
    if root:
        layer.tools(root)
    messages = load_jsonl(path)
    total = sum(m["role"] == "assistant" for m in messages)
    request, pending, saved = 0, [], 0.0
    for m in messages:
        if m["role"] == "assistant":
            request += 1
            pending = list(m.get("tool_calls") or [])
        elif m["role"] == "tool" and pending:
            call = pending.pop(0)
            if call["name"] in WRAPPED:
                folded = layer.fold(call_label(call["name"], call["args"]), m["text"])
                saved += (len(m["text"]) - len(folded)) / 4 * (total - request)
    return layer.stats, saved


def cmd_trajectories(args):
    rows = [r for r in results(args.out, args.limit) if r["config"] == "baseline"]
    out = []
    for name in args.datasets.split(","):
        data = data_dir(name)
        tasks = {t["tid"]: t for t in tasks_of(data)}
        agg = {"dataset": dataset_name(data), "tasks": 0, "outputs": 0, "folded": 0, "identical": 0, "chars_in": 0,
               "chars_out": 0, "tokens_saved": 0.0, "baseline_tokens": 0, "tasks_with_fold": 0}
        for r in rows:
            if r["dataset"] != agg["dataset"]:
                continue
            path = glob.glob(os.path.join(args.out, "runs", "baseline", agg["dataset"], r["tid"], ".spartacus",
                                          "sessions", "*.jsonl"))[0]
            task = tasks[r["tid"]]
            stats, saved = replay_trajectory(path, scenario_dir(data, task) if "scenario" in task else None)
            agg["tasks"] += 1
            agg["tasks_with_fold"] += stats["folded"] > 0
            agg["tokens_saved"] += saved
            agg["baseline_tokens"] += r["input_tokens"]
            for k in ("outputs", "folded", "identical", "chars_in", "chars_out"):
                agg[k] += stats[k]
        agg["tokens_saved"] = int(agg["tokens_saved"])
        agg["share_saved"] = round(agg["tokens_saved"] / max(1, agg["baseline_tokens"]), 4)
        out.append(agg)
        print(agg)
    with open(os.path.join(args.out, "trajectories.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out[0]))
        writer.writeheader()
        writer.writerows(out)


def cmd_plots(args):
    import plotty
    plotty.make_all(args.out, results(args.out, args.limit))


def main(argv=None):
    parser = argparse.ArgumentParser(description="MinHash reuse experiments on the Assignment 2 datasets.")
    parser.add_argument("command", choices=["run", "summary", "sweep", "replay", "trajectories", "plots"])
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--configs", default=",".join(CONFIGS))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default=os.path.join(ROOT, "out"))
    args = parser.parse_args(argv)
    args.datasets = args.datasets or (CORPUS if args.command == "replay" else DATASETS)
    os.makedirs(args.out, exist_ok=True)
    {"run": cmd_run, "summary": cmd_summary, "sweep": cmd_sweep, "replay": cmd_replay,
     "trajectories": cmd_trajectories, "plots": cmd_plots}[args.command](args)


if __name__ == "__main__":
    main()
