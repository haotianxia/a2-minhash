# MinHash reuse for the Spartacus agent

Haotian Xia. The assignment asks for three files:

| file | what it does |
|---|---|
| `minhash.py` | word 3-shingles, MinHash with 128 hash functions (seed 1115), banded LSH |
| `policy.py` | task-level reuse: `TaskReuse(dataset_dir)` with `lookup(task)` and `add(tid, task, answer)` |
| `content.py` | content-level reuse: `ContentReuse().tools(workdir)` returns the harness read tools with near-duplicate outputs folded into exact diffs |

`kickoff.py` runs the four configurations and the offline studies, `plotty.py` draws the figures, and `writeup.pdf` is the report. The harness, datasets and verifiers are unchanged.

## Setup

```bash
git clone https://github.com/haotianxia/a2-minhash && cd a2-minhash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
export GEMINI_API_KEY=YOUR_KEY
```

`requirements.txt` installs Spartacus from GitHub at commit `711db62`.

## Reproduce the report

```bash
python kickoff.py run --workers 16 --out out2   # all 420 tasks of 8 datasets x 4 configs, gemini-3.5-flash-lite, one row per task in out2/results/
python kickoff.py summary --out out2            # out2/summary.csv
python kickoff.py sweep --out out2              # threshold sweep from the baseline answers, out2/sweep.csv (no model calls)
python kickoff.py replay --out out2             # content layer over every scenario of hosts, configs, code and logs, out2/replay.csv (no model calls)
python kickoff.py trajectories --out out2       # baseline transcripts replayed through the content layer, out2/trajectories.csv (no model calls)
python kickoff.py plots --out out2              # out2/tables.md, out2/minhash_error.csv and out2/plots/*.png
```

`out/` holds the reported run: `out/results/` (one row per task and configuration), `out/tables.md`, the CSV files, `out/plots/` and `out/runs.tar.gz` with the working directory of every task and the harness session log of each of the 1,400 agent runs. `run` resumes from the rows in its output folder, so `run` without `--out` makes no model calls in a fresh clone. To recompute the reported tables and figures from the shipped transcripts, run `tar xzf out/runs.tar.gz -C out` and then the last five commands without `--out`.

Held-out or regenerated tasks of any of the ten datasets: put the dataset folder (with `env/`, `hidden/` and a copy of that dataset's `verify.py`) directly under `assignment2/datasets/` with any folder name, then run `python kickoff.py run --datasets assignment2/datasets/<folder> --out <new folder>`, with a new `--out` for each held-out folder (kickoff names its results after the part of the folder name before the first hyphen). `python kickoff.py summary --out <new folder>` then prints accuracy and cost per configuration; pass the same `--datasets` to `sweep` and `trajectories`, and note that `plots` draws only datasets whose folder name, up to the first hyphen, is tickets, bugs, docs, citations, hosts, configs, code or logs (for example `tickets-h1`), and stops with an error when the output folder holds none of them. `TaskReuse` recognises the dataset from the fields of its history records, so the folder name does not matter to the policy.

| option | meaning |
|---|---|
| `--datasets tickets,hosts` | dataset folder names under `assignment2/datasets/`, or paths to such folders (default: tickets, bugs, docs, citations, hosts, configs, code, logs; for `replay`, which needs `env/scenarios/`: hosts, configs, code, logs) |
| `--configs baseline,both` | any of `baseline`, `task`, `content`, `both` (default: all four) |
| `--workers 16` | number of (configuration, dataset) streams run in parallel (default: 8); tasks inside a stream run in `seq` order |
| `--limit 30` | only the first 30 tasks (by `seq`) of each dataset, for a cheaper run (default: all tasks) |
| `--out DIR` | output folder (default `out/`) |

A stream stops at the first task whose model calls still fail after three attempts (for example when API credit runs out) and does not record that task; running the same command again resumes every stream from its rows in the output folder's `results/`. Every task runs in its own working directory under the output folder's `runs/`, which keeps `answer.json` and the harness session log.

## Using the two layers with the harness

```python
from spartacus import Harness
from content import ContentReuse
from policy import TaskReuse

reuse = TaskReuse("assignment2/datasets/tickets")
hit = reuse.lookup(task)                      # None, or {"match", "sim", "tau", "answer"}
if hit is None:
    layer = ContentReuse()
    Harness(workdir, model="gemini-3.5-flash-lite", extra_tools=layer.tools(workdir)).run(prompt)
# after verify.py passes for the written answer:
reuse.add(task["tid"], task, answer)
```

---

# COMP 480/580 — Assignment 2: MinHash for Efficient Agent Trajectories

In this assignment, you will explore whether **MinHash** can improve the efficiency of an AI agent by identifying and reusing previously processed information without compromising task correctness.

The goal is simple:

> **Use MinHash to reduce redundant agent computation, tool calls, input-token usage, and task completion time while preserving verified task accuracy.**

You will use this repository and the [Spartacus agentic harness](https://github.com/sahiljoshi515/Spartacus). The tasks, evaluation datasets, and verifiers are already provided.

## Repository Structure

Each dataset follows this structure:

```text
datasets/<dataset_name>/
├── env/
│   ├── INSTRUCTIONS.md
│   ├── history.jsonl
│   └── incoming/
│       ├── <task_id>.json
│       └── ...
├── hidden/
│   └── labels.json
├── verify.py
├── README.md
└── MANIFEST.json
```

- `env/INSTRUCTIONS.md` contains the dataset-specific task instructions.
- `env/history.jsonl` contains previously completed tasks and their verified answers.
- `env/incoming/` contains new tasks the agent must solve.
- `verify.py` deterministically checks whether the generated `answer.json` is correct.
- `hidden/` contains ground-truth information for evaluation. **Neither the agent nor your MinHash implementation may access it to determine answers. We will check this during grading!**

The agent may use tools such as `read_file`, `grep`, `list_files`, and `bash` to solve each task. Its output must follow the dataset-specific `answer.json` format; different datasets may require different fields.

## Datasets

There are **10 datasets**, divided into two categories:

**Record-based datasets** focus primarily on identifying previously solved tasks and safely reusing their answers:

- `tickets` — technical support resolutions.
- `bugs` — software components and root causes.
- `docs` — document classification and duplicate identification.
- `citations` — bibliographic verification.

**Corpus-based datasets** also contain related files that may produce repeated or near-duplicate tool outputs:

- `hosts` — host inventory snapshots.
- `configs` — configuration differences across environments.
- `code` — similar source-code files and implementation errors.
- `threads` — email conversations.
- `revisions` — document version changes.
- `logs` — system logs and incidents.

For the experimental evaluation, **use 8 datasets of your choice** from the 10 provided, and identify them in your report.

## Task

First, establish a **baseline** by running the original agent on the tasks without MinHash-based optimizations. The agent should use the provided instructions and tools, produce `answer.json`, and be evaluated by the corresponding `verify.py`.

Next, implement two MinHash-based optimizations:

### 1. Implement MinHash and LSH (`minhash.py`)

Implement a MinHash-based similarity search method using **three-word shingles** and **128 hash functions**.

For a set of shingles, each MinHash signature component records the minimum hash value under one hash function. The fraction of matching components between two signatures estimates their Jaccard similarity:

$J(A,B) = \frac{|A \cap B|}{|A \cup B|}.$

Use **locality-sensitive hashing (LSH)** to efficiently retrieve potentially similar items. You may use standard hashing libraries, but **you must implement the MinHash logic yourself**.

### 2. Task-Level Reuse (`policy.py`)

Before invoking the agent, compare an incoming task against previously verified tasks using your MinHash-based search method.

- If a sufficiently similar **and valid** match is found, reuse its verified answer without invoking the agent.
- Otherwise, execute the agent normally.
- Index newly generated answers **only after successful verification**.

Your method must distinguish genuine duplicates from **hard negatives**: tasks that look similar but require different answers. High MinHash similarity alone does not guarantee that answer reuse is safe.

### 3. Content-Level Reuse (`content.py`)

During agent execution, identify repeated or near-duplicate outputs returned by tools. Instead of repeatedly passing similar content to the model, reuse previously observed information and provide **only the exact differences** when this is safe and useful.

Your implementation must preserve answer-critical information, including **numerical values, identifiers, and configuration changes**. If reuse is unsafe or provides insufficient savings, return the **original tool output**.

You should ideally implement these optimizations in just the three Python files above. **The provided agentic harness, datasets, and verifiers should remain unchanged.**

## Experiments

Evaluate these **four configurations** on the same tasks:

1. **Baseline:** Original agent without MinHash.
2. **Task-level reuse only.**
3. **Content-level reuse only.**
4. **Combined reuse:** Both task-level and content-level reuse.

Use the **same model, prompts, tasks, and execution settings** for a fair comparison. Report the following metrics:

- **Task accuracy:** Percentage of tasks that pass the deterministic verifier.
- **Tool calls:** Number of tool calls made by the agent.
- **Model requests:** Number of model invocations.
- **Input-token usage:** Total input tokens processed by the model.
- **Total completion time:** End-to-end time per task or evaluation run, **including MinHash lookup and content-processing overhead**.

Present appropriate **plots or tables** that demonstrate the trade-off between efficiency and correctness. Useful analyses include accuracy versus token savings, the effect of similarity thresholds, performance across datasets, and MinHash overhead versus time saved.

**The goal is not simply to maximize reuse:** An incorrect answer caused by unsafe reuse is a failure, even if it reduces token usage or tool calls.

## Questions to Answer in Your Report

1. **When is task reuse beneficial?** Which tasks are most suitable for MinHash-based reuse, and when can a duplicate eliminate agent execution entirely?
2. **Is high similarity sufficient for safe reuse?** Give an example of a hard negative and explain how your method avoids reusing the wrong answer.
3. **How does the similarity threshold affect performance?** Discuss the trade-off between detecting more duplicates and incorrectly matching hard negatives. Is one threshold appropriate for every dataset?
4. **When is content-level reuse effective?** Identify a dataset where it reduces token usage and explain why it may help less on another dataset.
5. **Which optimization is more valuable?** Compare task-level and content-level reuse in accuracy, token savings, tool calls, and completion time. When would you prefer each one?
6. **What are the limitations of MinHash for agents?** Discuss when lexical similarity fails to capture task equivalence or when MinHash overhead outweighs the savings. Suggest one improvement.

Use experimental evidence and concrete examples wherever appropriate.

## Reproducibility

Your submitted implementation must reproduce the reported experiments without manual intervention.

- Use fixed random seeds whenever applicable.
- Include the **exact commands** needed to reproduce your experiments in the report.
- Ensure your code works with the provided agentic harness and evaluation scripts **without modifying them**.
- Do not hard-code task answers or access hidden evaluation labels to determine answers.

**We will independently execute your implementation on the provided and additional held-out tasks.**

## Grading (100 Points)

| Component | Points |
|---|---:|
| MinHash implementation | 20 |
| Task-level reuse and correctness | 25 |
| Content-level reuse and correctness | 25 |
| Experimental evaluation | 15 |
| Report and discussion | 10 |
| Code quality and reproducibility | 5 |
| **Total** | **100** |

## Submission

Submit:

- A github repository link to your implementation. As specified earlier, you only need to implement 3 files for a successful implementation. We want the code to be minimal and readable.
- **A short report:** Describe your implementation, identify the 8 datasets evaluated, provide experimental results and plots, answer the six discussion questions, and include the exact commands to reproduce your experiments.

Good luck!
