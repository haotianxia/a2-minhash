---
title: "Assignment 2 Report: MinHash Reuse for Agent Trajectories"
author: "Haotian Xia"
geometry: margin=1in
header-includes:
  - \usepackage{float}
  - \floatplacement{figure}{H}
  - \usepackage{etoolbox}
  - \AtBeginEnvironment{longtable}{\footnotesize}
---

Code: <https://github.com/haotianxia/a2-minhash>. The three required files are `minhash.py`, `policy.py` and `content.py`; `kickoff.py` runs the experiments and `plotty.py` draws the figures.

**Summary.** I ran the four configurations on all 420 incoming tasks of eight datasets. Task-level reuse answered 140 tasks by reusing verified answers without running the agent, and none of these answers failed the verifier; input tokens fell by 32%, completion time by 24%, and accuracy rose from 92.4% to 95.5%. Content-level reuse loses no information, but its savings are small: replayed over the baseline transcripts it saves 1.7% of the input tokens, and in the live runs its effect stays within run-to-run noise.

# 1. Methodology

**MinHash and LSH (`minhash.py`).** A text is lower-cased and split into alphanumeric tokens, and every three consecutive tokens form a shingle. Each shingle is hashed once with 8-byte BLAKE2b and reduced modulo `p = 2^31 - 1`, and 128 functions `h_i(x) = (a_i x + b_i) mod p`, with `a_i` and `b_i` drawn from `numpy.random.default_rng(1115)`, map it to 128 values; the signature keeps the minimum of each function over the set. The fraction of positions where two signatures agree estimates their Jaccard similarity. `LSH(bands, rows)` files a signature under each of its bands, and two items with similarity J share at least one bucket with probability `1 - (1 - J^r)^b`. On 13,785 pairs of history records the estimate has a mean error below 0.01 in every 0.2-wide bin of J and a standard deviation within 0.004 of `sqrt(J(1-J)/128)`, and the measured LSH candidate rates follow the theoretical curves (Figure 1).

![MinHash estimate against the exact Jaccard similarity of record pairs (left) and the probability of becoming an LSH candidate, theory and measured on synthetic pairs (right).](out/plots/fig1_minhash.png){width=75%}

**Task-level reuse (`policy.py`).** MinHash can only say that two tasks look alike, and a hard negative is exactly a task that looks like a solved one and has a different answer. So the policy uses similarity to find candidates and lets each dataset's published decision rule settle whether an answer carries over. `TaskReuse(dataset_dir)` indexes `env/history.jsonl`, recognises the dataset from the answer fields of its records (so a held-out copy works under any folder name), and handles each incoming task in `seq` order:

1. it builds the task text from every field except ids and dates and queries an LSH index with 64 bands of 2 rows, which returns a record at Jaccard 0.3 with probability 0.998;
2. it keeps the candidates whose estimated similarity is at least `tau` and whose validation key, the fields that decide the answer under the rule in the dataset's `INSTRUCTIONS.md` and `README.md`, equals the task's key (table below);
3. it reuses an answer only when all kept candidates agree on it, and takes it from the most similar one.

| dataset | validation key |
|---|---|
| tickets | product, error code (`E-1042` = `e1042`), platform |
| bugs | exception class without module, top frame (`a.b.c` or `a/b.py::c`, no line number), version series `major.minor` |
| docs | identifier (`CHG`, `INC`, `POL`, `REL`) and status class (the first class, in the order approved, rejected, request, resolved, open, final, draft, whose stem occurs in the title or body) |
| citations | DOI, title tokens, (surname, initial) of each author in order, venue mapped through the aliases in `db.json`, year |
| the six corpus datasets | scenario, kind, target |

The threshold is calibrated on the history. `tau` starts at 0.3, and whenever a verified record enters the index while an indexed record with the same key has a different answer, `tau` becomes `max(tau, similarity + 0.1)`; the margin is slightly more than two standard deviations of the estimate at J = 0.5. History alone sets `tau` to 0.725 on tickets, where two records with the same key and different resolutions reach 0.625, and to 0.389 on bugs; the other six datasets keep 0.3. Every answer that passes `verify.py`, whether the agent wrote it or the policy reused it, is added to the index, so duplicates among the incoming tasks are found too. The policy reads only `env/` and the verifier's pass or fail.

**Content-level reuse (`content.py`).** `ContentReuse().tools(workdir)` returns `read_file`, `grep`, `list_files` and `bash` with their original schemas. Passed to `Harness` as `extra_tools`, they replace the built-in tools, so the harness is unchanged. Every tool output goes through `fold`:

1. Outputs under 400 characters pass unchanged. A longer output is split into parts at each line of at most 160 characters that names a scenario file, so a loop that prints `== config/dev.yaml` before each file gives one part per file. If no part reaches 400 characters, as in a grep hit list where every line names a file, the whole output is compared as one part. A `read_file` output is compared as one part without its line numbers.
2. A part identical to an earlier one becomes a short reference, `[reuse] identical to read_file(path=...) shown earlier (95 lines).`
3. Otherwise an LSH index with 32 bands of 4 rows over the earlier parts, on signatures with every hexadecimal token that contains a digit, and every other run of digits, replaced by `0`, returns candidates. For the three most similar ones with similarity at least 0.5 the layer writes an exact line diff (unified format, no context lines) against the unmasked earlier text and keeps the shortest.
4. The diff replaces the part only if it is at most 75% as long. Otherwise the part is returned unchanged and becomes a reference for later parts.

Masking only chooses the reference. The diff runs on the exact text, so every number, identifier or configuration line that changed appears in the folded output, and every line it leaves out is one the model has already read in the same trajectory. I passed all 851 scenario files of the four corpus datasets, the same 96 scenarios printed by one batched `bash` loop each, and the 420 baseline transcripts through the layer. This gives 829 folds (755 diffs and 74 identical repeats), and expanding each fold from its reference reproduces the original part in all 829 cases. The `INSTRUCTIONS.md` of the corpus datasets already tells the agent that a repeated file may come back as a `[reuse]` reference plus the lines that differ.

# 2. Experimental Setup

I use the record datasets tickets, bugs, docs and citations and the corpus datasets hosts, configs, code and logs, with all 420 of their incoming tasks: 70 each for tickets and citations, 60 each for bugs and docs and 40 for each corpus dataset. According to the manifests, 168 of them (40%) duplicate a history record or an earlier task and 83 (20%) are hard negatives. All configurations run `gemini-3.5-flash-lite` in the unmodified Spartacus harness (commit `711db62`) with its defaults, temperature 0.4 and 120 turns, on the unchanged datasets and `verify.py` scripts. Each task runs in a fresh directory with `INSTRUCTIONS.md`, `history.jsonl`, `task.json` and the scenario files or `db.json` it needs, under the same prompt for every task, "Solve the request in task.json: follow INSTRUCTIONS.md and write answer.json in this directory." A task counts as correct when `verify.py` passes. Up to 16 (configuration, dataset) streams run in parallel, and the task-level and combined streams take their tasks in `seq` order, which reuse needs. Tool calls, requests and tokens come from the harness events. Completion time sums lookup time and agent time over the tasks and adds the time to build the index and to index verified answers, so it includes all MinHash work.

Run-to-run variance is large. At temperature 0.4 one task can use 134 times as many input tokens in one run as in another: bugs T-27591a took 5.3M input tokens in the baseline and failed at the 120-turn limit, and 40K tokens in the combined configuration, where neither layer acted. I therefore compare each reused task with the baseline's cost on the same task, and I measure the content layer a second time by replaying the 420 baseline transcripts through it.

The content-level and combined runs of the first 30 tasks of each corpus dataset come from a later session, after I changed how `content.py` treats grep hit lists. In that session a request took 0.66 s on average, against 0.75 s (baseline) and 0.80 s (task-level) on the same tasks, where a few runs stalled at up to 4.4 s per request (the median is 0.61 to 0.64 s in all four), so part of the corpus time savings of these two configurations comes from the session.

# 3. Results

Table 1. All 420 tasks. Time is the sum of per-task completion times; percentages are relative to the baseline.

| configuration | accuracy | reused (wrong) | tool calls | requests | input tokens | time | MinHash overhead per task |
|-------------|----------------|-------|-------------|-------------|-------------|----------------|----------|
| baseline | 388/420 (92.4%) | | 4,831 | 5,251 | 67.3M | 97.4 min | |
| task-level reuse | 401/420 (95.5%) | 140 (0) | 3,432 (-29%) | 3,712 (-29%) | 46.1M (-32%) | 73.6 min (-24%) | 4.1 ms |
| content-level reuse | 387/420 (92.1%) | | 4,774 (-1%) | 5,191 (-1%) | 64.1M (-5%) | 90.1 min (-7%) | 24.2 ms |
| both | 396/420 (94.3%) | 140 (0) | 3,632 (-25%) | 3,910 (-26%) | 53.0M (-21%) | 74.3 min (-24%) | 15.3 ms |

Task-level reuse carries almost all of the savings. The 140 tasks it answered by reuse had cost the baseline 30% of its tool calls and requests, 29% of its input tokens and 27% of its time, and none of the reused answers failed the verifier. Content-level reuse used 1% fewer tool calls and requests, 5% fewer input tokens and 7% less time than the baseline, but paired bootstrap 95% intervals over the 420 tasks include zero for every metric. Combining both saves about as much time as task-level reuse alone but about ten points less of the input tokens (question 5).

\newpage

Table 2. Per dataset (70 tasks for tickets and citations, 60 for bugs and docs, 40 for each corpus dataset). Cells list baseline / task-level / content-level / both; the reused column lists task-level / both.

| dataset | correct | reused | input tokens per task (K) | tool calls per task | time per task (s) |
|---------|------------------|--------|---------------------------|---------------------------|---------------------------|
| tickets | 61 / 66 / 68 / 67 | 18 / 18 | 95 / 110 / 88 / 131 | 10.0 / 8.9 / 10.3 / 9.9 | 14.4 / 12.2 / 14.7 / 13.9 |
| bugs | 54 / 57 / 55 / 55 | 20 / 21 | 346 / 144 / 298 / 262 | 13.7 / 7.9 / 13.1 / 9.8 | 18.3 / 12.0 / 16.9 / 14.2 |
| docs | 55 / 60 / 56 / 59 | 23 / 23 | 189 / 148 / 169 / 118 | 11.1 / 8.6 / 10.5 / 7.5 | 16.0 / 12.8 / 13.3 / 10.9 |
| citations | 58 / 59 / 50 / 55 | 25 / 24 | 256 / 216 / 294 / 206 | 14.5 / 10.8 / 14.4 / 11.5 | 19.0 / 14.9 / 18.6 / 15.7 |
| hosts | 40 / 40 / 40 / 40 | 13 / 13 | 58 / 27 / 50 / 33 | 9.9 / 6.2 / 9.2 / 6.1 | 9.1 / 6.6 / 8.2 / 5.9 |
| configs | 40 / 39 / 38 / 40 | 15 / 15 | 82 / 45 / 71 / 46 | 10.3 / 6.9 / 11.4 / 7.2 | 10.1 / 7.8 / 8.9 / 6.1 |
| code | 40 / 40 / 40 / 40 | 12 / 12 | 56 / 34 / 53 / 42 | 9.9 / 6.9 / 9.3 / 7.2 | 8.3 / 6.2 / 7.1 / 5.1 |
| logs | 40 / 40 / 40 / 40 | 14 / 14 | 71 / 38 / 59 / 44 | 10.6 / 6.6 / 10.7 / 7.0 | 8.6 / 5.3 / 7.3 / 4.9 |

Figure 2 plots accuracy against input tokens per task and Figure 3 shows the per-task metrics of each dataset. Both configurations with task-level reuse use fewer tokens than the baseline on seven of the eight datasets. The exception is tickets, where one runaway run in each, T-c24ed9 with task-level reuse (2.7M tokens) and T-1cb53c in the combined configuration (4.0M), costs more than the 2.0M tokens that the 18 reuses save.

![Accuracy against input tokens per task, one panel per dataset; the grey line marks baseline accuracy.](out/plots/fig2_tradeoff.png){width=85%}

Figure 4 replays the policy over the same 420 tasks for thresholds from 0.1 to 1.0, using the baseline's verified answers for the tasks it does not reuse, and compares it with plain nearest-neighbour reuse, which takes the answer of the most similar indexed record whenever its similarity reaches the threshold.

Table 3 shows the content layer. Inside the content-level runs it folded 253 parts in 4,296 tool outputs. Replayed over the baseline transcripts it makes 201 folds and saves 1.16M input tokens, 1.7% of the baseline; 55% of that comes from eight trajectories that read `history.jsonl` a second time, and everything else saves 0.8%. Task-level work costs 4.1 ms per task including the index build and the content layer 24.2 ms per task (2.4 ms per tool output), against 13.9 s for an average baseline task. Figure 5 sets this overhead against the time saved. The content-level points, from 0.22 s lost per task on tickets to 2.71 s saved on docs, are run-to-run variation and, on the corpus datasets, the faster session of Section 2, since under a least-squares fit of agent time on requests and tokens (question 6) the tokens the layer removes in the replay are worth only 11 to 157 ms per task.

![Performance across datasets: input tokens, tool calls, model requests, completion time, accuracy and MinHash overhead per task.](out/plots/fig3_datasets.png){width=62%}

![Impact of the similarity threshold: correct and wrong reuses of the validated policy and of similarity-only reuse; the vertical line marks the threshold calibrated from history.](out/plots/fig4_threshold.png){width=85%}

\newpage

Table 3. Content layer. Live: the content-level runs. Replay: the 420 baseline transcripts passed through the layer, with tokens saved estimated as a quarter of the removed characters times the number of later requests. Scenario files: every file of every corpus scenario, read one by one.

| dataset | live: parts folded / outputs | replay: folds (identical) | replay: share of baseline tokens saved | scenario files: characters kept |
|---------|------------|------------|------------|------------|
| tickets | 24 / 642 | 24 (20) | 0.69% | |
| bugs | 30 / 712 | 15 (5) | 0.97% | |
| docs | 55 / 558 | 41 (8) | 4.95% | |
| citations | 30 / 932 | 31 (14) | 0.63% | |
| hosts | 3 / 328 | 10 (5) | 2.12% | 66% |
| configs | 74 / 415 | 40 (0) | 2.22% | 57% |
| code | 15 / 324 | 19 (2) | 3.27% | 42% |
| logs | 22 / 385 | 21 (20) | 1.50% | 100% |

![Computational overhead against savings: MinHash time per task and completion time saved per task (baseline minus configuration), one dot per dataset.](out/plots/fig5_overhead.png){width=68%}

# 4. Discussion

**1. When is task reuse beneficial?** Task reuse pays off when an incoming task repeats a verified one and the answer depends only on information that both texts contain. The agent then does not run at all: once the fields of the validation rule agree and the similarity clears `tau`, the policy writes the stored answer after a lookup of about 0.4 ms. The baseline got 134 of the 140 reused tasks right and reuse got all 140, which gives 6 of the 13 extra tasks that task-level reuse solved (401 against 388); the other 7 are chance in the agent runs. The most suitable tasks are templated requests whose decisive fields are explicit. The corpus questions name a scenario and a question kind, and 12 to 15 of the 40 tasks of each corpus dataset were reused, worth 30% to 39% of the baseline tokens. Docs and citations requests carry explicit identifiers (a CHG, INC, POL or REL number; title, authors, venue, year and mostly a DOI) and reused 23 of 60 and 25 of 70 tasks. Free-text tickets are the weakest case, with 18 of 70 reused (26%, against 30% to 38% elsewhere): the resolution depends on the two sentences of the body that describe the cause, so history pushes their threshold to 0.725. Indexing verified answers also matters, since 35 of the 140 reuses matched an earlier incoming task.

**2. Is high similarity sufficient for safe reuse?** No. In docs, task T-b7e9fb and history record H-4b14df have an estimated similarity of 0.93. Both are the POL-534 policy for the Payroll Gateway and are word for word the same, except that the task says twice that the policy is in draft where the record says it is in force, so the record's answer, policy-final, fails. In citations, T-25850c and H-cf84aa reach 0.89 and differ only in the first author's surname, Petrv for Petrov, so the task's verdict must also list `authors` among its wrong fields. The validation key rejects both, since the status class and the author list differ. In tickets, T-27a7f4 and H-f01546 share product, error code and platform and reach 0.59; the one different paragraph describes a power failure during a sync in the task and a self-update on Tuesday in the record, and the resolutions differ. This pair passes the key, so the calibrated threshold of 0.725 is what stops it. When several candidates pass both checks, their answers must also agree. Over the 420 tasks the most similar LSH candidate carried a wrong answer for 224 tasks, and the policy reused none of those answers.

**3. How does the similarity threshold affect performance?** Lowering the threshold finds more duplicates and more hard negatives at the same time (Figure 4). Plain nearest-neighbour reuse is wrong for 112 of 229 reuses at 0.5 and still for 6 of 86 at 0.9. With the key check only tickets and bugs need a threshold at all. On the other six datasets the key alone blocks every hard negative, and a threshold of 0.1 would add 14 correct reuses to the 102 found at 0.3. On tickets the policy reuses 9 wrong answers at 0.5 and still one at 0.6, where T-e08d15 takes the resolution of H-1b89e5 at an estimated 0.602, although the exact Jaccard similarity of the two texts is 0.548. The nine wrong tickets reuses at 0.5 lie at an exact similarity of 0.52 to 0.55, where one 128-function estimate has a standard deviation of 0.044, so a fixed threshold needs about two standard deviations of margin, around 0.65. Bugs is safe from 0.3 in this sweep.

So a single threshold for every dataset gives up either correctness or reuse. A threshold of 0.3 everywhere gives 154 reuses with 10 wrong, 0.6 gives 122 with 1 wrong and 0.7 gives 109 with none, while the per-dataset thresholds calibrated from history give 140 with none wrong. The calibrated 0.725 on tickets comes from history conflicts estimated at 0.625 whose exact similarity is also about 0.55, so its margin of 0.1 covers the same estimation error. The calibration uses only `history.jsonl` and the verifier's pass or fail, so it runs unchanged on held-out tasks.

**4. When is content-level reuse effective?** When one trajectory reads the same or nearly the same text more than once. Code is the clearest case: the replay saves 3.3% of its input tokens, 71% of it from `read_file` diffs between sibling adapters. In the combined run of code T-b372a7, `adapters/checkout.py` (4,222 characters) came back as a 1,124-character diff against `adapters/adyen.py`. The diff lists the name, base URL, signature header, decline codes and class name, plus the two lines where checkout behaves differently: its retry loop is one attempt shorter and its `refund` sends the idempotency key.

Logs shows why the same approach can give almost nothing. With digits masked, the log files of a scenario look alike (median similarity to the closest earlier file 1.0), but nearly every line differs in its timestamp or request id, so the exact diff is about twice as long as the file and the layer returns the original. Almost all of the 1.5% that logs saves in the replay comes from 20 exact repeats, 18 of them greps for ERROR over `logs/`. The record datasets contain no near-duplicate files, so tickets and citations save under 0.7%, and 73% of the docs saving and 91% of the bugs saving come from seven trajectories that read `history.jsonl` a second time. Counted inside the live runs, the layer removed 4.2% of the input tokens, 72% of it in ten second reads of `history.jsonl`, so both counts mostly measure how often an agent happens to read a large file twice.

**5. Which optimization is more valuable?** Task-level reuse, by a wide margin. It cut tool calls and requests by 29%, input tokens by 32% and time by 24% and raised accuracy from 92.4% to 95.5%. Content-level reuse only shortens tool outputs, so it removes no tool calls or requests by itself, and it saves 1.7% of the input tokens in the replay and 4.2% inside its own runs (0.8% and 1.2% without the `history.jsonl` re-reads). Little of its 7% time saving comes from the layer: under the fit of question 6, all of its fewer requests and tokens together account for 2.3 of the 7.3 minutes, and the 120 corpus tasks of the faster session saved 1.8 minutes more than the fit predicts for them. Folding did not hurt accuracy. Content-level reuse is within one task of the baseline (387 against 388), and its 124 runs with at least one fold passed 112 times against 111 for the baseline on the same tasks. On citations it passed 8 fewer tasks than the baseline, 6 of them in the 49 runs where the layer folded nothing (35 against 41). Both overheads, 4 ms and 24 ms per task, are negligible next to an agent run of 13.9 s.

Combining the two did not add up: the combined configuration used 53.0M tokens against 46.1M for task-level reuse alone, and three runaway combined runs in which the layer folded nothing, bugs T-704c01 (5.4M), tickets T-1cb53c (4.0M) and citations T-24c0bd (2.5M), used 11.0M more tokens than the same tasks with task-level reuse alone, more than the whole gap of 6.9M.

I would use task-level reuse whenever requests repeat and a verifier and a decisive key exist, as in bug triage, citation checks or templated operations questions. Content-level reuse suits workloads where every task is new and there is no verified history, but trajectories read families of near-identical files such as per-environment configs or copy-pasted adapters. Its diffs are exact, so it needs no verifier.

**6. What are the limitations of MinHash for agents?**

- *Paraphrases share few shingles.* In code, T-2a8e03 asks which adapter breaks rule R2 on `TIMEOUT_S` and H-cb9c70 asks which cashier adapter drifted on its HTTP timeout constant. Both concern scenario SDK-1722 and have the same answer, at an estimated similarity of 0.10. In the task-level stream, 30 agent tasks had an indexed record with the same key and a passing answer below their dataset's threshold.
- *The key has to be written for each dataset.* One token can change the answer at similarity 0.9 (question 2), so MinHash needs a validation key derived from a published decision rule. The calibration also learns only from same-key conflicts in the history, so a short history without one leaves `tau` at 0.3, where tickets gets wrong reuses (question 3).
- *Overhead can exceed savings.* A least-squares fit of agent time on requests, input tokens and output tokens over the 1,400 agent runs puts 1,000 input tokens at 17 ms, so the 654 tokens the layer saves per tickets task in the replay are worth 11 ms. Inside the experiment, where the streams share one Python process, the layer spent 29 ms per tickets task (6 ms when replaying alone). Task-level reuse sits at the other end: 1.7 s of MinHash time in total saved 23.7 minutes.

One improvement is to keep the validation key as a blocking key and compare the free text inside each block with a sentence embedding, keeping the agreement rule and calibrating the threshold the same way. The SDK-1722 pair shares the key, the subject line and the meaning but few other shingles (exact Jaccard 0.13), so an embedding could recover such paraphrases, as long as same-key hard negatives such as the tickets pair in question 2 stay below the new threshold.

# 5. Reproducing

```bash
git clone https://github.com/haotianxia/a2-minhash && cd a2-minhash
python3 -m venv .venv && . .venv/bin/activate   # Python 3.13
pip install -r requirements.txt   # numpy, matplotlib and Spartacus at commit 711db62
export GEMINI_API_KEY=...
python kickoff.py run --workers 16 --out out2
python kickoff.py summary --out out2
python kickoff.py sweep --out out2
python kickoff.py replay --out out2
python kickoff.py trajectories --out out2
python kickoff.py plots --out out2
```

`out/` holds the reported run: rows in `out/results/`, tables in `out/tables.md`, figures in `out/plots/`, and in `out/runs.tar.gz` every task's working directory with its harness session. After `tar xzf out/runs.tar.gz -C out`, the five commands after `run`, without `--out`, recompute the tables and figures without model calls. A held-out dataset folder, with its `env/`, `hidden/` and a copy of its `verify.py`, goes under `assignment2/datasets/` with any name and runs with

```bash
python kickoff.py run --datasets assignment2/datasets/<folder> --out <new folder>
```

MinHash uses seed 1115 and banding has no randomness, so the same transcripts always give the same reuse decisions, folds, sweep and replays apart from measured times; the agent samples at temperature 0.4 without a seed.
