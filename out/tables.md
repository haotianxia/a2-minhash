Table A. All 8 datasets together.

| configuration | accuracy | reused (wrong) | tool calls | model requests | input tokens | completion time | MinHash overhead |
|---|---|---|---|---|---|---|---|
| baseline | 222/240 (92.5%) | 0 (0) | 2598 (+0%) | 2838 (+0%) | 34.2M (+0%) | 51 min (+0%) | 0.00 s |
| task-level reuse | 232/240 (96.7%) | 82 (0) | 1839 (-29%) | 1997 (-30%) | 24.4M (-29%) | 39 min (-24%) | 1.49 s |
| content-level reuse | 222/240 (92.5%) | 0 (0) | 2643 (+2%) | 2881 (+2%) | 32.9M (-4%) | 50 min (-3%) | 7.06 s |
| both | 228/240 (95.0%) | 81 (0) | 1942 (-25%) | 2099 (-26%) | 24.7M (-28%) | 39 min (-24%) | 4.19 s |

Table B. Per dataset, per task averages (accuracy over all tasks).

| dataset | configuration | accuracy | reused | tool calls | requests | input tokens (K) | time (s) | overhead (ms) |
|---|---|---|---|---|---|---|---|---|
| tickets | baseline | 25/30 | 0 | 9.3 | 10.3 | 92.4 | 15.3 | 0.0 |
| tickets | task-level reuse | 29/30 | 7 | 10.4 | 11.1 | 153.1 | 15.1 | 10.3 |
| tickets | content-level reuse | 29/30 | 0 | 10.9 | 11.9 | 89.7 | 17.5 | 50.0 |
| tickets | both | 28/30 | 7 | 12.1 | 12.9 | 205.6 | 17.6 | 23.8 |
| bugs | baseline | 28/30 | 0 | 12.2 | 13.2 | 305.3 | 16.7 | 0.0 |
| bugs | task-level reuse | 29/30 | 10 | 7.9 | 8.6 | 125.9 | 12.9 | 10.6 |
| bugs | content-level reuse | 28/30 | 0 | 12.7 | 13.7 | 299.0 | 17.2 | 40.7 |
| bugs | both | 28/30 | 10 | 8.2 | 8.9 | 173.3 | 12.0 | 37.1 |
| docs | baseline | 27/30 | 0 | 10.0 | 11.0 | 172.1 | 16.0 | 0.0 |
| docs | task-level reuse | 30/30 | 11 | 7.1 | 7.7 | 108.7 | 11.5 | 10.5 |
| docs | content-level reuse | 27/30 | 0 | 10.5 | 11.4 | 172.6 | 15.4 | 49.0 |
| docs | both | 29/30 | 11 | 6.6 | 7.3 | 103.2 | 11.4 | 23.4 |
| citations | baseline | 22/30 | 0 | 14.6 | 15.6 | 306.0 | 20.9 | 0.0 |
| citations | task-level reuse | 25/30 | 11 | 9.8 | 10.4 | 279.6 | 15.2 | 10.0 |
| citations | content-level reuse | 20/30 | 0 | 13.6 | 14.6 | 299.2 | 20.2 | 52.7 |
| citations | both | 23/30 | 10 | 10.7 | 11.3 | 177.9 | 16.8 | 22.0 |
| hosts | baseline | 30/30 | 0 | 9.6 | 10.6 | 54.8 | 7.7 | 0.0 |
| hosts | task-level reuse | 30/30 | 12 | 5.6 | 6.2 | 25.2 | 4.9 | 3.7 |
| hosts | content-level reuse | 30/30 | 0 | 9.2 | 10.2 | 54.3 | 6.7 | 8.3 |
| hosts | both | 30/30 | 12 | 5.3 | 5.9 | 28.0 | 3.9 | 6.8 |
| configs | baseline | 30/30 | 0 | 10.0 | 11.0 | 79.4 | 8.5 | 0.0 |
| configs | task-level reuse | 29/30 | 13 | 6.4 | 7.0 | 41.0 | 6.3 | 2.1 |
| configs | content-level reuse | 28/30 | 0 | 11.4 | 12.4 | 73.1 | 8.4 | 16.8 |
| configs | both | 30/30 | 13 | 6.4 | 7.0 | 40.3 | 4.8 | 7.8 |
| code | baseline | 30/30 | 0 | 10.4 | 11.4 | 60.6 | 8.2 | 0.0 |
| code | task-level reuse | 30/30 | 7 | 7.7 | 8.4 | 40.0 | 6.3 | 1.1 |
| code | content-level reuse | 30/30 | 0 | 9.1 | 10.1 | 53.5 | 6.5 | 10.8 |
| code | both | 30/30 | 7 | 8.1 | 8.9 | 47.5 | 5.8 | 10.4 |
| logs | baseline | 30/30 | 0 | 10.5 | 11.5 | 70.6 | 9.1 | 0.0 |
| logs | task-level reuse | 30/30 | 11 | 6.5 | 7.1 | 38.2 | 5.5 | 1.4 |
| logs | content-level reuse | 30/30 | 0 | 10.7 | 11.7 | 56.2 | 7.4 | 7.0 |
| logs | both | 30/30 | 11 | 7.2 | 7.8 | 48.9 | 5.2 | 8.3 |

Table C. Content layer inside the agent (content-level reuse only).

| dataset | tool outputs | parts folded | output chars in | output chars out | kept |
|---|---|---|---|---|---|
| tickets | 293 | 13 | 1169486 | 1140596 | 97.5% |
| bugs | 347 | 11 | 3222258 | 3206710 | 99.5% |
| docs | 281 | 35 | 3177870 | 2736579 | 86.1% |
| citations | 376 | 13 | 3418433 | 3408672 | 99.7% |
| hosts | 246 | 2 | 679255 | 677408 | 99.7% |
| configs | 309 | 51 | 776994 | 729561 | 93.9% |
| code | 236 | 14 | 714704 | 680995 | 95.3% |
| logs | 288 | 19 | 535318 | 494656 | 92.4% |

Table D. Paired view: the same task ids in a configuration and in the baseline.

| dataset | configuration | reused | baseline accuracy on reused | baseline tokens of reused (K) | agent tasks | accuracy config / baseline | tokens config / baseline (K) | median tokens config / baseline (K) |
|---|---|---|---|---|---|---|---|---|
| tickets | task-level reuse | 7 | 5/7 | 817 | 23 | 22 / 20 | 4594 / 1955 | 65.6 / 40.5 |
| tickets | content-level reuse | 0 | 0/0 | 0 | 30 | 29 / 25 | 2690 / 2772 | 59.2 / 54.2 |
| tickets | both | 7 | 5/7 | 817 | 23 | 21 / 20 | 6167 / 1955 | 51.8 / 40.5 |
| bugs | task-level reuse | 10 | 10/10 | 2841 | 20 | 19 / 18 | 3777 / 6320 | 150.6 / 160.7 |
| bugs | content-level reuse | 0 | 0/0 | 0 | 30 | 28 / 28 | 8970 / 9160 | 163.9 / 170.2 |
| bugs | both | 10 | 10/10 | 2841 | 20 | 18 / 18 | 5199 / 6320 | 156.0 / 160.7 |
| docs | task-level reuse | 11 | 10/11 | 1506 | 19 | 19 / 17 | 3262 / 3658 | 183.6 / 162.2 |
| docs | content-level reuse | 0 | 0/0 | 0 | 30 | 27 / 27 | 5178 / 5163 | 182.4 / 159.0 |
| docs | both | 11 | 10/11 | 1506 | 19 | 18 / 17 | 3096 / 3658 | 153.9 / 162.2 |
| citations | task-level reuse | 11 | 8/11 | 3517 | 19 | 14 / 14 | 8387 / 5664 | 272.9 / 275.7 |
| citations | content-level reuse | 0 | 0/0 | 0 | 30 | 20 / 22 | 8976 / 9181 | 278.3 / 281.5 |
| citations | both | 10 | 8/10 | 3206 | 20 | 13 / 14 | 5338 / 5975 | 256.0 / 282.3 |
| hosts | task-level reuse | 12 | 12/12 | 698 | 18 | 18 / 18 | 757 / 945 | 38.0 / 44.5 |
| hosts | content-level reuse | 0 | 0/0 | 0 | 30 | 30 / 30 | 1630 / 1643 | 55.0 / 50.7 |
| hosts | both | 12 | 12/12 | 698 | 18 | 18 / 18 | 841 / 945 | 34.8 / 44.5 |
| configs | task-level reuse | 13 | 13/13 | 675 | 17 | 16 / 17 | 1230 / 1705 | 32.5 / 58.7 |
| configs | content-level reuse | 0 | 0/0 | 0 | 30 | 28 / 30 | 2193 / 2381 | 58.3 / 52.6 |
| configs | both | 13 | 13/13 | 675 | 17 | 17 / 17 | 1208 / 1705 | 49.4 / 58.7 |
| code | task-level reuse | 7 | 7/7 | 407 | 23 | 23 / 23 | 1199 / 1413 | 46.2 / 50.1 |
| code | content-level reuse | 0 | 0/0 | 0 | 30 | 30 / 30 | 1604 / 1819 | 45.4 / 51.1 |
| code | both | 7 | 7/7 | 407 | 23 | 23 / 23 | 1425 / 1413 | 43.5 / 50.1 |
| logs | task-level reuse | 11 | 11/11 | 733 | 19 | 19 / 19 | 1146 / 1384 | 46.7 / 48.5 |
| logs | content-level reuse | 0 | 0/0 | 0 | 30 | 30 / 30 | 1687 / 2117 | 47.6 / 51.3 |
| logs | both | 11 | 11/11 | 733 | 19 | 19 / 19 | 1467 / 1384 | 69.6 / 48.5 |
