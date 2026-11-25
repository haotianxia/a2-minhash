Table A. All 8 datasets together.

| configuration | accuracy | reused (wrong) | tool calls | model requests | input tokens | completion time | MinHash overhead |
|---|---|---|---|---|---|---|---|
| baseline | 388/420 (92.4%) | 0 (0) | 4831 (+0%) | 5251 (+0%) | 67.3M (+0%) | 97 min (+0%) | 0.00 s |
| task-level reuse | 401/420 (95.5%) | 140 (0) | 3432 (-29%) | 3712 (-29%) | 46.1M (-32%) | 74 min (-24%) | 1.73 s |
| content-level reuse | 387/420 (92.1%) | 0 (0) | 4774 (-1%) | 5191 (-1%) | 64.1M (-5%) | 90 min (-7%) | 10.15 s |
| both | 396/420 (94.3%) | 140 (0) | 3632 (-25%) | 3910 (-26%) | 53.0M (-21%) | 74 min (-24%) | 6.43 s |

Table B. Per dataset, per task averages (accuracy over all tasks).

| dataset | configuration | accuracy | reused | tool calls | requests | input tokens (K) | time (s) | overhead (ms) |
|---|---|---|---|---|---|---|---|---|
| tickets | baseline | 61/70 | 0 | 10.0 | 11.0 | 94.6 | 14.4 | 0.0 |
| tickets | task-level reuse | 66/70 | 18 | 8.9 | 9.6 | 110.0 | 12.2 | 5.2 |
| tickets | content-level reuse | 68/70 | 0 | 10.3 | 11.3 | 87.9 | 14.7 | 29.0 |
| tickets | both | 67/70 | 18 | 9.9 | 10.6 | 131.3 | 13.9 | 16.0 |
| bugs | baseline | 54/60 | 0 | 13.7 | 14.7 | 346.3 | 18.3 | 0.0 |
| bugs | task-level reuse | 57/60 | 20 | 7.9 | 8.5 | 143.8 | 12.0 | 6.7 |
| bugs | content-level reuse | 55/60 | 0 | 13.1 | 14.1 | 297.5 | 16.9 | 31.4 |
| bugs | both | 55/60 | 21 | 9.8 | 10.5 | 261.7 | 14.2 | 26.7 |
| docs | baseline | 55/60 | 0 | 11.1 | 12.1 | 189.1 | 16.0 | 0.0 |
| docs | task-level reuse | 60/60 | 23 | 8.6 | 9.2 | 148.1 | 12.8 | 5.9 |
| docs | content-level reuse | 56/60 | 0 | 10.5 | 11.5 | 169.4 | 13.3 | 38.1 |
| docs | both | 59/60 | 23 | 7.5 | 8.1 | 117.9 | 10.9 | 21.0 |
| citations | baseline | 58/70 | 0 | 14.5 | 15.5 | 255.7 | 19.0 | 0.0 |
| citations | task-level reuse | 59/70 | 25 | 10.8 | 11.5 | 216.3 | 14.9 | 4.8 |
| citations | content-level reuse | 50/70 | 0 | 14.4 | 15.4 | 293.8 | 18.6 | 33.0 |
| citations | both | 55/70 | 24 | 11.5 | 12.1 | 206.3 | 15.7 | 17.0 |
| hosts | baseline | 40/40 | 0 | 9.9 | 10.9 | 57.9 | 9.1 | 0.0 |
| hosts | task-level reuse | 40/40 | 13 | 6.2 | 6.9 | 27.3 | 6.6 | 2.9 |
| hosts | content-level reuse | 40/40 | 0 | 9.2 | 10.2 | 50.2 | 8.2 | 7.7 |
| hosts | both | 40/40 | 13 | 6.1 | 6.8 | 32.9 | 5.9 | 6.7 |
| configs | baseline | 40/40 | 0 | 10.3 | 11.3 | 81.7 | 10.1 | 0.0 |
| configs | task-level reuse | 39/40 | 15 | 6.9 | 7.5 | 44.5 | 7.8 | 1.8 |
| configs | content-level reuse | 38/40 | 0 | 11.4 | 12.4 | 71.2 | 8.9 | 14.7 |
| configs | both | 40/40 | 15 | 7.2 | 7.8 | 45.6 | 6.1 | 8.1 |
| code | baseline | 40/40 | 0 | 9.9 | 10.9 | 56.3 | 8.3 | 0.0 |
| code | task-level reuse | 40/40 | 12 | 6.9 | 7.6 | 34.1 | 6.2 | 1.0 |
| code | content-level reuse | 40/40 | 0 | 9.3 | 10.3 | 53.4 | 7.1 | 10.8 |
| code | both | 40/40 | 12 | 7.2 | 7.9 | 41.7 | 5.1 | 9.0 |
| logs | baseline | 40/40 | 0 | 10.6 | 11.6 | 71.2 | 8.6 | 0.0 |
| logs | task-level reuse | 40/40 | 14 | 6.6 | 7.2 | 37.8 | 5.3 | 1.2 |
| logs | content-level reuse | 40/40 | 0 | 10.7 | 11.7 | 58.9 | 7.3 | 7.8 |
| logs | both | 40/40 | 14 | 7.0 | 7.7 | 44.0 | 4.9 | 7.7 |

Table C. Content layer inside the agent (content-level reuse only).

| dataset | tool outputs | parts folded | output chars in | output chars out | kept |
|---|---|---|---|---|---|
| tickets | 642 | 24 | 2948828 | 2901209 | 98.4% |
| bugs | 712 | 30 | 5972850 | 5817657 | 97.4% |
| docs | 558 | 55 | 5958745 | 5227614 | 87.7% |
| citations | 932 | 30 | 7237155 | 6974611 | 96.4% |
| hosts | 328 | 3 | 817558 | 814715 | 99.7% |
| configs | 415 | 74 | 965229 | 891737 | 92.4% |
| code | 324 | 15 | 935302 | 898265 | 96.0% |
| logs | 385 | 22 | 757401 | 684879 | 90.4% |

Table D. Paired view: the same task ids in a configuration and in the baseline.

| dataset | configuration | reused (history / earlier task) | baseline accuracy on reused | baseline tokens of reused (K) | share of baseline tokens | agent tasks | accuracy config / baseline | tokens config / baseline (K) | median tokens config / baseline (K) |
|---|---|---|---|---|---|---|---|---|---|
| tickets | task-level reuse | 18 (11 / 7) | 16/18 | 2006 | 30% | 52 | 48 / 45 | 7701 / 4616 | 56.0 / 53.1 |
| tickets | content-level reuse | 0 (0 / 0) | 0/0 | 0 | 0% | 70 | 68 / 61 | 6153 / 6622 | 60.4 / 55.3 |
| tickets | both | 18 (11 / 7) | 16/18 | 2006 | 30% | 52 | 49 / 45 | 9194 / 4616 | 58.3 / 53.1 |
| bugs | task-level reuse | 20 (14 / 6) | 20/20 | 4144 | 20% | 40 | 37 / 34 | 8628 / 16634 | 151.9 / 157.7 |
| bugs | content-level reuse | 0 (0 / 0) | 0/0 | 0 | 0% | 60 | 55 / 54 | 17852 / 20778 | 154.4 / 155.5 |
| bugs | both | 21 (14 / 7) | 20/21 | 4298 | 21% | 39 | 34 / 34 | 15702 / 16480 | 197.5 / 160.3 |
| docs | task-level reuse | 23 (18 / 5) | 22/23 | 3312 | 29% | 37 | 37 / 33 | 8888 / 8036 | 183.6 / 182.2 |
| docs | content-level reuse | 0 (0 / 0) | 0/0 | 0 | 0% | 60 | 56 / 55 | 10163 / 11348 | 153.9 / 144.3 |
| docs | both | 23 (18 / 5) | 22/23 | 3312 | 29% | 37 | 36 / 33 | 7074 / 8036 | 163.3 / 182.2 |
| citations | task-level reuse | 25 (20 / 5) | 22/25 | 6642 | 37% | 45 | 34 / 36 | 15138 / 11261 | 228.9 / 157.3 |
| citations | content-level reuse | 0 (0 / 0) | 0/0 | 0 | 0% | 70 | 50 / 58 | 20568 / 17902 | 273.8 / 241.2 |
| citations | both | 24 (20 / 4) | 22/24 | 6331 | 35% | 46 | 31 / 36 | 14439 / 11572 | 273.6 / 181.9 |
| hosts | task-level reuse | 13 (11 / 2) | 13/13 | 904 | 39% | 27 | 27 / 27 | 1091 / 1410 | 37.6 / 42.1 |
| hosts | content-level reuse | 0 (0 / 0) | 0/0 | 0 | 0% | 40 | 40 / 40 | 2008 / 2314 | 44.3 / 50.7 |
| hosts | both | 13 (11 / 2) | 13/13 | 904 | 39% | 27 | 27 / 27 | 1318 / 1410 | 37.8 / 42.1 |
| configs | task-level reuse | 15 (10 / 5) | 15/15 | 972 | 30% | 25 | 24 / 25 | 1782 / 2294 | 28.5 / 58.7 |
| configs | content-level reuse | 0 (0 / 0) | 0/0 | 0 | 0% | 40 | 38 / 40 | 2849 / 3267 | 57.6 / 58.6 |
| configs | both | 15 (10 / 5) | 15/15 | 972 | 30% | 25 | 25 / 25 | 1823 / 2294 | 49.4 / 58.7 |
| code | task-level reuse | 12 (10 / 2) | 12/12 | 679 | 30% | 28 | 28 / 28 | 1365 / 1573 | 35.0 / 47.7 |
| code | content-level reuse | 0 (0 / 0) | 0/0 | 0 | 0% | 40 | 40 / 40 | 2137 / 2252 | 42.2 / 49.5 |
| code | both | 12 (10 / 2) | 12/12 | 679 | 30% | 28 | 28 / 28 | 1668 / 1573 | 43.1 / 47.7 |
| logs | task-level reuse | 14 (11 / 3) | 14/14 | 1001 | 35% | 26 | 26 / 26 | 1512 / 1849 | 46.9 / 47.1 |
| logs | content-level reuse | 0 (0 / 0) | 0/0 | 0 | 0% | 40 | 40 / 40 | 2354 / 2849 | 47.6 / 52.7 |
| logs | both | 14 (11 / 3) | 14/14 | 1001 | 35% | 26 | 26 / 26 | 1760 / 1849 | 62.4 / 47.1 |
