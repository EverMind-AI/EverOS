# Search throughput

On a 4-core, 8 GB server holding 168,821 memories, `hybrid` search sustains
**14 to 16 requests per second on 1.8.0, against fewer than 4 on 1.7.1**, in about a sixth of
the memory; `keyword` search sustains **120 to 140, against 70**.
Measured on main at `c5d9ace9`, ahead of the 1.8.0 release, with the model providers
stubbed so that only EverOS is measured. Everything needed to reproduce it is in
[`throughput/`](throughput/); [Reproduce](#reproduce) has the steps.

> Also available in Chinese: [throughput.zh.md](throughput.zh.md).

## Results

`hybrid` ran two rounds and `keyword` three; a rate counts only when every round kept
up at it. `keyword` was sent by four load-generator processes, after a single one froze at
160 and 180 requests/s in a first run (see [Limits](#limits)). `glibc` and `jemalloc` are the same
1.8.0 build: on Linux the 1.8.0 CLI preloads
jemalloc when `libjemalloc2` is installed, and runs on glibc otherwise.

**Highest sustained rate**, with latency, CPU and peak memory at that rate (the worst
round), and the highest rate any single round kept up at:

| Search | Version | Sustained requests/s | p50 | p95 | CPU (cores) | Peak memory | Best round |
|---|---|---|---|---|---|---|---|
| `hybrid` | 1.7.1 | < 4 | – | – | – | – | 4 |
| `hybrid` | 1.8.0, glibc | 14 | 1.06 s | 3.41 s | 2.50 | 681 MB | 14 |
| `hybrid` | 1.8.0, jemalloc | 16 | 0.72 s | 2.48 s | 2.07 | 595 MB | 16 |
| `keyword` | 1.7.1 | 70 | 75 ms | 345 ms | 2.35 | 597 MB | 70 |
| `keyword` | 1.8.0, glibc | 120 | 25 ms | 197 ms | 1.75 | 700 MB | 150 |
| `keyword` | 1.8.0, jemalloc | 140 | 24 ms | 241 ms | 1.56 | 563 MB | 170 |

1.7.1 did not sustain `hybrid`'s lowest rate: at 4/s its second round completed 3.45
requests per second (see the same-load table below).

Near the top the rounds disagree. A step there can fall behind after a stall of a few
seconds that the little headroom left cannot clear within the step: in the first
`jemalloc` round 130 fell behind while 140 and 150 kept up, and the third round kept up
to 170. What stalls the server was not found; its log shows nothing at those moments.

**At the same load**: `hybrid` at 4/s, the lowest rate tried (1.7.1 kept up there in one
round of two), and `keyword` at 60/s from the first run (one load generator, 30 s
steps, two rounds), where all three kept up:

| Search, rate | Version | p50 | p95 | CPU (cores) | Peak memory |
|---|---|---|---|---|---|
| `hybrid`, 4/s | 1.7.1 | 4.50 s | 8.70 s | 3.45 | 4,093 MB |
| | 1.8.0, glibc | 0.48 s | 2.21 s | 0.68 | 673 MB |
| | 1.8.0, jemalloc | 0.52 s | 2.17 s | 0.71 | 518 MB |
| `keyword`, 60/s | 1.7.1 | 27 ms | 50 ms | 1.91 | 590 MB |
| | 1.8.0, glibc | 16 ms | 25 ms | 0.86 | 639 MB |
| | 1.8.0, jemalloc | 12 ms | 20 ms | 0.74 | 455 MB |

Behind the `hybrid` rows is the vector index. 1.7.1's IVF_FLAT index of the atomic
facts (0.66 GB) did not fit the index cache, so each query read about 0.5 GB of it
again: measured separately on this store, 449 ms and 1.1 GB of transient memory per
query. 1.8.0 builds
IVF_RQ indexes (24 MB for the same facts): 35 to 50 ms and 59 to 130 MB. A store
written by 1.7.1 is converted on its first maintenance pass after the upgrade.

`jemalloc`'s 16 `hybrid` searches per second are EverOS's own limit, not the query
embedding's. A `hybrid` search waits for one query embedding, and EverOS runs at most 16
at once (`[embedding] query_max_concurrent`); at 20/s, with the stub's mean latency of
0.78 s, about 15.6 would be in flight, close to that cap. A control run on 2026-11-25
raised the cap to 64: up to 60 embedding requests were then open at once, yet `jemalloc`
still kept up at 16/s and fell behind at 20/s in both rounds, completing about 16 per
second with about 2.3 of its 4 cores in use. Which part of EverOS sets that limit was
not found.

<details>
<summary>Control run: query embedding cap 16 and 64, <code>jemalloc</code></summary>

| Method | Server | QPS | Round | OK | Done/s | p50 | p95 | CPU | Memory | Kept up |
|---|---|---|---|---|---|---|---|---|---|---|
| hybrid | 1.8.0-jemalloc-qmc16 | 16 | r1 | 960/960 | 15.8 | 728 ms | 2802 ms | 2.14 | 536 MB | yes |
| hybrid | 1.8.0-jemalloc-qmc16 | 20 | r1 | 1200/1200 | 16.8 | 6100 ms | 9933 ms | 2.37 | 572 MB | no |
| hybrid | 1.8.0-jemalloc-qmc16 | 24 | r1 | 1440/1440 | 16.2 | 14019 ms | 24387 ms | 2.29 | 594 MB | no |
| hybrid | 1.8.0-jemalloc-qmc64 | 16 | r1 | 960/960 | 15.7 | 711 ms | 2760 ms | 2.06 | 537 MB | yes |
| hybrid | 1.8.0-jemalloc-qmc64 | 16 | r2 | 960/960 | 15.8 | 702 ms | 2898 ms | 2.07 | 542 MB | yes |
| hybrid | 1.8.0-jemalloc-qmc64 | 20 | r1 | 1200/1200 | 15.8 | 7081 ms | 15346 ms | 2.31 | 629 MB | no |
| hybrid | 1.8.0-jemalloc-qmc64 | 20 | r2 | 1200/1200 | 15.9 | 5846 ms | 15041 ms | 2.31 | 620 MB | no |
| hybrid | 1.8.0-jemalloc-qmc64 | 24 | r1 | 1440/1440 | 15.3 | 17685 ms | 29169 ms | 2.24 | 624 MB | no |
| hybrid | 1.8.0-jemalloc-qmc64 | 24 | r2 | 1440/1440 | 14.7 | 19242 ms | 32716 ms | 2.22 | 646 MB | no |

</details>

<details>
<summary>Every step of every round</summary>

| Method | Server | QPS | Round | OK | Done/s | p50 | p95 | CPU | Memory | Kept up |
|---|---|---|---|---|---|---|---|---|---|---|
| hybrid | 1.8.0-glibc | 4 | r1 | 240/240 | 4 | 460 ms | 2206 ms | 0.64 | 660 MB | yes |
| hybrid | 1.8.0-glibc | 4 | r2 | 240/240 | 4 | 483 ms | 2100 ms | 0.68 | 673 MB | yes |
| hybrid | 1.8.0-glibc | 5 | r1 | 300/300 | 4.9 | 577 ms | 3391 ms | 0.81 | 660 MB | yes |
| hybrid | 1.8.0-glibc | 5 | r2 | 300/300 | 4.9 | 578 ms | 3181 ms | 0.8 | 674 MB | yes |
| hybrid | 1.8.0-glibc | 6 | r1 | 360/360 | 5.9 | 549 ms | 2421 ms | 0.96 | 662 MB | yes |
| hybrid | 1.8.0-glibc | 6 | r2 | 360/360 | 5.9 | 533 ms | 2326 ms | 0.96 | 675 MB | yes |
| hybrid | 1.8.0-glibc | 8 | r1 | 480/480 | 7.8 | 589 ms | 2624 ms | 1.29 | 662 MB | yes |
| hybrid | 1.8.0-glibc | 8 | r2 | 480/480 | 7.8 | 621 ms | 2662 ms | 1.32 | 676 MB | yes |
| hybrid | 1.8.0-glibc | 10 | r1 | 600/600 | 9.8 | 626 ms | 2771 ms | 1.58 | 663 MB | yes |
| hybrid | 1.8.0-glibc | 10 | r2 | 600/600 | 9.8 | 668 ms | 2741 ms | 1.63 | 676 MB | yes |
| hybrid | 1.8.0-glibc | 12 | r1 | 720/720 | 11.6 | 722 ms | 3235 ms | 1.95 | 664 MB | yes |
| hybrid | 1.8.0-glibc | 12 | r2 | 720/720 | 11.6 | 788 ms | 3362 ms | 2.01 | 679 MB | yes |
| hybrid | 1.8.0-glibc | 14 | r1 | 840/840 | 13.7 | 1047 ms | 3413 ms | 2.43 | 670 MB | yes |
| hybrid | 1.8.0-glibc | 14 | r2 | 840/840 | 13.6 | 1061 ms | 3256 ms | 2.5 | 681 MB | yes |
| hybrid | 1.8.0-glibc | 16 | r1 | 960/960 | 11.8 | 14572 ms | 16521 ms | 2.69 | 734 MB | no |
| hybrid | 1.8.0-glibc | 16 | r2 | 960/960 | 11.7 | 10052 ms | 17330 ms | 2.64 | 786 MB | no |
| hybrid | 1.8.0-glibc | 20 | r1 | 1200/1200 | 14.2 | 12361 ms | 21976 ms | 2.72 | 732 MB | no |
| hybrid | 1.8.0-glibc | 20 | r2 | 1200/1200 | 11.8 | 21410 ms | 36113 ms | 2.45 | 786 MB | no |
| hybrid | 1.8.0-jemalloc | 4 | r1 | 240/240 | 4 | 463 ms | 2088 ms | 0.59 | 518 MB | yes |
| hybrid | 1.8.0-jemalloc | 4 | r2 | 240/240 | 4 | 515 ms | 2173 ms | 0.71 | 513 MB | yes |
| hybrid | 1.8.0-jemalloc | 5 | r1 | 300/300 | 4.9 | 575 ms | 3436 ms | 0.81 | 529 MB | yes |
| hybrid | 1.8.0-jemalloc | 5 | r2 | 300/300 | 4.9 | 587 ms | 3185 ms | 0.78 | 527 MB | yes |
| hybrid | 1.8.0-jemalloc | 6 | r1 | 360/360 | 5.9 | 525 ms | 2249 ms | 0.87 | 525 MB | yes |
| hybrid | 1.8.0-jemalloc | 6 | r2 | 360/360 | 5.9 | 552 ms | 2247 ms | 0.89 | 520 MB | yes |
| hybrid | 1.8.0-jemalloc | 8 | r1 | 480/480 | 7.9 | 562 ms | 2580 ms | 1.12 | 524 MB | yes |
| hybrid | 1.8.0-jemalloc | 8 | r2 | 480/480 | 7.9 | 577 ms | 2613 ms | 1.09 | 527 MB | yes |
| hybrid | 1.8.0-jemalloc | 10 | r1 | 600/600 | 9.8 | 559 ms | 2742 ms | 1.29 | 532 MB | yes |
| hybrid | 1.8.0-jemalloc | 10 | r2 | 600/600 | 9.8 | 600 ms | 2833 ms | 1.34 | 535 MB | yes |
| hybrid | 1.8.0-jemalloc | 12 | r1 | 720/720 | 11.7 | 577 ms | 3158 ms | 1.54 | 543 MB | yes |
| hybrid | 1.8.0-jemalloc | 12 | r2 | 720/720 | 11.6 | 664 ms | 3198 ms | 1.66 | 536 MB | yes |
| hybrid | 1.8.0-jemalloc | 14 | r1 | 840/840 | 13.8 | 606 ms | 2906 ms | 1.77 | 540 MB | yes |
| hybrid | 1.8.0-jemalloc | 14 | r2 | 840/840 | 13.8 | 641 ms | 2956 ms | 1.83 | 544 MB | yes |
| hybrid | 1.8.0-jemalloc | 16 | r1 | 960/960 | 15.7 | 685 ms | 2415 ms | 2.01 | 595 MB | yes |
| hybrid | 1.8.0-jemalloc | 16 | r2 | 960/960 | 15.7 | 719 ms | 2484 ms | 2.07 | 548 MB | yes |
| hybrid | 1.8.0-jemalloc | 20 | r1 | 1200/1200 | 18 | 3496 ms | 6365 ms | 2.35 | 564 MB | no |
| hybrid | 1.8.0-jemalloc | 20 | r2 | 1200/1200 | 17.8 | 4237 ms | 7333 ms | 2.36 | 562 MB | no |
| hybrid | 1.8.0-jemalloc | 24 | r1 | 1440/1440 | 17 | 11659 ms | 21342 ms | 2.27 | 583 MB | no |
| hybrid | 1.8.0-jemalloc | 24 | r2 | 1440/1440 | 16.2 | 13432 ms | 24626 ms | 2.31 | 595 MB | no |
| hybrid | v1.7.1 | 4 | r1 | 240/240 | 3.7 | 4215 ms | 6358 ms | 3.45 | 2830 MB | yes |
| hybrid | v1.7.1 | 4 | r2 | 240/240 | 3.4 | 4503 ms | 8698 ms | 3.31 | 4093 MB | no |
| hybrid | v1.7.1 | 5 | r1 | 300/300 | 3.9 | 7953 ms | 14582 ms | 3.32 | 4379 MB | no |
| hybrid | v1.7.1 | 5 | r2 | 300/300 | 3.2 | 18934 ms | 23948 ms | 3.11 | 3644 MB | no |
| hybrid | v1.7.1 | 6 | r1 | 360/360 | 4.5 | 11031 ms | 16040 ms | 3.46 | 3882 MB | no |
| hybrid | v1.7.1 | 6 | r2 | 360/360 | 4.4 | 12641 ms | 17468 ms | 3.43 | 3533 MB | no |
| hybrid | v1.7.1 | 8 | r2 | 480/480 | 3.7 | 30924 ms | 40560 ms | 3.5 | 4081 MB | no |
| hybrid | v1.7.1 | 10 | r2 | 575/600 | 3 | 52015 ms | 59110 ms | 3.45 | 4254 MB | no |
| keyword | 1.8.0-glibc | 100 | r1 | 6000/6000 | 100 | 23 ms | 34 ms | 1.45 | 613 MB | yes |
| keyword | 1.8.0-glibc | 100 | r2 | 6000/6000 | 100 | 28 ms | 73 ms | 1.55 | 681 MB | yes |
| keyword | 1.8.0-glibc | 100 | r3 | 6000/6000 | 100 | 25 ms | 111 ms | 1.48 | 699 MB | yes |
| keyword | 1.8.0-glibc | 110 | r1 | 6600/6600 | 110 | 24 ms | 56 ms | 1.59 | 613 MB | yes |
| keyword | 1.8.0-glibc | 110 | r2 | 6600/6600 | 109.9 | 22 ms | 39 ms | 1.53 | 681 MB | yes |
| keyword | 1.8.0-glibc | 110 | r3 | 6600/6600 | 110 | 26 ms | 122 ms | 1.62 | 700 MB | yes |
| keyword | 1.8.0-glibc | 120 | r1 | 7200/7200 | 120 | 22 ms | 117 ms | 1.71 | 613 MB | yes |
| keyword | 1.8.0-glibc | 120 | r2 | 7200/7200 | 120 | 18 ms | 33 ms | 1.55 | 681 MB | yes |
| keyword | 1.8.0-glibc | 120 | r3 | 7200/7200 | 120 | 25 ms | 197 ms | 1.75 | 700 MB | yes |
| keyword | 1.8.0-glibc | 130 | r1 | 7800/7800 | 130 | 35 ms | 221 ms | 1.98 | 614 MB | yes |
| keyword | 1.8.0-glibc | 130 | r2 | 7800/7800 | 129.9 | 46 ms | 351 ms | 1.99 | 684 MB | yes |
| keyword | 1.8.0-glibc | 130 | r3 | 7800/7800 | 99.7 | 5485 ms | 19056 ms | 2.05 | 780 MB | no |
| keyword | 1.8.0-glibc | 140 | r1 | 8400/8400 | 139.2 | 40 ms | 263 ms | 2.12 | 614 MB | yes |
| keyword | 1.8.0-glibc | 140 | r2 | 8400/8400 | 102.4 | 7889 ms | 22638 ms | 2.04 | 780 MB | no |
| keyword | 1.8.0-glibc | 140 | r3 | 8400/8400 | 99.7 | 8157 ms | 27920 ms | 2.03 | 789 MB | no |
| keyword | 1.8.0-glibc | 150 | r1 | 9000/9000 | 149.9 | 106 ms | 354 ms | 2.34 | 615 MB | yes |
| keyword | 1.8.0-glibc | 150 | r2 | 9000/9000 | 103.5 | 9037 ms | 29356 ms | 2.03 | 782 MB | no |
| keyword | 1.8.0-glibc | 160 | r1 | 9600/9600 | 107.1 | 9800 ms | 32184 ms | 2.1 | 720 MB | no |
| keyword | 1.8.0-glibc | 170 | r1 | 10200/10200 | 101.4 | 12189 ms | 41712 ms | 2.06 | 726 MB | no |
| keyword | 1.8.0-jemalloc | 100 | r1 | 6000/6000 | 100 | 14 ms | 24 ms | 1.07 | 464 MB | yes |
| keyword | 1.8.0-jemalloc | 100 | r2 | 6000/6000 | 100 | 18 ms | 26 ms | 1.15 | 466 MB | yes |
| keyword | 1.8.0-jemalloc | 100 | r3 | 6000/6000 | 100 | 17 ms | 27 ms | 1.21 | 461 MB | yes |
| keyword | 1.8.0-jemalloc | 110 | r1 | 6600/6600 | 110 | 14 ms | 24 ms | 1.17 | 466 MB | yes |
| keyword | 1.8.0-jemalloc | 110 | r2 | 6600/6600 | 110 | 20 ms | 31 ms | 1.19 | 473 MB | yes |
| keyword | 1.8.0-jemalloc | 110 | r3 | 6600/6600 | 110 | 17 ms | 307 ms | 1.34 | 473 MB | yes |
| keyword | 1.8.0-jemalloc | 120 | r1 | 7200/7200 | 117.8 | 21 ms | 739 ms | 1.3 | 479 MB | yes |
| keyword | 1.8.0-jemalloc | 120 | r2 | 7200/7200 | 120 | 19 ms | 31 ms | 1.31 | 468 MB | yes |
| keyword | 1.8.0-jemalloc | 120 | r3 | 7200/7200 | 120 | 18 ms | 58 ms | 1.29 | 472 MB | yes |
| keyword | 1.8.0-jemalloc | 130 | r1 | 7800/7800 | 97.4 | 7261 ms | 19846 ms | 1.49 | 607 MB | no |
| keyword | 1.8.0-jemalloc | 130 | r2 | 7800/7800 | 130 | 17 ms | 36 ms | 1.36 | 471 MB | yes |
| keyword | 1.8.0-jemalloc | 130 | r3 | 7800/7800 | 130 | 18 ms | 70 ms | 1.39 | 470 MB | yes |
| keyword | 1.8.0-jemalloc | 140 | r1 | 8400/8400 | 139.9 | 21 ms | 241 ms | 1.55 | 563 MB | yes |
| keyword | 1.8.0-jemalloc | 140 | r2 | 8400/8400 | 140 | 24 ms | 219 ms | 1.56 | 473 MB | yes |
| keyword | 1.8.0-jemalloc | 140 | r3 | 8400/8400 | 140 | 19 ms | 233 ms | 1.49 | 471 MB | yes |
| keyword | 1.8.0-jemalloc | 150 | r1 | 9000/9000 | 149.9 | 31 ms | 316 ms | 1.66 | 562 MB | yes |
| keyword | 1.8.0-jemalloc | 150 | r2 | 9000/9000 | 130.2 | 327 ms | 8754 ms | 1.69 | 544 MB | no |
| keyword | 1.8.0-jemalloc | 150 | r3 | 9000/9000 | 149.9 | 18 ms | 195 ms | 1.58 | 470 MB | yes |
| keyword | 1.8.0-jemalloc | 160 | r1 | 9600/9600 | 102.2 | 10378 ms | 35900 ms | 1.54 | 606 MB | no |
| keyword | 1.8.0-jemalloc | 160 | r2 | 9600/9600 | 97.2 | 10287 ms | 42540 ms | 1.63 | 601 MB | no |
| keyword | 1.8.0-jemalloc | 160 | r3 | 9600/9600 | 160 | 22 ms | 380 ms | 1.68 | 479 MB | yes |
| keyword | 1.8.0-jemalloc | 170 | r3 | 10200/10200 | 169.5 | 74 ms | 347 ms | 1.81 | 482 MB | yes |
| keyword | 1.8.0-jemalloc | 180 | r3 | 10791/10800 | 103.9 | 12592 ms | 42714 ms | 1.63 | 588 MB | no |
| keyword | v1.7.1 | 40 | r1 | 2400/2400 | 40 | 43 ms | 60 ms | 1.25 | 596 MB | yes |
| keyword | v1.7.1 | 40 | r2 | 2400/2400 | 40 | 52 ms | 69 ms | 1.29 | 530 MB | yes |
| keyword | v1.7.1 | 40 | r3 | 2400/2400 | 40 | 52 ms | 66 ms | 1.29 | 546 MB | yes |
| keyword | v1.7.1 | 50 | r1 | 3000/3000 | 50 | 55 ms | 67 ms | 1.59 | 596 MB | yes |
| keyword | v1.7.1 | 50 | r2 | 3000/3000 | 50 | 54 ms | 71 ms | 1.63 | 530 MB | yes |
| keyword | v1.7.1 | 50 | r3 | 3000/3000 | 50 | 54 ms | 69 ms | 1.65 | 546 MB | yes |
| keyword | v1.7.1 | 60 | r1 | 3600/3600 | 60 | 49 ms | 67 ms | 1.86 | 596 MB | yes |
| keyword | v1.7.1 | 60 | r2 | 3600/3600 | 59.7 | 52 ms | 108 ms | 1.92 | 530 MB | yes |
| keyword | v1.7.1 | 60 | r3 | 3600/3600 | 59.9 | 53 ms | 123 ms | 1.98 | 546 MB | yes |
| keyword | v1.7.1 | 70 | r1 | 4200/4200 | 70 | 56 ms | 226 ms | 2.25 | 597 MB | yes |
| keyword | v1.7.1 | 70 | r2 | 4200/4200 | 70 | 36 ms | 118 ms | 2.23 | 586 MB | yes |
| keyword | v1.7.1 | 70 | r3 | 4200/4200 | 69.5 | 75 ms | 345 ms | 2.35 | 547 MB | yes |
| keyword | v1.7.1 | 80 | r1 | 4800/4800 | 78.8 | 308 ms | 1065 ms | 2.67 | 600 MB | no |
| keyword | v1.7.1 | 80 | r2 | 4800/4800 | 69.4 | 4111 ms | 8022 ms | 2.67 | 629 MB | no |
| keyword | v1.7.1 | 80 | r3 | 4800/4800 | 74.7 | 590 ms | 3967 ms | 2.72 | 567 MB | no |
| keyword | v1.7.1 | 90 | r1 | 5400/5400 | 73.3 | 6157 ms | 11616 ms | 2.74 | 666 MB | no |
| keyword | v1.7.1 | 90 | r2 | 5400/5400 | 67.3 | 9534 ms | 16985 ms | 2.65 | 682 MB | no |
| keyword | v1.7.1 | 90 | r3 | 5399/5400 | 65.5 | 9638 ms | 20018 ms | 2.78 | 648 MB | no |

</details>

<details>
<summary>The first <code>keyword</code> run: one load generator, 30 s steps</summary>

| Method | Server | QPS | Round | OK | Done/s | p50 | p95 | CPU | Memory | Kept up |
|---|---|---|---|---|---|---|---|---|---|---|
| keyword | 1.8.0-glibc | 40 | r1 | 1200/1200 | 40 | 14 ms | 20 ms | 0.56 | 639 MB | yes |
| keyword | 1.8.0-glibc | 40 | r2 | 1200/1200 | 40 | 16 ms | 22 ms | 0.6 | 601 MB | yes |
| keyword | 1.8.0-glibc | 60 | r1 | 1800/1800 | 60 | 14 ms | 18 ms | 0.77 | 639 MB | yes |
| keyword | 1.8.0-glibc | 60 | r2 | 1800/1800 | 60 | 16 ms | 25 ms | 0.86 | 601 MB | yes |
| keyword | 1.8.0-glibc | 80 | r1 | 2400/2400 | 80 | 15 ms | 24 ms | 1.02 | 639 MB | yes |
| keyword | 1.8.0-glibc | 80 | r2 | 2400/2400 | 79.9 | 19 ms | 28 ms | 1.19 | 601 MB | yes |
| keyword | 1.8.0-glibc | 100 | r1 | 3000/3000 | 100 | 16 ms | 22 ms | 1.22 | 639 MB | yes |
| keyword | 1.8.0-glibc | 100 | r2 | 3000/3000 | 99.9 | 18 ms | 35 ms | 1.4 | 602 MB | yes |
| keyword | 1.8.0-glibc | 120 | r1 | 3600/3600 | 119.8 | 17 ms | 41 ms | 1.56 | 640 MB | yes |
| keyword | 1.8.0-glibc | 120 | r2 | 3600/3600 | 119.9 | 26 ms | 210 ms | 1.72 | 603 MB | yes |
| keyword | 1.8.0-glibc | 140 | r1 | 4197/4200 | 105.7 | 1420 ms | 12582 ms | 1.85 | 663 MB | no |
| keyword | 1.8.0-glibc | 140 | r2 | 4200/4200 | 117.9 | 1464 ms | 8989 ms | 1.95 | 638 MB | no |
| keyword | 1.8.0-glibc | 160 | r1 | 3255/4800 | 8.7 | 8213 ms | 445323 ms | 0.15 | 685 MB | no |
| keyword | 1.8.0-glibc | 160 | r2 | 3208/4800 | 7.8 | 9563 ms | 381327 ms | 0.13 | 711 MB | no |
| keyword | 1.8.0-jemalloc | 40 | r1 | 1200/1200 | 40 | 13 ms | 20 ms | 0.5 | 452 MB | yes |
| keyword | 1.8.0-jemalloc | 40 | r2 | 1200/1200 | 40 | 13 ms | 19 ms | 0.54 | 453 MB | yes |
| keyword | 1.8.0-jemalloc | 60 | r1 | 1800/1800 | 60 | 12 ms | 16 ms | 0.67 | 455 MB | yes |
| keyword | 1.8.0-jemalloc | 60 | r2 | 1800/1800 | 60 | 12 ms | 20 ms | 0.74 | 455 MB | yes |
| keyword | 1.8.0-jemalloc | 80 | r1 | 2400/2400 | 80 | 12 ms | 18 ms | 0.87 | 456 MB | yes |
| keyword | 1.8.0-jemalloc | 80 | r2 | 2400/2400 | 80 | 12 ms | 20 ms | 0.93 | 458 MB | yes |
| keyword | 1.8.0-jemalloc | 100 | r1 | 3000/3000 | 100 | 13 ms | 18 ms | 1.04 | 459 MB | yes |
| keyword | 1.8.0-jemalloc | 100 | r2 | 3000/3000 | 100 | 16 ms | 41 ms | 1.22 | 458 MB | yes |
| keyword | 1.8.0-jemalloc | 120 | r1 | 3600/3600 | 119.9 | 14 ms | 21 ms | 1.24 | 460 MB | yes |
| keyword | 1.8.0-jemalloc | 120 | r2 | 3600/3600 | 119.9 | 17 ms | 81 ms | 1.38 | 461 MB | yes |
| keyword | 1.8.0-jemalloc | 140 | r1 | 4200/4200 | 140 | 14 ms | 70 ms | 1.45 | 468 MB | yes |
| keyword | 1.8.0-jemalloc | 140 | r2 | 4200/4200 | 139.9 | 18 ms | 256 ms | 1.59 | 464 MB | yes |
| keyword | 1.8.0-jemalloc | 160 | r1 | 4800/4800 | 159.9 | 16 ms | 240 ms | 1.69 | 464 MB | yes |
| keyword | 1.8.0-jemalloc | 160 | r2 | 3455/4800 | 6.3 | 5531 ms | 707986 ms | 0.06 | 566 MB | no |
| keyword | 1.8.0-jemalloc | 180 | r1 | 3749/5400 | 2.9 | 7080 ms | 1361558 ms | 0.03 | 531 MB | no |
| keyword | 1.8.0-jemalloc | 180 | r2 | 2547/5400 | 0.9 | 263991 ms | 1461539 ms | 0.03 | 541 MB | no |
| keyword | v1.7.1 | 40 | r1 | 1200/1200 | 40 | 20 ms | 27 ms | 1.16 | 544 MB | yes |
| keyword | v1.7.1 | 40 | r2 | 1200/1200 | 40 | 22 ms | 31 ms | 1.27 | 590 MB | yes |
| keyword | v1.7.1 | 60 | r1 | 1800/1800 | 60 | 22 ms | 34 ms | 1.73 | 545 MB | yes |
| keyword | v1.7.1 | 60 | r2 | 1800/1800 | 60 | 27 ms | 50 ms | 1.91 | 590 MB | yes |
| keyword | v1.7.1 | 80 | r1 | 2400/2400 | 73.8 | 1087 ms | 2386 ms | 2.7 | 556 MB | no |
| keyword | v1.7.1 | 80 | r2 | 2400/2400 | 61.4 | 3796 ms | 9431 ms | 2.67 | 628 MB | no |
| keyword | v1.7.1 | 100 | r1 | 3000/3000 | 67.1 | 6964 ms | 15886 ms | 2.67 | 614 MB | no |
| keyword | v1.7.1 | 100 | r2 | 3000/3000 | 50.1 | 11586 ms | 21381 ms | 2.36 | 660 MB | no |
| keyword | v1.7.1 | 120 | r1 | 3599/3600 | 58.3 | 15303 ms | 36253 ms | 2.33 | 614 MB | no |

</details>

## What was measured

**Machine.** One Linux server, Intel Xeon Silver 4310 (2.1 GHz). EverOS ran on 4 cores
(`taskset -c 0-3`, `CPUQuota=400%`) with 8 GB of memory (`MemoryMax=8G`, a systemd scope);
the load generator and the stub providers ran on 4 other cores. Throughput depends on
single-core speed: `core_bench.py` took 506 ms (Python loop) and 2,160 ms (numpy) on
this machine, so run it before comparing your numbers with these.

**Store.** 168,821 memories (7,375 episodes and 161,446 atomic facts) of 132 users, built by
EverOS's own extraction from public conversations: the 10 LoCoMo10 conversations (each
speaker is a user) and the haystacks of the first 112 LongMemEval-S questions (one user
each). Extraction used `openai/gpt-4.1-mini`; embeddings were `Qwen/Qwen3-Embedding-4B` at
1024 dimensions. A user holds 1,346 memories at the median and 1,661 at most. The store
was built by EverOS main at `c2c8d6fc`, between 1.7.0 and 1.7.1; 1.7.1 searched it as
built, and 1.8.0 searched a copy whose vector indexes it had rebuilt as IVF_RQ. A store
built by 1.8.0 was not searched with 1.7.1.

**Queries.** 2,270 questions from the same datasets, each asked as the user it is about:
every LongMemEval question to its haystack's user, every LoCoMo question to the speaker it
names. Each request draws one at random and asks for the top 10.

**Model providers.** Stubbed, so the numbers measure EverOS rather than a provider. A
search makes one model call, the query embedding; the stub answers it after a latency drawn
at random from 14,252 recorded calls to `Qwen/Qwen3-Embedding-4B` on DeepInfra (median
0.39 s, p95 2.8 s, mean 0.78 s). `keyword` makes no model call.

**Load.** Open loop: request *i* of a step goes out at *i* / rate seconds whether or not
earlier requests have returned, so a server that falls behind shows climbing latency
instead of quietly lowering the rate. Each step lasts 60 s (30 s in the first `keyword`
run), on a freshly started server after a 30 s warm-up at 2 requests/s that is not
counted. A step **kept up** when every request returned 200, at least 90% of the rate
completed while requests were going out, and the median latency of its last third stayed
under twice that of its first third, or within 0.5 s of it. Rates rise until two steps
fall behind.

**Rounds.** `hybrid` ran twice, the second round with the servers in reverse order;
`keyword` ran three times, the servers in a different order each round. The
**sustained rate** is the highest rate at which every round kept up; the latency, CPU and
memory beside it are the worst round's. CPU is cores in use while requests were
being sent; memory is the server's peak resident set.

**Versions.** 1.7.1 from PyPI (`pip install everos==1.7.1`), and main at
`c5d9ace9` installed from GitHub, ahead of the 1.8.0 release. Each version ran on its
own copy of the store; 1.8.0's copy was converted to IVF_RQ before the first round.

**Second `keyword` run.** The `keyword` rates come from a second run on 2026-11-25, on the
same machine, store and builds, with four load-generator processes (`SENDERS=4`) and 60 s
steps (`STEP_SECONDS=60`). Round 2 of both 1.8.0 configurations was run again, because
other jobs ran on the machine during it.

## Reproduce

Everything is in [`benchmarks/throughput/`](throughput/). The store is built once with
real providers; every measurement after that uses the stubs and costs nothing.

1. **Build the store** (about 2 hours; with `openai/gpt-4.1-mini` on OpenRouter and
   DeepInfra embeddings it cost us $43). Fetch LoCoMo10 and the cleaned LongMemEval-S:

   ```bash
   bash benchmarks/download_data.sh locomo
   curl -L -o benchmarks/data/longmemeval_s_cleaned.json \
     https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json
   ```

   Create an empty root (`everos init --root <store>`) and start a server on it with your
   LLM and embedding providers configured. Use `Qwen/Qwen3-Embedding-4B` at `dim = 1024`:
   `sweep.sh` starts EverOS under that model name, and a store opens only under the
   embedding model it was indexed with. Then ingest:

   ```bash
   python benchmarks/throughput/build_store.py http://127.0.0.1:8000 benchmarks/data --root <store>
   ```

   It ingests `locomo:0-9,lme:0-111` and waits until background extraction has finished:
   five minutes in a row with an idle cascade, an empty memorize queue and no change to
   the markdown under `<store>`.
   A request that still fails after three attempts leaves its session unfinished, and the
   build exits 1 listing those sessions; on 1.8.0 or later, run it again (a resent
   message is not stored twice).
   Extraction is an LLM's work, so a rebuilt store holds about, not exactly, the same
   memories. Stop the server; its root (`EVEROS_ROOT`) is the store.

2. **Write the queries** next to the store:

   ```bash
   python benchmarks/throughput/make_queries.py <store> benchmarks/data <store>/../queries.jsonl
   ```

3. **Install each version under test** into its own virtual environment:

   ```bash
   uv venv v1.7.1 && uv pip install --python v1.7.1/bin/python "everos==1.7.1"
   ```

4. **Sweep.** One run per version and method; the store is copied first and never
   changed. On Linux, pin and cap the server as above:

   ```bash
   export SERVER_WRAP="systemd-run --user --scope --quiet -p MemoryMax=8G -p CPUQuota=400% taskset -c 0-3"
   export CLIENT_WRAP="taskset -c 4-7"
   bash benchmarks/throughput/sweep.sh v1.7.1/bin/everos <store> results/v1.7.1/r1/hybrid hybrid
   SENDERS=4 STEP_SECONDS=60 \
     bash benchmarks/throughput/sweep.sh v1.7.1/bin/everos <store> results/v1.7.1/r1/keyword keyword
   ```

   `keyword` is sent by four `load.py` processes (see Limits). The numbers above stepped
   by 10: `RATES="40 50 60 70 80 90 100"` for 1.7.1 and
   `RATES="100 110 120 130 140 150 160 170 180"` for 1.8.0. Each run copies the store,
   its `everos.toml` included, into `<out_dir>/store`: keep provider keys in the
   environment rather than in that file.

   A store written by v1.7.1 or earlier gets its vector indexes rebuilt as IVF_RQ by
   later versions, a few minutes after start-up; set `WAIT_INDEX=IvfRq` so the sweep
   starts after that (or point the next runs at the rebuilt copy in `<out_dir>/store`).
   `sweep.sh` documents its other settings at the top.

5. **Tables:** `python benchmarks/throughput/summarize.py results`.

## Limits

- **One machine, one store.** Rates follow single-core speed. How they change with the
  size of the store or of a user was not measured.
- **Search only.** Nothing is written while searches run. Concurrent writes compete for
  the same cores and lower these rates.
- **One provider's latency.** `hybrid` waits for the query embedding; a slower provider
  raises its latency and, once EverOS's concurrent query embeddings
  (`[embedding] query_max_concurrent`, 16) are all waiting, lowers its sustained rate.
- **Load generator.** In the first `keyword` run one `load.py` process froze at 160 and
  180 requests/s: requests that returned 200 took up to 1,469 s under a 60 s timeout,
  which only a stalled client produces, while the server used under 0.2 cores. Why it
  froze was not found. The `keyword` rates above were sent by four processes
  (`SENDERS=4`, each with its own query order); in that run no request took longer than
  the timeout and the server used over 1 core in every step. `summarize.py` marks a rate
  as a lower bound (≥) when the rate above it failed with the server under 0.3 cores,
  which a stalled client and a hung server both produce.
- **Queries lean on 20 users.** 2,158 of the 2,270 questions (95%) belong to the 20
  LoCoMo speakers; each LongMemEval user gets one. A search works within one user's
  memories, so 168,821 is the store's size, not what a query searches.
