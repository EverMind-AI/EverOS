# 检索吞吐

在一台 4 核 8 GB、存着 168,821 条记忆的服务器上，`hybrid` 检索 **1.8.0 每秒撑得住 14 到 16 次，1.7.1 不到 4 次**，内存约为原来的六分之一；`keyword` 检索 **1.8.0 是 120 到 140 次，1.7.1 是 70 次**。测的是 1.8.0 发版前 main 的 `c5d9ace9`，模型服务用替身，只测 EverOS 本身。复现所需的一切都在 [`throughput/`](throughput/)，步骤见[复现](#复现)。

> English version: [throughput.md](throughput.md).

## 结果

`hybrid` 跑两轮，`keyword` 跑三轮，某个频率每一轮都撑住才算数。`keyword` 用四个发压进程发，因为第一次测试里单个进程在每秒 160 和 180 次时卡死了（见[局限](#局限)）。`glibc` 和 `jemalloc` 是同一个 1.8.0 包：Linux 上装了 `libjemalloc2`，1.8.0 的命令行就预加载 jemalloc，没装就用 glibc。

**撑得住的最高频率**，以及在该频率下的延迟、CPU 和内存峰值（取最差的一轮），和任意一轮撑住过的最高频率：

| 检索 | 版本 | 每秒撑得住的请求数 | p50 | p95 | CPU（核） | 内存峰值 | 单轮最高 |
|---|---|---|---|---|---|---|---|
| `hybrid` | 1.7.1 | < 4 | – | – | – | – | 4 |
| `hybrid` | 1.8.0，glibc | 14 | 1.06 s | 3.41 s | 2.50 | 681 MB | 14 |
| `hybrid` | 1.8.0，jemalloc | 16 | 0.72 s | 2.48 s | 2.07 | 595 MB | 16 |
| `keyword` | 1.7.1 | 70 | 75 ms | 345 ms | 2.35 | 597 MB | 70 |
| `keyword` | 1.8.0，glibc | 120 | 25 ms | 197 ms | 1.75 | 700 MB | 150 |
| `keyword` | 1.8.0，jemalloc | 140 | 24 ms | 241 ms | 1.56 | 563 MB | 170 |

1.7.1 连 `hybrid` 最低的一档都没撑住：每秒 4 次时，第二轮每秒只处理完 3.45 次（见下面同样负载的表）。

靠近上限时，各轮结果不一致。那里的一档可能因为服务停顿几秒就没撑住：剩下的余量太少，这一档结束前消化不完积压。`jemalloc` 第一轮 130 没撑住，140 和 150 却撑住了；第三轮一直撑到 170。服务为什么停顿没有查到，那几秒的日志里什么也没有。

**同样负载下**：`hybrid` 取最低的一档 4 次/秒（1.7.1 两轮里只有一轮撑住）；`keyword` 取第一次测试（一个发压进程、每档 30 秒、两轮）里三者都撑住的 60 次/秒：

| 检索、频率 | 版本 | p50 | p95 | CPU（核） | 内存峰值 |
|---|---|---|---|---|---|
| `hybrid`，4 次/秒 | 1.7.1 | 4.50 s | 8.70 s | 3.45 | 4,093 MB |
| | 1.8.0，glibc | 0.48 s | 2.21 s | 0.68 | 673 MB |
| | 1.8.0，jemalloc | 0.52 s | 2.17 s | 0.71 | 518 MB |
| `keyword`，60 次/秒 | 1.7.1 | 27 ms | 50 ms | 1.91 | 590 MB |
| | 1.8.0，glibc | 16 ms | 25 ms | 0.86 | 639 MB |
| | 1.8.0，jemalloc | 12 ms | 20 ms | 0.74 | 455 MB |

`hybrid` 几行背后是向量索引。1.7.1 给原子事实建的 IVF_FLAT 索引有 0.66 GB，放不进索引缓存，每次查询要把其中约 0.5 GB 重读一遍：在这份库上单独测，每次查询 449 ms、临时占 1.1 GB 内存。1.8.0 建 IVF_RQ 索引（同样的事实 24 MB）：35 到 50 ms，59 到 130 MB。1.7.1 写的库，升级后第一次维护时就会转换。

`jemalloc` 的 `hybrid` 每秒 16 次是 EverOS 自己的上限，不是查询向量卡的。一次 `hybrid` 检索要等一个查询向量，EverOS 同时最多算 16 个（`[embedding] query_max_concurrent`）；每秒 20 次、替身平均耗时 0.78 秒时，平均约有 15.6 个在途，接近这个上限。2026-11-25 的对照测试把上限调到 64：同时发出的向量请求最多到了 60 个，`jemalloc` 仍然是每秒 16 次撑住、20 次跟不上，两轮都一样，每秒只处理完约 16 次，4 核只用了约 2.3 核。卡在 EverOS 的哪一部分没有查到。

<details>
<summary>对照测试：查询向量并发上限 16 和 64，<code>jemalloc</code></summary>

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
<summary>每一轮的每一档</summary>

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
<summary>第一次 <code>keyword</code> 测试：一个发压进程，每档 30 秒</summary>

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

## 测了什么

**机器。** 一台 Linux 服务器，Intel Xeon Silver 4310（2.1 GHz）。EverOS 跑在 4 个核上（`taskset -c 0-3`、`CPUQuota=400%`），内存 8 GB（`MemoryMax=8G`，systemd scope）；发压程序和模型替身跑在另外 4 个核上。吞吐取决于单核速度：这台机器上 `core_bench.py` 是 506 ms（Python 循环）和 2,160 ms（numpy），拿你的数字和这里比之前先跑一下它。

**数据。** 132 个用户的 168,821 条记忆（7,375 个情节、161,446 条原子事实），由 EverOS 自己的抽取从公开对话建成：LoCoMo10 的全部 10 段对话（每个说话人是一个用户），和 LongMemEval-S 前 112 题的对话堆（每题一个用户）。抽取用 `openai/gpt-4.1-mini`，向量用 `Qwen/Qwen3-Embedding-4B`，1024 维。每个用户的记忆数中位 1,346 条，最多 1,661 条。这份库由 EverOS main 的 `c2c8d6fc`（1.7.0 与 1.7.1 之间）建成；1.7.1 直接检索它，1.8.0 检索的是一份向量索引已重建为 IVF_RQ 的拷贝。用 1.8.0 建的库没有拿 1.7.1 测过。

**问题。** 同两个数据集里的 2,270 个问题，每个都以它所问的用户身份发出：LongMemEval 的每道题发给它那堆对话的用户，LoCoMo 的每道题发给题里点名的说话人。每个请求随机抽一个问题，取前 10 条。

**模型服务。** 用替身，所以数字衡量的是 EverOS 而不是模型服务商。一次检索只调用一次模型，就是查询向量；替身从 14,252 次真实调用 DeepInfra 上 `Qwen/Qwen3-Embedding-4B` 的耗时里随机抽一个（中位 0.39 s，p95 2.8 s，均值 0.78 s），等这么久再回。`keyword` 不调用模型。

**加压方式。** 开环：一档里第 *i* 个请求在第 *i* / 频率 秒发出，不管前面的有没有返回，所以服务跟不上时表现为延迟上涨，而不是悄悄少发。每档 60 秒（第一次 `keyword` 测试是 30 秒）；每次都是新起的服务，先以每秒 2 次预热 30 秒，不计入结果。一档**撑住**的意思是：所有请求都返回 200，发请求期间处理完的请求至少达到频率的 90%，并且后 1/3 请求的延迟中位数不超过前 1/3 的两倍，或比它多不到 0.5 秒。频率逐档升高，直到有两档没撑住。

**轮次。** `hybrid` 跑两轮，第二轮服务的顺序倒过来；`keyword` 跑三轮，每轮服务的顺序都不同。**撑得住的频率**取每一轮都撑住的最高档；旁边的延迟、CPU、内存取最差的一轮。CPU 是发请求期间服务用掉的核数，内存是服务的常驻内存峰值。

**版本。** 1.7.1 来自 PyPI（`pip install everos==1.7.1`）；另一个是 1.8.0 发版前 main 的 `c5d9ace9`，从 GitHub 安装。每个版本用自己的一份库拷贝；1.8.0 那份在第一轮之前已转换成 IVF_RQ。

**第二次 `keyword` 测试。** `keyword` 的频率来自 2026-11-25 的第二次测试：同一台机器、同一份库、同样的版本，用四个发压进程（`SENDERS=4`），每档 60 秒（`STEP_SECONDS=60`）。两个 1.8.0 配置的第 2 轮重跑过，因为当时机器上跑着别的任务。

## 复现

全部在 [`benchmarks/throughput/`](throughput/)。库用真实模型服务建一次；之后每次测量都用替身，不花钱。

1. **建库**（约 2 小时；我们用 OpenRouter 上的 `openai/gpt-4.1-mini` 和 DeepInfra 的向量服务，花了 $43）。下载 LoCoMo10 和清洗过的 LongMemEval-S：

   ```bash
   bash benchmarks/download_data.sh locomo
   curl -L -o benchmarks/data/longmemeval_s_cleaned.json \
     https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json
   ```

   建一个空的数据目录（`everos init --root <store>`），在上面起服务，配好你的 LLM 和向量服务。向量模型用 `Qwen/Qwen3-Embedding-4B`、`dim = 1024`：`sweep.sh` 用这个模型名起 EverOS，而库只能用建索引时的那个向量模型打开。然后导入：

   ```bash
   python benchmarks/throughput/build_store.py http://127.0.0.1:8000 benchmarks/data --root <store>
   ```

   它导入 `locomo:0-9,lme:0-111`，并等后台抽取做完：连续五分钟索引队列为空、memorize 队列为空、`<store>` 下的 markdown 没有变化。某个请求试了三次仍失败时，那段对话就不再往下导，全部导完后脚本列出这些对话并以 1 退出；用 1.8.0 及以后的版本，直接再跑一遍即可（重发的消息不会存两次）。抽取是 LLM 做的，重建的库和我们的记忆数大致相同，但不会完全一样。停掉服务，它的数据目录（`EVEROS_ROOT`）就是这份库。

2. **生成问题**，放在库旁边：

   ```bash
   python benchmarks/throughput/make_queries.py <store> benchmarks/data <store>/../queries.jsonl
   ```

3. **每个要测的版本装进自己的虚拟环境**：

   ```bash
   uv venv v1.7.1 && uv pip install --python v1.7.1/bin/python "everos==1.7.1"
   ```

4. **逐档加压。** 每个版本、每种检索跑一次；库先拷贝一份，原件不动。在 Linux 上像上面那样给服务绑核、限资源：

   ```bash
   export SERVER_WRAP="systemd-run --user --scope --quiet -p MemoryMax=8G -p CPUQuota=400% taskset -c 0-3"
   export CLIENT_WRAP="taskset -c 4-7"
   bash benchmarks/throughput/sweep.sh v1.7.1/bin/everos <store> results/v1.7.1/r1/hybrid hybrid
   SENDERS=4 STEP_SECONDS=60 \
     bash benchmarks/throughput/sweep.sh v1.7.1/bin/everos <store> results/v1.7.1/r1/keyword keyword
   ```

   `keyword` 用四个 `load.py` 进程发（见局限）。上面的数字每档相差 10：1.7.1 用 `RATES="40 50 60 70 80 90 100"`，1.8.0 用 `RATES="100 110 120 130 140 150 160 170 180"`。每次运行都会把库连同它的 `everos.toml` 拷到 `<out_dir>/store`：服务商的密钥放在环境变量里，不要写在这个文件里。

   1.7.1 及更早版本写的库，到了之后的版本会在启动几分钟内把向量索引重建成 IVF_RQ；设 `WAIT_INDEX=IvfRq` 让加压等它做完再开始（或者后面几次直接用 `<out_dir>/store` 里重建好的那份）。`sweep.sh` 的其他设置写在它开头。

5. **出表：** `python benchmarks/throughput/summarize.py results`。

## 局限

- **一台机器、一份库。** 频率随单核速度变化。库或单个用户的大小对它的影响没有测。
- **只测检索。** 检索期间没有写入。同时写入会和检索争同样的核，这些频率会降低。
- **一家服务商的延迟。** `hybrid` 要等查询向量；服务商更慢，它的延迟就更高，而且一旦 EverOS 的并发查询向量（`[embedding] query_max_concurrent`，16）都在等，撑得住的频率也会降低。
- **发压程序。** 第一次 `keyword` 测试里，单个 `load.py` 进程在每秒 160 和 180 次时卡死了：返回 200 的请求最长花了 1,469 秒，而超时设的是 60 秒，只有发压端自己停住才会这样；同时服务只用了不到 0.2 核。为什么卡死没有查到。上面的 `keyword` 频率是四个进程发的（`SENDERS=4`，各用自己的问题顺序）；那次测试里没有请求超过超时时间，每一档服务都用了 1 核以上。某个频率的上一档失败时服务用不到 0.3 核，`summarize.py` 就把它标成下限（≥）；发压端卡住和服务卡死都会这样。
- **问题集中在 20 个用户。** 2,270 个问题里 2,158 个（95%）属于 LoCoMo 的 20 个说话人，LongMemEval 的每个用户只有一个。检索只在一个用户自己的记忆里进行，所以 168,821 是整个库的大小，不是一次查询要搜的量。
