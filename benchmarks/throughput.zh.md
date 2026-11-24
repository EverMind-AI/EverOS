# 检索吞吐

在一台 4 核 8 GB、存着 168,821 条记忆的服务器上，`hybrid` 检索 **1.8.0 每秒撑得住 14 到 16 次，1.7.1 是 4 次**，内存约为原来的六分之一；`keyword` 检索 **1.8.0 是 120 到至少 140 次，1.7.1 是 60 次**。测的是 1.8.0 发版前 main 的 `c5d9ace9`，模型服务用替身，只测 EverOS 本身。复现所需的一切都在 [`throughput/`](throughput/)，步骤见[复现](#复现)。

> English version: [throughput.md](throughput.md).

## 结果

每种配置跑两轮，某个频率两轮都撑住才算数。`glibc` 和 `jemalloc` 是同一个 1.8.0 包：Linux 上装了 `libjemalloc2`，1.8.0 的命令行就预加载 jemalloc，没装就用 glibc。

**撑得住的最高频率**，以及在该频率下的延迟、CPU 和内存峰值（取两轮里较差的一轮）：

| 检索 | 版本 | 每秒撑得住的请求数 | p50 | p95 | CPU（核） | 内存峰值 |
|---|---|---|---|---|---|---|
| `hybrid` | 1.7.1 | 4 | 4.50 s | 8.70 s | 3.45 | 4,093 MB |
| `hybrid` | 1.8.0，glibc | 14 | 1.06 s | 3.41 s | 2.50 | 681 MB |
| `hybrid` | 1.8.0，jemalloc | 16 | 0.72 s | 2.48 s | 2.07 | 595 MB |
| `keyword` | 1.7.1 | 60 | 27 ms | 50 ms | 1.91 | 590 MB |
| `keyword` | 1.8.0，glibc | 120 | 26 ms | 210 ms | 1.72 | 640 MB |
| `keyword` | 1.8.0，jemalloc | ≥ 140 | 18 ms | 256 ms | 1.59 | 468 MB |

≥ 140：再高一档 160，一轮撑住；另一轮没撑住，只是因为发压程序卡住了（见[局限](#局限)），服务本身的上限更高。

**同样负载下**，取三者都撑得住的最高频率：

| 检索、频率 | 版本 | p50 | p95 | CPU（核） | 内存峰值 |
|---|---|---|---|---|---|
| `hybrid`，4 次/秒 | 1.7.1 | 4.50 s | 8.70 s | 3.45 | 4,093 MB |
| | 1.8.0，glibc | 0.48 s | 2.21 s | 0.68 | 673 MB |
| | 1.8.0，jemalloc | 0.52 s | 2.17 s | 0.71 | 518 MB |
| `keyword`，60 次/秒 | 1.7.1 | 27 ms | 50 ms | 1.91 | 590 MB |
| | 1.8.0，glibc | 16 ms | 25 ms | 0.86 | 639 MB |
| | 1.8.0，jemalloc | 12 ms | 20 ms | 0.74 | 455 MB |

`hybrid` 几行背后是向量索引。1.7.1 给原子事实建的 IVF_FLAT 索引有 0.66 GB，放不进索引缓存，每次查询要把其中约 0.5 GB 重读一遍：在这份库上单独测，每次查询 449 ms、临时占 1.1 GB 内存。1.8.0 建 IVF_RQ 索引（同样的事实 24 MB）：35 到 50 ms，59 到 130 MB。1.7.1 写的库，升级后第一次维护时就会转换。

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
| hybrid | v1.7.1 | 4 | r2 | 240/240 | 3.4 | 4503 ms | 8698 ms | 3.31 | 4093 MB | yes |
| hybrid | v1.7.1 | 5 | r1 | 300/300 | 3.9 | 7953 ms | 14582 ms | 3.32 | 4379 MB | no |
| hybrid | v1.7.1 | 5 | r2 | 300/300 | 3.2 | 18934 ms | 23948 ms | 3.11 | 3644 MB | yes |
| hybrid | v1.7.1 | 6 | r1 | 360/360 | 4.5 | 11031 ms | 16040 ms | 3.46 | 3882 MB | no |
| hybrid | v1.7.1 | 6 | r2 | 360/360 | 4.4 | 12641 ms | 17468 ms | 3.43 | 3533 MB | yes |
| hybrid | v1.7.1 | 8 | r2 | 480/480 | 3.7 | 30924 ms | 40560 ms | 3.5 | 4081 MB | no |
| hybrid | v1.7.1 | 10 | r2 | 575/600 | 3 | 52015 ms | 59110 ms | 3.45 | 4254 MB | no |
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

**数据。** 132 个用户的 168,821 条记忆（7,375 个情节、161,446 条原子事实），由 EverOS 自己的抽取从公开对话建成：LoCoMo10 的全部 10 段对话（每个说话人是一个用户），和 LongMemEval-S 前 112 题的对话堆（每题一个用户）。抽取用 `openai/gpt-4.1-mini`，向量用 `Qwen/Qwen3-Embedding-4B`，1024 维。每个用户的记忆数中位 1,346 条，最多 1,661 条。

**问题。** 同两个数据集里的 2,270 个问题，每个都以它所问的用户身份发出：LongMemEval 的每道题发给它那堆对话的用户，LoCoMo 的每道题发给题里点名的说话人。每个请求随机抽一个问题，取前 10 条。

**模型服务。** 用替身，所以数字衡量的是 EverOS 而不是模型服务商。一次检索只调用一次模型，就是查询向量；替身从 14,252 次真实调用 DeepInfra 上 `Qwen/Qwen3-Embedding-4B` 的耗时里随机抽一个（中位 0.39 s，p95 2.8 s，均值 0.78 s），等这么久再回。`keyword` 不调用模型。

**加压方式。** 开环：一档里第 *i* 个请求在第 *i* / 频率 秒发出，不管前面的有没有返回，所以服务跟不上时表现为延迟上涨，而不是悄悄少发。`hybrid` 每档 60 秒，`keyword` 每档 30 秒；每次都是新起的服务，先以每秒 2 次预热 30 秒，不计入结果。一档**撑住**的意思是：所有请求都返回 200，并且后 1/3 请求的延迟中位数不超过前 1/3 的两倍，或比它多不到 0.5 秒。频率逐档升高，直到有两档没撑住。

**两轮。** 每组都跑两轮，第二轮服务的顺序倒过来。**撑得住的频率**取两轮都撑住的最高档；旁边的延迟、CPU、内存取两轮里较差的一轮。CPU 是发请求期间服务用掉的核数，内存是服务的常驻内存峰值。

**版本。** 1.7.1 来自 PyPI（`pip install everos==1.7.1`）；另一个是 1.8.0 发版前 main 的 `c5d9ace9`，从 GitHub 安装。每个版本用自己的一份库拷贝；1.8.0 那份在第一轮之前已转换成 IVF_RQ。

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
   python benchmarks/throughput/build_store.py http://127.0.0.1:8000 benchmarks/data
   ```

   它导入 `locomo:0-9,lme:0-111`，并等后台抽取做完。抽取是 LLM 做的，重建的库和我们的记忆数大致相同，但不会完全一样。停掉服务，它的数据目录（`EVEROS_ROOT`）就是这份库。

2. **生成问题**，放在库旁边：

   ```bash
   python benchmarks/throughput/make_queries.py <store> benchmarks/data <store>/../queries.jsonl
   ```

3. **每个要测的版本装进自己的虚拟环境**：

   ```bash
   uv venv v1.7.1 && uv pip install --python v1.7.1 "everos==1.7.1"
   ```

4. **逐档加压。** 每个版本、每种检索跑一次；库先拷贝一份，原件不动。在 Linux 上像上面那样给服务绑核、限资源：

   ```bash
   export SERVER_WRAP="systemd-run --user --scope --quiet -p MemoryMax=8G -p CPUQuota=400% taskset -c 0-3"
   export CLIENT_WRAP="taskset -c 4-7"
   for method in hybrid keyword; do
     bash benchmarks/throughput/sweep.sh v1.7.1/bin/everos <store> results/v1.7.1/r1/$method $method
   done
   ```

   1.7.1 及更早版本写的库，到了之后的版本会在启动几分钟内把向量索引重建成 IVF_RQ；设 `WAIT_INDEX=IvfRq` 让加压等它做完再开始（或者后面几次直接用 `<out_dir>/store` 里重建好的那份）。`sweep.sh` 的其他设置写在它开头。

5. **出表：** `python benchmarks/throughput/summarize.py results`。

## 局限

- **一台机器、一份库。** 频率随单核速度变化。库或单个用户的大小对它的影响没有测。
- **只测检索。** 检索期间没有写入。同时写入会和检索争同样的核，这些频率会降低。
- **一家服务商的延迟。** `hybrid` 要等查询向量；服务商更慢，它的延迟就更高，而且一旦 EverOS 的并发查询向量（`[embedding] query_max_concurrent`，16）都在等，撑得住的频率也会降低。
- **一个发压进程。** 在这台机器的发压核上，单个 `load.py` 到每秒 140 到 180 次之间会卡住：赶不上发送时间表，延迟和吞吐一起崩，服务却几乎闲着。某个频率的上一档只因为这个原因失败时（发请求期间服务用不到 0.3 核），`summarize.py` 把它标成下限（≥）。要测更高的 `keyword` 频率，就开多个 `load.py`（`--seed` 让每个用自己的问题顺序）。
