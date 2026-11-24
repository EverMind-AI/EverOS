# Search throughput

On a 4-core, 8 GB server holding 168,821 memories, `hybrid` search sustains
**14 to 16 requests per second on 1.8.0, against 4 on 1.7.1**, in about a sixth of
the memory; `keyword` search sustains **120 to at least 140, against 60**.
Measured on main at `c5d9ace9`, ahead of the 1.8.0 release, with the model providers
stubbed so that only EverOS is measured. Everything needed to reproduce it is in
[`throughput/`](throughput/); [Reproduce](#reproduce) has the steps.

> Also available in Chinese: [throughput.zh.md](throughput.zh.md).

## Results

Two rounds per configuration; a rate counts only when both rounds kept up at it.
`glibc` and `jemalloc` are the same 1.8.0 build: on Linux the 1.8.0 CLI preloads
jemalloc when `libjemalloc2` is installed, and runs on glibc otherwise.

**Highest sustained rate**, with latency, CPU and peak memory at that rate (the worse
of the two rounds):

| Search | Version | Sustained requests/s | p50 | p95 | CPU (cores) | Peak memory |
|---|---|---|---|---|---|---|
| `hybrid` | 1.7.1 | 4 | 4.50 s | 8.70 s | 3.45 | 4,093 MB |
| `hybrid` | 1.8.0, glibc | 14 | 1.06 s | 3.41 s | 2.50 | 681 MB |
| `hybrid` | 1.8.0, jemalloc | 16 | 0.72 s | 2.48 s | 2.07 | 595 MB |
| `keyword` | 1.7.1 | 60 | 27 ms | 50 ms | 1.91 | 590 MB |
| `keyword` | 1.8.0, glibc | 120 | 26 ms | 210 ms | 1.72 | 640 MB |
| `keyword` | 1.8.0, jemalloc | ≥ 140 | 18 ms | 256 ms | 1.59 | 468 MB |

≥ 140: the next rate, 160, kept up in one round and failed in the other only because
the load generator stalled (see [Limits](#limits)); the server's own ceiling is
higher.

**At the same load**, the highest rate all three sustain:

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
1024 dimensions. A user holds 1,346 memories at the median and 1,661 at most.

**Queries.** 2,270 questions from the same datasets, each asked as the user it is about:
every LongMemEval question to its haystack's user, every LoCoMo question to the speaker it
names. Each request draws one at random and asks for the top 10.

**Model providers.** Stubbed, so the numbers measure EverOS rather than a provider. A
search makes one model call, the query embedding; the stub answers it after a latency drawn
at random from 14,252 recorded calls to `Qwen/Qwen3-Embedding-4B` on DeepInfra (median
0.39 s, p95 2.8 s, mean 0.78 s). `keyword` makes no model call.

**Load.** Open loop: request *i* of a step goes out at *i* / rate seconds whether or not
earlier requests have returned, so a server that falls behind shows climbing latency
instead of quietly lowering the rate. Each step lasts 60 s for `hybrid` and 30 s for
`keyword`, on a freshly started server after a 30 s warm-up at 2 requests/s that is not
counted. A step **kept up** when every request returned 200 and the median latency of its
last third stayed under twice that of its first third, or within 0.5 s of it. Rates rise
until two steps fall behind.

**Rounds.** Every sweep ran twice, the second round with the servers in reverse order. The
**sustained rate** is the highest rate at which both rounds kept up; the latency, CPU and
memory beside it are the worse of the two rounds. CPU is cores in use while requests were
being sent; memory is the server's peak resident set.

**Versions.** 1.7.1 from PyPI (`pip install everos==1.7.1`), and main at
`c5d9ace9` installed from GitHub, ahead of the 1.8.0 release. Each version ran on its
own copy of the store; 1.8.0's copy was converted to IVF_RQ before the first round.

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
   python benchmarks/throughput/build_store.py http://127.0.0.1:8000 benchmarks/data
   ```

   It ingests `locomo:0-9,lme:0-111` and waits until background extraction has finished.
   Extraction is an LLM's work, so a rebuilt store holds about, not exactly, the same
   memories. Stop the server; its root (`EVEROS_ROOT`) is the store.

2. **Write the queries** next to the store:

   ```bash
   python benchmarks/throughput/make_queries.py <store> benchmarks/data <store>/../queries.jsonl
   ```

3. **Install each version under test** into its own virtual environment:

   ```bash
   uv venv v1.7.1 && uv pip install --python v1.7.1 "everos==1.7.1"
   ```

4. **Sweep.** One run per version and method; the store is copied first and never
   changed. On Linux, pin and cap the server as above:

   ```bash
   export SERVER_WRAP="systemd-run --user --scope --quiet -p MemoryMax=8G -p CPUQuota=400% taskset -c 0-3"
   export CLIENT_WRAP="taskset -c 4-7"
   for method in hybrid keyword; do
     bash benchmarks/throughput/sweep.sh v1.7.1/bin/everos <store> results/v1.7.1/r1/$method $method
   done
   ```

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
- **One load generator process.** On this machine's client cores one `load.py`
  stalls somewhere between 140 and 180 requests/s: it cannot keep its send schedule,
  latency and throughput collapse, and the server sits nearly idle. `summarize.py`
  marks a rate as a lower bound (≥) when the rate above it failed only that way
  (server under 0.3 cores while requests went out). Measure higher `keyword` rates
  with several `load.py` processes (`--seed` gives each its own query order).
