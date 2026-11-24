"""Single-core speed of this machine, to set beside the throughput numbers.

usage: python core_bench.py

Two timings, best of five, one thread: a pure-Python loop (what request handling
spends its time on) and element-wise float32 dot products over 20k x 1024
vectors (numpy's SIMD loops, no BLAS). Lower is faster.
"""

from __future__ import annotations

import os
from collections.abc import Callable

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import time

import numpy as np


def best(fn: Callable[[], float]) -> float:
    return min(fn() for _ in range(5))


def py_loop() -> float:
    t, s = time.perf_counter(), 0
    for i in range(10_000_000):
        s += i
    return time.perf_counter() - t


M = np.random.default_rng(0).standard_normal((20_000, 1024), dtype=np.float32)
q = np.random.default_rng(1).standard_normal(1024, dtype=np.float32)


def dots() -> float:
    t = time.perf_counter()
    for _ in range(50):
        (M * q).sum(axis=1).argmax()
    return time.perf_counter() - t


print(
    f"python loop 1e7: {best(py_loop) * 1000:.0f} ms   "
    f"numpy 50 x (20k x 1024 dot): {best(dots) * 1000:.0f} ms"
)
