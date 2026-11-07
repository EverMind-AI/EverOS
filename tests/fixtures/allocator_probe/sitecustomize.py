"""Test-only native allocation probe, loaded before the CLI launcher.

Deliberately create arenas before importing EverOS. A working launcher must
replace this allocator, not merely change an environment variable. No model,
service, or background sampler runs here. Each probe thread is joined before
application code starts; the exit hook only reads malloc_info.
"""

from __future__ import annotations

import atexit
import ctypes
import json
import os
import sys
import threading
import xml.etree.ElementTree as ET


def probe():
    output = os.environ.get("EVEROS_ALLOCATOR_PROBE_LOG")
    if not output:
        return
    libc = ctypes.CDLL(None)
    libc.malloc.argtypes = [ctypes.c_size_t]
    libc.malloc.restype = ctypes.c_void_p
    libc.free.argtypes = [ctypes.c_void_p]
    libc.free.restype = None
    libc.tmpfile.argtypes = []
    libc.tmpfile.restype = ctypes.c_void_p
    libc.fileno.argtypes = [ctypes.c_void_p]
    libc.fileno.restype = ctypes.c_int
    libc.malloc_info.argtypes = [ctypes.c_int, ctypes.c_void_p]
    libc.malloc_info.restype = ctypes.c_int
    libc.fflush.argtypes = [ctypes.c_void_p]
    libc.fflush.restype = ctypes.c_int
    libc.fclose.argtypes = [ctypes.c_void_p]
    libc.fclose.restype = ctypes.c_int

    def record(phase):
        stream = libc.tmpfile()
        if not stream:
            raise RuntimeError("tmpfile failed")
        try:
            if libc.malloc_info(0, stream) != 0 or libc.fflush(stream) != 0:
                raise RuntimeError("malloc_info failed")
            fd = libc.fileno(stream)
            os.lseek(fd, 0, os.SEEK_SET)
            chunks = []
            while chunk := os.read(fd, 65536):
                chunks.append(chunk)
            xml = b"".join(chunks).decode()
        finally:
            libc.fclose(stream)
        row = {
            "phase": phase,
            "pid": os.getpid(),
            "arena_max": os.environ.get("MALLOC_ARENA_MAX"),
            "tunables": os.environ.get("GLIBC_TUNABLES"),
            "argv": sys.orig_argv,
            "heaps": len(ET.fromstring(xml).findall("heap")),
            "malloc_info": xml,
        }
        with open(output, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    count = 16
    start = threading.Barrier(count + 1, timeout=10)
    ready = threading.Barrier(count + 1, timeout=10)
    release = threading.Event()

    def allocate():
        pointers = []
        try:
            start.wait()
            for _ in range(256):
                ptr = libc.malloc(8192)
                if not ptr:
                    raise MemoryError("native allocation failed")
                pointers.append(ptr)
            ready.wait()
            if not release.wait(10):
                raise TimeoutError("probe did not release allocations")
        finally:
            for ptr in pointers:
                libc.free(ptr)

    threads = [threading.Thread(target=allocate) for _ in range(count)]
    for thread in threads:
        thread.start()
    try:
        start.wait()
        ready.wait()
        record("startup")
    finally:
        release.set()
        for thread in threads:
            thread.join(timeout=10)
    if any(thread.is_alive() for thread in threads):
        raise RuntimeError("probe thread did not exit")
    atexit.register(record, "exit")


probe()
