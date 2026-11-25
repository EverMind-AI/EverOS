"""Stand-in model providers for the search throughput benchmark.

usage: python stub_models.py <port> [latency.csv]

Serves the two calls a search makes on an OpenAI-compatible embedding endpoint
and a vLLM-style rerank endpoint, so that EverOS, not a remote provider, is what
the load measures:

  POST /emb/embeddings  unit vectors hashed from the input's words; before
                        answering, waits a latency drawn at random from
                        ``latency.csv`` (default: the recorded query-embedding
                        latencies next to this file)
  POST /rerank/rerank   scores by words shared with the query, no wait
  POST /llm/...         400: a search should not call the LLM, and a call shows
                        up as a failed request rather than a fast fake answer

Prints the number of calls per route when it stops (SIGINT / SIGTERM).
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import math
import random
import re
import sys
from collections import Counter
from collections.abc import AsyncIterator
from pathlib import Path

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

_WORD = re.compile(r"\w+")
port = int(sys.argv[1])
latency_file = (
    Path(sys.argv[2])
    if len(sys.argv) > 2
    else Path(__file__).with_name("query_embedding_latency.csv")
)
latencies = [float(x) for x in latency_file.read_text().split()[1:]]
rng = random.Random(11)
calls: Counter[str] = Counter()


def embed(text: str, dim: int) -> list[float]:
    vec = [0.0] * dim
    for word in _WORD.findall(text.lower()):
        digest = hashlib.sha256(word.encode()).digest()
        vec[int.from_bytes(digest[:4], "big") % dim] += 1.0 if digest[4] & 1 else -1.0
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm else [1.0] + [0.0] * (dim - 1)


async def embeddings(request: Request) -> JSONResponse:
    body = await request.json()
    inputs = body["input"] if isinstance(body["input"], list) else [body["input"]]
    dim = int(body.get("dimensions") or 1024)
    calls["emb"] += 1
    await asyncio.sleep(rng.choice(latencies))
    data = [
        {"object": "embedding", "index": i, "embedding": embed(t, dim)}
        for i, t in enumerate(inputs)
    ]
    usage = {"prompt_tokens": len(inputs), "total_tokens": len(inputs)}
    return JSONResponse({"object": "list", "data": data, "usage": usage})


async def rerank(request: Request) -> JSONResponse:
    body = await request.json()
    calls["rerank"] += 1
    query = set(_WORD.findall(body["query"].lower()))
    results = [
        {"index": i, "relevance_score": len(query & set(_WORD.findall(d.lower()))) / 10}
        for i, d in enumerate(body["documents"])
    ]
    return JSONResponse({"results": results})


async def llm(request: Request) -> JSONResponse:
    calls["llm"] += 1
    return JSONResponse({"error": "no LLM in a search benchmark"}, status_code=400)


@contextlib.asynccontextmanager
async def lifespan(_: Starlette) -> AsyncIterator[None]:
    print(f"stub models on 127.0.0.1:{port}, {len(latencies)} latencies", flush=True)
    yield
    print(f"calls {dict(calls)}", flush=True)


app = Starlette(
    routes=[
        Route("/emb/embeddings", embeddings, methods=["POST"]),
        Route("/rerank/rerank", rerank, methods=["POST"]),
        Route("/llm/{path:path}", llm, methods=["POST"]),
    ],
    lifespan=lifespan,
)
uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning", backlog=1024)
