"""Dependency-free BM25Okapi selection of candidate source messages."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import TYPE_CHECKING

from .cases import decode_snapshot
from .scope import same_tenant

if TYPE_CHECKING:
    from smartcomment.runtime import ExecNetwork

    from .models import FailedQueryCase


def tokenize(text: str) -> list[str]:
    """Match the original MemTrace English lexical tokenizer."""
    return re.findall(r"[a-z0-9_]+", text.lower())


def bm25_scores(documents: list[str], query: str) -> list[float]:
    """BM25Okapi defaults: k1=1.5, b=0.75, negative-IDF epsilon=0.25."""
    corpus = [tokenize(text) for text in documents]
    if not corpus:
        return []
    frequencies = [Counter(tokens) for tokens in corpus]
    document_frequency: Counter[str] = Counter()
    for counts in frequencies:
        document_frequency.update(counts.keys())
    n = len(corpus)
    average_length = sum(map(len, corpus)) / n
    if not document_frequency or not average_length:
        return [0.0] * n
    idf = {
        term: math.log(n - count + 0.5) - math.log(count + 0.5)
        for term, count in document_frequency.items()
    }
    epsilon = 0.25 * sum(idf.values()) / len(idf)
    idf = {term: epsilon if weight < 0 else weight for term, weight in idf.items()}
    scores = []
    for tokens, counts in zip(corpus, frequencies, strict=True):
        normalizer = 1.5 * (0.25 + 0.75 * len(tokens) / average_length)
        scores.append(
            sum(
                idf.get(term, 0.0) * counts[term] * 2.5 / (counts[term] + normalizer)
                for term in tokenize(query)
            )
        )
    return scores


def retrieve_starting_nodes(
    graph: ExecNetwork, case: FailedQueryCase, limit: int = 8
) -> list[str]:
    """Rank raw messages by question plus reference; these are pseudo evidence."""
    if limit < 1:
        raise ValueError("The retrieval limit must be positive.")
    query_node = graph.get_variable(case.query_full_node_id)
    nodes = sorted(
        (
            node
            for node in graph.search_variables(category="message")
            if same_tenant(node, query_node)
            and node.created_at <= query_node.created_at
        ),
        key=lambda node: (node.created_at, node.full_node_id),
    )
    documents = []
    usable_nodes = []
    for node in nodes:
        snapshot = decode_snapshot(node.value)
        text = snapshot.get("content", "") if isinstance(snapshot, dict) else snapshot
        if isinstance(text, str) and text.strip():
            documents.append(text)
            usable_nodes.append(node)
    query = case.query + "\n" + "\n".join(case.golden_answers)
    scores = bm25_scores(documents, query)
    ranked = sorted(
        zip(usable_nodes, scores, strict=True),
        key=lambda item: (-item[1], item[0].created_at, item[0].full_node_id),
    )
    return [node.full_node_id for node, _score in ranked[:limit]]
