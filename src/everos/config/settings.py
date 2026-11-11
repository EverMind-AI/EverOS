"""Application settings.

Loaded by :func:`load_settings`. Source priority (later wins):

    1. ``config/default.toml`` (shipped values; lowest priority)
    2. ``<root>/everos.toml`` (user config; optional; ``<root>`` resolved by
       :func:`resolve_root`)
    3. ``EVEROS_<SECTION>__<KEY>`` environment variables
    4. The evaluation harness's read-only switches (``EVEROS_DISABLE_CASCADE``,
       ``EVEROS_DISABLE_CASCADE_WATCHER``, ``EVEROS_DISABLE_OME``), each only
       ever turning its setting off
    5. Init args passed programmatically (highest priority)

The memory root is resolved by :func:`resolve_root`:
``explicit arg > EVEROS_ROOT env > ~/.everos``.

The settings tree mirrors the TOML structure: ``settings.sqlite.busy_timeout_ms``
maps to ``[sqlite].busy_timeout_ms`` and to ``EVEROS_SQLITE__BUSY_TIMEOUT_MS``.

``load_settings`` is ``functools.cache``-d so callers in hot paths (e.g.
:mod:`everos.core.utils.datetime`) don't re-parse the TOML on every
call. Tests that mutate environment variables must call
``load_settings.cache_clear()`` after the mutation to invalidate.
"""

from __future__ import annotations

import difflib
import os
import re
import tomllib
from collections.abc import Mapping
from functools import cache
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

from everos.core.observability.logging import get_logger

logger = get_logger(__name__)

_DEFAULT_TOML_PATH = Path(__file__).parent / "default.toml"
_DEFAULT_ROOT = Path("~/.everos")


def resolve_root(explicit: str | None = None) -> Path:
    """Resolve the memory-root path.

    Priority: explicit arg > EVEROS_ROOT env > ~/.everos default.

    Args:
        explicit: Caller-supplied path string (e.g. from ``--root`` CLI flag).

    Returns:
        Absolute resolved path to the memory root.
    """
    if explicit:
        return Path(explicit).expanduser().resolve()
    from_env = os.environ.get("EVEROS_ROOT")
    if from_env:
        return Path(from_env).expanduser().resolve()
    return _DEFAULT_ROOT.expanduser().resolve()


ResponseFormatMode = Literal["json_schema", "json_object", "prompt_only"]
"""How an LLM section asks its endpoint for JSON; see
:attr:`LLMSettings.response_format_mode`."""


class _Section(BaseModel):
    """One configuration section: a key it does not declare is an error.

    Settings used to ignore unknown keys, so a misspelt key in ``everos.toml``
    or an ``EVEROS_<SECTION>__<KEY>`` variable fell back to its default without
    a word. :class:`Settings` reports every unknown key at once, with where it
    came from; ``extra="forbid"`` keeps a section built on its own as strict.
    """

    model_config = ConfigDict(extra="forbid")


class MemorySettings(_Section):
    """Memory configuration."""

    timezone: str = "UTC"
    """Effective timezone for date buckets and timestamps.

    Default ``"UTC"``. Override via ``[memory] timezone = "..."`` in
    TOML or ``EVEROS_MEMORY__TIMEZONE`` env var. Validated against
    :class:`zoneinfo.ZoneInfo` at load time, so an invalid name fails
    fast (no silent fallback). This is the **sole** source of truth for
    the project's effective timezone — the OS ``TZ`` env var is *not*
    consulted, keeping the configuration deterministic.
    """

    @field_validator("timezone")
    @classmethod
    def _validate_timezone(cls, v: str) -> str:
        try:
            ZoneInfo(v)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(f"invalid timezone: {v!r}") from exc
        return v


class ApiSettings(_Section):
    """HTTP API server bind configuration.

    Default ``host = "127.0.0.1"`` keeps the server on loopback only,
    matching the threat model in ``SECURITY.md``: EverOS ships **no
    built-in authentication**, so binding to a routable interface
    (``0.0.0.0`` etc.) without your own gateway / auth layer in front
    is unsupported.

    Env binding:
        EVEROS_API__HOST
        EVEROS_API__PORT
    """

    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)


class SqliteSettings(_Section):
    """SQLite tunables applied as PRAGMAs on every new connection."""

    journal_mode: Literal["WAL", "DELETE", "MEMORY", "OFF", "TRUNCATE", "PERSIST"] = (
        "WAL"
    )
    synchronous: Literal["FULL", "NORMAL", "OFF", "EXTRA"] = "NORMAL"
    foreign_keys: bool = True
    temp_store: Literal["DEFAULT", "FILE", "MEMORY"] = "MEMORY"
    busy_timeout_ms: int = Field(default=5000, ge=0)
    pool_size: int = Field(default=5, ge=1)
    """Connections the pool keeps open. SQLAlchemy's own default; named here so
    the next two are tunable alongside it rather than inherited invisibly."""
    max_overflow: int = Field(default=10, ge=0)
    """Extra connections allowed above ``pool_size`` under load."""
    pool_timeout_seconds: float = Field(default=30.0, gt=0)
    """How long a caller waits for a free connection before raising.

    The reason this is configurable rather than left at the library default: a
    checkout that never returns turns every later caller into a silent hang. Two
    benchmark servers died exactly that way -- aiosqlite connection threads grew
    from a steady 6-10 to 20 and 58, with 7 and 22 of them parked inside
    ``aiosqlite``'s connect path, and every SQLite file stopped being written
    (one froze for 3h33m, the other 2h17m). Neither process was dead: HTTP still
    answered, the event loop still ran, and the OME queue simply stopped
    draining because no strategy could persist its own result. A bounded wait
    converts that into a loud, retryable error instead.
    """
    pool_recycle_seconds: int = Field(default=1800, ge=-1)
    """Discard and reopen a connection older than this; ``-1`` disables.

    A leaked-but-idle connection is reclaimed on its next checkout attempt
    instead of being held for the life of the process."""
    pool_pre_ping: bool = True
    """Verify a pooled connection is alive before handing it out.

    Costs one round trip per checkout against a local file; buys detection of
    connections whose underlying aiosqlite thread is no longer serviceable --
    the state both stalled servers were in."""
    journal_size_limit_bytes: int = Field(default=64 * 1024 * 1024, ge=0)
    cache_size_kb: int = Field(default=2048, ge=0)


class LLMSettings(_Section):
    """LLM client configuration.

    Read by the service layer when lazily constructing the LLM client
    handed to algo extractors. Provider-agnostic field names — the
    project follows the OpenAI API protocol so any OpenAI-compatible
    endpoint plugs in via ``base_url``.

    Env binding (via parent ``Settings``):
        EVEROS_LLM__MODEL
        EVEROS_LLM__API_KEY
        EVEROS_LLM__BASE_URL
        EVEROS_LLM__TIMEOUT_SECONDS
        EVEROS_LLM__EXTRA
        EVEROS_LLM__RESPONSE_FORMAT_MODE
        EVEROS_LLM__INVALID_REPLY_RETRIES
        EVEROS_LLM__TEMPERATURE
        EVEROS_LLM__RETRY_TEMPERATURE
    """

    model: str = "gpt-4.1-mini"
    api_key: SecretStr | None = None
    base_url: str | None = None
    timeout_seconds: float = Field(default=60.0, gt=0)
    """Per-request deadline handed to the algo client.

    Matches the algo default, so leaving it alone changes nothing. It exists
    because extraction latency is a property of the endpoint, not of this
    project: a hosted frontier model answers an extraction prompt in seconds,
    while a self-hosted mid-size model on a shared gateway can take minutes for
    the same prompt. Without this the deadline was unreachable from config, and
    a slow endpoint could only fail -- three attempts, three timeouts, one
    dead-lettered memory.
    """
    extra: dict[str, Any] = Field(default_factory=dict)
    """Provider-specific request fields merged into every chat call.

    An escape hatch for what the OpenAI protocol does not cover, kept generic
    because each gateway spells its own knobs differently. The case it was added
    for: a reasoning model served over an OpenAI-compatible endpoint keeps
    thinking by default, and thinking is billed against ``max_tokens`` -- so an
    extraction call can spend its entire budget reasoning and return an empty
    ``content``, which reads downstream as "the model found no facts" rather
    than as a misconfiguration.

    **Non-standard fields must be nested under ``extra_body``.** These entries
    are passed as keyword arguments to the OpenAI SDK's ``create()``, which
    raises ``TypeError`` on an unrecognised top-level name and forwards only
    ``extra_body`` into the request JSON. Getting this wrong fails loudly at the
    first call, which is the good case; getting it *absent* is the quiet one::

        EVEROS_LLM__EXTRA='{"extra_body":
            {"chat_template_kwargs": {"enable_thinking": false}}}'

    Measured on one gateway with an atomic-facts prompt: nested = 12.7s and 2264
    characters, bare = ``TypeError``, omitted = 42.4s and **zero** characters.

    Merged under the per-call ``extra``, so a caller can still override a key.
    """

    response_format_mode: ResponseFormatMode = "json_schema"
    """How the reply format is requested from the endpoint.

    ``json_schema`` sends the strict schema of the operator's reply model,
    ``json_object`` sends JSON mode, ``prompt_only`` sends neither and relies on
    the prompt (the reply is still JSON, and is still validated locally). It is
    a deployment choice because endpoint support differs: everalgo's contract
    (``llm-contract.md`` §7.2) measured 24 of 32 OpenRouter endpoints for one
    model accepting a strict schema, 29 accepting JSON mode, and all 32 with
    neither -- and sending a format the provider lacks reroutes the request.
    """
    invalid_reply_retries: int = Field(default=1, ge=0)
    """Extra attempts after a reply that fails validation against the
    operator's model, has no choice, or was aborted by the server. Bounded
    here, cheaply, rather than by rerunning the whole operator; ``0`` hands the
    first failure straight to the caller. A retry is the same request, except
    that it is sampled at :attr:`retry_temperature`."""
    temperature: float | None = Field(default=0.0, ge=0)
    """Sampling temperature of every extraction call (boundary, episode, profile,
    the picture route, group memory), unless a call sets its own.
    ``None`` leaves the field out of the request, so the endpoint's default applies.

    0.0 because extraction should be repeatable: the same conversation should cut
    the same memcells and write the same memories. Every request sent 0.0 until
    2026-10-30, when the provider's default became "leave it out"; DeepSeek then
    samples at 1.0, and two runs over the same 10K-word transcript (2026-11-11) cut
    5 and 9 memcells and cost 50% apart. ``None`` is for an endpoint that rejects
    the field."""
    retry_temperature: float | None = Field(default=0.3, ge=0)
    """Sampling temperature of a retry after an unusable reply
    (:attr:`invalid_reply_retries`). At temperature 0 the identical request tends to
    reproduce the identical bad reply, so a retry samples a little more freely.
    Used only when it is above the first attempt's temperature; ``None``, or a
    first attempt that sends no temperature, resends the request unchanged."""

    @field_validator("temperature", "retry_temperature", mode="before")
    @classmethod
    def _none_spelled_out(cls, value: object) -> object:
        """``"none"`` / ``"null"`` mean ``None``: a TOML file and an environment
        variable have no other way to leave the field out."""
        if isinstance(value, str) and value.strip().lower() in ("none", "null"):
            return None
        return value


class DeciderSettings(_Section):
    """LLM driving the multi-round retrieval decider.

    A flat section mirroring ``[llm]``, kept separate for the same reason
    ``[multimodal]`` is: the two jobs have different demands. ``[llm]`` extracts
    memories during ingestion -- a long, throughput-bound batch job -- while this one
    runs inside the search request and decides, round by round, which episodes are
    core and what to query next. Sharing one setting forced the decider to be whatever
    model the store happened to be extracted with, and made "which model made the
    retrieval decisions" unanswerable after the fact.

    Empty ``model`` falls back to ``[llm]``, so existing deployments and every store
    built before this section existed behave exactly as they did.

    The loop-tuning fields below used to exist only as ``EVEROS_LLMMR_*`` environment
    variables read once at import time inside
    :mod:`everos.memory.search.llm_multiround`. That made them undiscoverable -- nothing
    in the config named them, so an operator could not find out they existed, let
    alone set one from a file. They are declared here for the same reason the model
    is: a run's behaviour should be readable off its configuration. The legacy env
    names still win when set, so nothing in flight changes.

    Env binding (via parent ``Settings``):
        EVEROS_DECIDER__MODEL
        EVEROS_DECIDER__API_KEY
        EVEROS_DECIDER__BASE_URL
        EVEROS_DECIDER__TIMEOUT_SECONDS
        EVEROS_DECIDER__EXTRA
        EVEROS_DECIDER__MAX_ROUNDS ... and one per field below
    """

    model: str = ""
    api_key: SecretStr | None = None
    base_url: str | None = None
    timeout_seconds: float = Field(default=60.0, gt=0)
    """Per-request deadline for one decider round. See
    :attr:`LLMSettings.timeout_seconds`; separate because this one sits inside a
    search request, where the acceptable wait is bounded by the caller rather
    than by a background queue."""
    extra: dict[str, Any] = Field(default_factory=dict)
    """Provider-specific request fields for the decider. See
    :attr:`LLMSettings.extra`; kept separate so the decider and the extractor can
    run on gateways with different vocabularies.

    The field a Qwen endpoint needs lives here::

        EVEROS_DECIDER__EXTRA='
          {"extra_body": {"chat_template_kwargs": {"enable_thinking": false}}}'

    Without it a Qwen chat template leaves thinking ON, and an un-finetuned model
    spends the whole budget reasoning: measured on Qwen3.5-0.8B as decider, 39.3% of
    rounds returned nothing within the 60s deadline, each burning ~729s of retries
    before falling back to a fixed top-3 core. A finetuned policy does not show this
    because its targets were rendered with thinking off, so it emits the empty
    ``<think></think>`` and stops."""
    max_tokens: int | None = Field(default=512, ge=1)
    """Output cap for one decider reply. The contract is a single JSON object of a
    few dozen tokens, so this is a guard, not a budget: uncapped, a model that will
    not stop generates until the deadline and the round is lost to a timeout rather
    than to a parse failure that would have been visible."""
    temperature: float | None = Field(default=0.0, ge=0)
    """Sampling temperature of every decider call. ``None`` leaves it out of the
    request, so the endpoint's own default applies.

    0.0 because the decider's picks decide what the reader sees, and the training and
    evaluation code the decider policy was built with sends 0.0. Left out, a vLLM
    endpoint applies the model's ``generation_config.json`` -- for Qwen3.8-27B
    temperature 1.0, top_k 20, top_p 0.95 -- and searches stop being repeatable:
    measured 2026-11-10, byte-identical round-0 inputs reproduced the same picks only
    30-47% of the time. ``None`` is for an endpoint that rejects the field."""
    sdk_max_retries: int | None = Field(default=0, ge=0)
    """Retries inside the OpenAI SDK. 0 because :attr:`retries` below already retries
    at this layer, and the two multiply -- see the note in ``OpenAIProvider``. None
    keeps the SDK default.

    Bounded at zero: the SDK reads it as a count, so a negative value is not "fewer
    retries" but a misconfiguration that would reach ``AsyncOpenAI`` unexamined."""

    response_format_mode: ResponseFormatMode = "prompt_only"
    """How the decider's reply format is requested; never inherited from ``[llm]``.

    ``prompt_only`` because that is how the decider policy was trained and
    evaluated: the exp code asks for the JSON object in the prompt and sends no
    ``response_format``. The parity check of 2026-11-10 (deviation R1) found this
    tree sending a strict json_schema on every call instead, which a vLLM endpoint
    enforces by grammar-constraining the decider -- different decoding from the one
    its numbers were measured under. Inheriting ``[llm]``'s mode would bring that
    back on every deployment whose decider runs the extraction model, since
    ``[llm]`` defaults to ``json_schema``.

    ``json_object`` / ``json_schema`` remain an explicit opt-in for an endpoint
    that needs one. Either way the reply is parsed tolerantly, as the exp decider
    parses it, so there is no ``invalid_reply_retries`` here: an unusable reply
    is one failed attempt of :attr:`retries`."""

    # ── multi-round loop tuning ──────────────────────────────────────────
    max_rounds: int = Field(default=3, ge=1)
    """Hard cap on retrieval rounds. The cost bound: each round is one decider call
    plus one recall per sub-query."""
    seed_topk: int = Field(default=50, ge=1)
    """Round-0 block size -- how many candidates the original question contributes."""
    subq_topk: int = Field(default=20, ge=1)
    """Per-sub-query block size on round >= 1. Narrower than the seed on purpose:
    breadth comes from having several blocks, not from each being large."""
    max_subqueries: int = Field(default=3, ge=1)
    """Breadth cap -- gap-covering sub-queries the decider may issue per round."""
    rrf_k: int = Field(default=60, ge=1)
    """RRF smoothing constant for fusing each sub-query's sparse and dense lists."""
    no_new_core_patience: int = Field(default=1, ge=0)
    """Stop after this many consecutive rounds add no new core (after >= 1 round)."""
    per_subquery_guarantee: int = Field(default=1, ge=0)
    """Final-injection backfill: guarantee this many top non-core candidates per
    sub-query a slot, so every facet keeps coverage instead of being crowded out by
    one strong sub-query."""
    retries: int = Field(default=3, ge=0)
    """Retries on a transient decider failure before falling back to a fixed core."""
    retry_backoff_seconds: float = Field(default=0.5, ge=0)
    """Base delay between decider retries, doubled per attempt. Retrying a reasoning
    decider back-to-back tends to reproduce the same empty completion, so the pause is
    what makes the attempts meaningfully independent."""
    core_overflow: bool = False
    """Let the core-first stage exceed ``top_k`` (the pre-2026-08-06 behaviour).
    Measured at 316/1522 questions (20.8%) returning more items than asked for on
    SubtleMemory, up to 68 for ``top_k=20`` -- which breaks the ``top_k`` contract and
    invalidates any same-budget comparison. Set only to reproduce a pre-fix run."""
    fallback_core: int = Field(default=3, ge=0)
    """Core size to fall back to when every decider attempt fails.

    Not 0: an empty core silently disables core-first injection -- the very mechanism
    under test -- and is indistinguishable in the output from a decider that chose
    nothing on purpose."""
    fit_to_window: bool = True
    """Re-render a round prompt the decider endpoint rejects as too long. Default on.

    Off, a prompt over the decider's context window is rejected (HTTP 400) on every
    attempt and the round falls back (``decider_failed``). Measured 2026-11-10 on BEAM
    500K / 1M / 10M: up to 4.7% of questions for the RL decider, whose core grows to
    about 50 items that are re-rendered in full every round. On, and only after an
    attempt was rejected for its length, the prompt is re-rendered to fit
    ``window - max_tokens`` tokens as counted by the endpoint itself (vLLM's
    ``/tokenize``, through a client that can ask it): CORE SO FAR texts are truncated
    first, longest first under one shared cap, then candidate texts, then the last
    candidates are dropped. Deterministic; the round trace records what was trimmed
    (``decider_fit``). A prompt that fits is never touched, and an endpoint that cannot
    count leaves the round as it was, so a round that never overflows sends and records
    exactly what it did with the switch off. Set it ``false`` only to reproduce a run
    from before 2026-11-11."""
    salvage_truncated_core: bool = True
    """Keep the core indices of a decider reply that was cut short. Default on.

    Off, a reply such as ``{"core": [0, 3],`` (the model stopped after the core list)
    fails to parse on every attempt and the round falls back. On, and only after every
    attempt failed to parse, the complete indices of the first such reply become the
    round's core and the loop stops there, since no ``next_queries`` were emitted. The
    round trace records it (``decider_salvaged``). A round with a reply that parses is
    untouched. Set it ``false`` only to reproduce a run from before 2026-11-11."""
    image_candidates_per_round: int = Field(default=0, ge=0)
    """Pictures the decider may look at per round, highest-ranked first.

    Default 0: a decider that was never trained on picture candidates has no
    reason to be shown any, and every picture costs visual tokens on a call
    that runs inside the search request. Set it above 0 for a vision decider
    -- the pictures past the cap are still candidates, just rendered as text.
    """
    image_detail: Literal["low", "high", "auto"] = "low"
    """``detail`` hint on each picture part. ``low`` matches the 448px
    thumbnail: requesting ``high`` on a thumbnail only multiplies tokens."""
    image_thumbnail_edge: int = Field(default=448, ge=16)
    """Long side of the decider's copy, in pixels. Separate from
    ``[images] thumbnail_edge`` because the two answer to different owners:
    this one is the loop's own cost knob, the other is the store's."""
    image_quality: int = Field(default=85, ge=1, le=100)
    """JPEG quality of the decider's copy."""


class MultimodalSettings(_Section):
    """Multimodal parsing LLM config (everalgo-parser).

    Flat section mirroring ``[llm]``. The model must accept multimodal
    ``image_url`` parts (image / pdf / audio); it is kept independent from
    the main ``[llm]`` so parsing can target a vision/audio-capable
    endpoint without affecting boundary / extraction.

    Env binding (via parent ``Settings``):
        EVEROS_MULTIMODAL__MODEL
        EVEROS_MULTIMODAL__API_KEY
        EVEROS_MULTIMODAL__BASE_URL
        EVEROS_MULTIMODAL__MAX_CONCURRENCY
        EVEROS_MULTIMODAL__FILE_URI_ALLOW_DIRS
        EVEROS_MULTIMODAL__FILE_URI_MAX_BYTES
    """

    model: str = "google/gemini-3.8-flash"
    api_key: SecretStr | None = None
    base_url: str | None = None
    max_concurrency: int = 4
    response_format_mode: ResponseFormatMode = "json_schema"
    """How the reply format is requested. See
    :attr:`LLMSettings.response_format_mode`."""
    invalid_reply_retries: int = Field(default=1, ge=0)
    """Extra attempts after an unusable reply. See
    :attr:`LLMSettings.invalid_reply_retries`."""

    # ``file://`` content-item support (read locally by EverOS, not everalgo).
    file_uri_allow_dirs: list[str] = []
    """Allowlisted base dirs for ``file://`` uris. Empty = allow any readable
    file (local-first default); set to confine reads when the API is exposed."""
    file_uri_max_bytes: int = 50 * 1024 * 1024
    """Max size (bytes) of a ``file://`` asset; larger files are rejected."""


class ImagesSettings(_Section):
    """``[images]`` — how picture content items are stored and scaled.

    Pictures are content-addressed once per distinct file, so these limits
    bound *ingress*, not the store: the same picture arriving twice costs
    one file and one more reference.

    ``max_bytes`` bounds one picture after decoding the content item;
    ``max_per_message`` bounds how many a single message may carry (a
    message past it keeps its first N, and the rest are reported as
    rejected rather than silently dropped); ``max_request_bytes`` bounds
    the decoded total of one add request. ``max_per_request`` bounds the
    whole request the same way, because a hundred small pictures cost the
    extraction model a hundred image tokens each.

    ``thumbnail_edge`` and ``model_edge`` are the long-side caps of the two
    derived copies under ``figures/.cache/``. The decider sees the
    thumbnail; the extraction and answer models see the model copy. Both
    are derived, never the truth: they can be rebuilt from the original at
    any time, which is why nothing records their hashes.

    Env binding (via parent ``Settings``):
        EVEROS_IMAGES__MAX_BYTES / __MAX_PER_MESSAGE / __MAX_REQUEST_BYTES /
        __MAX_PER_REQUEST / __THUMBNAIL_EDGE / __MODEL_EDGE / __CACHE_QUALITY
    """

    max_bytes: int = Field(default=10 * 1024 * 1024, ge=1)
    """Largest accepted picture, after the content item is decoded."""
    max_per_message: int = Field(default=4, ge=0)
    """Pictures a single message may carry; the rest are rejected."""
    max_request_bytes: int = Field(default=32 * 1024 * 1024, ge=1)
    """Decoded picture bytes one add request may carry in total."""
    max_per_request: int = Field(default=16, ge=0)
    """Pictures one add request may carry in total; the rest are rejected."""
    thumbnail_edge: int = Field(default=448, ge=16)
    """Long-side cap of the decider's copy, in pixels. The spec pins 448:
    at patch 16 with 2x2 merging that is ~200 visual tokens per picture."""
    model_edge: int = Field(default=1024, ge=16)
    """Long-side cap of the copy handed to the extraction / answer model."""
    cache_quality: int = Field(default=85, ge=1, le=100)
    """JPEG quality of both derived copies."""
    formats: list[str] = Field(default_factory=lambda: ["png", "jpeg", "webp", "gif"])
    """Accepted picture formats, by the format ``Pillow`` reports."""


class ExtractionSettings(_Section):
    """``[extraction]`` — what the extraction model can read.

    ``vision`` says whether the model behind ``[llm]`` accepts ``image_url``
    content parts. With it on, picture content items are handed to that
    model directly and it writes each picture's memory itself. With it off,
    the picture is first turned into text by ``[multimodal]`` (the parser
    route) and the memory is written from that text -- for a text-only
    extraction model, at the cost of one extra call per picture.

    Env binding: EVEROS_EXTRACTION__VISION
    """

    vision: bool = True
    """Whether ``[llm]``'s model reads images. Default on: the 2.0
    extractor is a vision model, and the text-only route costs a call."""


class EmbeddingSettings(_Section):
    """Embedding client configuration.

    OpenAI-compatible embedding endpoint. ``model`` / ``api_key`` /
    ``base_url`` are required at runtime when the embedding capability
    is enabled; the runtime knobs (``timeout`` etc.) have sensible
    defaults.

    Env binding:
        EVEROS_EMBEDDING__MODEL
        EVEROS_EMBEDDING__API_KEY
        EVEROS_EMBEDDING__BASE_URL
        EVEROS_EMBEDDING__DIMENSIONS
        EVEROS_EMBEDDING__TIMEOUT_SECONDS
        EVEROS_EMBEDDING__MAX_RETRIES
        EVEROS_EMBEDDING__BATCH_SIZE
        EVEROS_EMBEDDING__MAX_CONCURRENT
        EVEROS_EMBEDDING__QUERY_MAX_CONCURRENT
    """

    model: str | None = None
    api_key: SecretStr | None = None
    base_url: str | None = None
    dimensions: int | None = None
    """API-level ``dimensions`` parameter for MRL-capable models.
    When set, passed to the embedding API so the server truncates
    with proper re-normalization. When ``None`` (default), the
    parameter is omitted and client-side truncation to ``dim``
    handles dimension alignment."""
    dim: int = Field(default=1024, ge=1)
    """Width of the ``vector`` column in every index table.

    The embedding model must return **at least** this many dimensions; a
    longer vector is truncated to this width, which only preserves recall
    for Matryoshka-trained models. A shorter one is refused at startup.
    Changing it invalidates every stored vector, so the index has to be
    rebuilt (``everos cascade rebuild``)."""
    timeout_seconds: float = Field(default=30.0, gt=0)
    max_retries: int = Field(default=3, ge=0)
    batch_size: int = Field(default=10, ge=1)
    max_concurrent: int = Field(default=10, ge=1)
    query_max_concurrent: int = Field(default=16, ge=1)
    """Concurrency budget reserved for interactive query embeds, separate
    from ``max_concurrent``, which the cascade's background writes draw on.

    Sharing one budget means a burst of writes queues ahead of the query a
    user is waiting on. Measured under three concurrent writers: vector
    search p50 48 s on the shared budget, against 0.49 s for the keyword
    leg, which needs no embedding at all (issue #473). Total in-flight
    requests to the endpoint are ``max_concurrent + query_max_concurrent``;
    lower this if the endpoint is rate-limited.

    Each embedding call holds one slot -- a multi-round search makes one
    for the question and one per sub-query of each later round -- so the
    sustainable call rate is about this value divided by the endpoint's
    per-call latency: at 4 and a remote endpoint answering in ~0.45 s, a
    search route making one call per request capped near 9 requests/s and a
    steady 10/s queued without bound (p50 64 s); at 16 the same load held
    p50 0.42 s, p95 1.45 s. Size it as embedding calls per second x per-call
    latency, with headroom."""


class VisionEmbeddingSettings(_Section):
    """``[vision_embedding]`` — the picture's own embedding endpoint.

    A separate section rather than another field on ``[embedding]``: the two
    have different models, endpoints and widths, and a deployment that runs
    text-only must be able to leave this section empty without touching the
    text configuration.

    The endpoint is an OpenAI-protocol ``/embeddings`` that accepts chat-style
    input, because a picture has to travel as an ``image_url`` content part,
    and it must apply the model's own input recipe (for
    ``Qwen3-VL-Embedding-*``: the system instruction and the generation
    prompt, pooled at the last token). Stock vLLM 0.32's chat-embedding path
    does not — it drops the prompt's last token and returns vectors with
    cosine 0.35–0.56 to the reference — so check a server against the model
    card's reference similarities before pointing this section at it.

    Env binding:
        EVEROS_VISION_EMBEDDING__MODEL
        EVEROS_VISION_EMBEDDING__API_KEY
        EVEROS_VISION_EMBEDDING__BASE_URL
        EVEROS_VISION_EMBEDDING__DIM
        EVEROS_VISION_EMBEDDING__TIMEOUT_SECONDS
        EVEROS_VISION_EMBEDDING__MAX_RETRIES
        EVEROS_VISION_EMBEDDING__BATCH_SIZE
        EVEROS_VISION_EMBEDDING__MAX_CONCURRENT
        EVEROS_VISION_EMBEDDING__QUERY_MAX_CONCURRENT
    """

    model: str | None = None
    api_key: SecretStr | None = None
    base_url: str | None = None
    dim: int = Field(default=4096, ge=1)
    """Width of the ``visual_vector`` column.

    Defaults to the native width of ``Qwen3-VL-Embedding-8B``;
    ``WeMM-Embedding-4B`` is 2560 and ``Qwen3-VL-Embedding-2B`` is 2048. A
    longer vector is cut to this width client-side, which only keeps recall
    for Matryoshka-trained models (``Qwen3-VL-Embedding`` is, 64–4096).
    Changing it invalidates every stored visual vector
    (``everos cascade rebuild``)."""
    timeout_seconds: float = Field(default=60.0, gt=0)
    """Longer than the text default: a picture is many more tokens than a
    sentence, and an 8B encoder over one 1024px image is not a 30 s call."""
    max_retries: int = Field(default=3, ge=0)
    batch_size: int = Field(default=8, ge=1)
    max_concurrent: int = Field(default=4, ge=1)
    query_max_concurrent: int = Field(default=16, ge=1)
    """Separate budget for the interactive query embed, as on ``[embedding]``."""


class RerankSettings(_Section):
    """Rerank client configuration.

    Unlike LLM / embedding (single OpenAI-compatible shape), rerank API
    schemas differ between providers — DeepInfra uses ``POST {base_url}/
    {model}`` with a custom body, vLLM uses ``POST {base_url}/rerank``
    with ``{model, query, documents}``. ``provider`` picks which client
    implementation the factory builds.

    Env binding:
        EVEROS_RERANK__PROVIDER
        EVEROS_RERANK__MODEL
        EVEROS_RERANK__API_KEY
        EVEROS_RERANK__BASE_URL
        EVEROS_RERANK__TIMEOUT_SECONDS
        EVEROS_RERANK__MAX_RETRIES
        EVEROS_RERANK__BATCH_SIZE
        EVEROS_RERANK__MAX_CONCURRENT
    """

    provider: Literal["deepinfra", "vllm", "dashscope"] = "deepinfra"
    model: str | None = None
    api_key: SecretStr | None = None
    base_url: str | None = None
    timeout_seconds: float = Field(default=30.0, gt=0)
    max_retries: int = Field(default=3, ge=0)
    batch_size: int = Field(default=10, ge=1)
    max_concurrent: int = Field(default=50, ge=1)


class BoundaryDetectionSettings(_Section):
    """Hard limits passed through to ``everalgo.boundary.adetect_boundaries``."""

    hard_token_limit: int = Field(default=65536, ge=1)
    hard_msg_limit: int = Field(default=500, ge=1)


class AsyncQueueSettings(_Section):
    """``async_mode`` memorize queue (``[memorize.async_queue]``).

    With ``enabled`` off, ``async_mode=true`` requests run synchronously; the
    worker still drains rows accepted earlier. Up to ``max_concurrency``
    batches are delivered at once, never two from one session: a session's
    batches run in arrival order, sessions run side by side, so one slow
    session does not hold up the others; ``1`` serialises everything. The
    three caps bound the queue: a session
    over ``max_pending_per_session`` or a process over ``max_pending_total``
    rejects new rows with 429, and a body over ``max_payload_bytes`` with
    413. A delivered row drops its body at once, so pending and delivered rows
    stay around ``max_pending_total * max_payload_bytes``; failed rows keep
    their bodies for the retention period and are not capped, and SQLite does
    not shrink the file after a purge. The caps are checked without a lock,
    so concurrent requests overshoot a little.

    A row that fails ``max_attempts`` times (``retry_backoff_seconds``
    apart) is marked ``failed`` and skipped so one bad batch cannot hold
    up its session. The default is a single attempt: ``memorize()`` is
    not atomic (boundary detection writes memcells and consumes the
    session buffer before extraction runs), so re-running a batch after a
    partial failure duplicates memcells or reports success with the
    episode missing. Raise it only where that is acceptable. A row that was
    in flight when the process died is requeued and replayed on the next
    start (at-least-once); one that has been replayed ``max_replays`` times
    without ever finishing is marked ``failed`` instead, so a batch that
    kills the process cannot crash-loop it forever. Every row, whatever its
    status, is deleted ``row_retention_hours`` (default seven days) after
    it was created.

    Env binding:
        EVEROS_MEMORIZE__ASYNC_QUEUE__ENABLED
        EVEROS_MEMORIZE__ASYNC_QUEUE__MAX_CONCURRENCY
        EVEROS_MEMORIZE__ASYNC_QUEUE__MAX_PENDING_PER_SESSION
        EVEROS_MEMORIZE__ASYNC_QUEUE__MAX_PENDING_TOTAL
        EVEROS_MEMORIZE__ASYNC_QUEUE__MAX_PAYLOAD_BYTES
        EVEROS_MEMORIZE__ASYNC_QUEUE__SCAN_INTERVAL_SECONDS
        EVEROS_MEMORIZE__ASYNC_QUEUE__MAX_ATTEMPTS
        EVEROS_MEMORIZE__ASYNC_QUEUE__RETRY_BACKOFF_SECONDS
        EVEROS_MEMORIZE__ASYNC_QUEUE__MAX_REPLAYS
        EVEROS_MEMORIZE__ASYNC_QUEUE__ROW_RETENTION_HOURS
    """

    enabled: bool = True
    max_concurrency: int = Field(default=4, ge=1)
    max_pending_per_session: int = Field(default=50, ge=1)
    max_pending_total: int = Field(default=200, ge=1)
    max_payload_bytes: int = Field(default=4 * 1024 * 1024, ge=1)
    scan_interval_seconds: float = Field(default=5.0, gt=0)
    max_attempts: int = Field(default=1, ge=1)
    retry_backoff_seconds: float = Field(default=2.0, ge=0)
    max_replays: int = Field(default=3, ge=1)
    row_retention_hours: float = Field(default=7 * 24.0, gt=0, le=366 * 24.0)


class MemorizeSettings(_Section):
    """Memorize use-case configuration.

    ``mode`` selects what boundary detection sees and which pipelines are
    dispatched. A service process serves one mode at a time; toggling
    requires a restart.

        - ``"chat"``  -> tool rows are dropped before boundary detection and
          only the user-memory pipeline runs.
        - ``"agent"`` -> tool rows stay in the cells (the model sees only the
          chat messages) and both user-memory + agent-memory pipelines run.

    ``session_lock_timeout_seconds`` caps how long one ``add()`` / ``flush()``
    invocation can hold the per-session lock. Covers boundary LLM call +
    memcell DB writes + (synchronous portion of) pipeline dispatch. Stops
    a stuck LLM from deadlocking subsequent concurrent calls on the same
    session_id: on timeout the outer ``asyncio.timeout`` cancels the task
    and the lock auto-releases.

    ``delete_wait_seconds`` caps how long a memory deletion of one session
    waits for that session's memorize lock (an ``add()`` in progress); past
    it the deletion answers 503 ``SESSION_BUSY`` and deletes nothing.

    Env binding:
        EVEROS_MEMORIZE__MODE
        EVEROS_MEMORIZE__SESSION_LOCK_TIMEOUT_SECONDS
        EVEROS_MEMORIZE__DELETE_WAIT_SECONDS
        EVEROS_MEMORIZE__ASYNC_QUEUE__*  (see :class:`AsyncQueueSettings`)
    """

    mode: Literal["chat", "agent"] = "agent"
    session_lock_timeout_seconds: float = Field(default=360.0, gt=0)
    delete_wait_seconds: float = Field(default=10.0, gt=0)
    async_queue: AsyncQueueSettings = AsyncQueueSettings()


class AgentMemorySettings(_Section):
    """Agent-memory extraction (``[agent_memory]``).

    ``case_gate_enabled`` puts everalgo's correctness gate in front of
    agent-case extraction: the extraction LLM judges whether the
    conversation's work was completed correctly, and only admitted
    conversations become cases (each admitted case then drives skill
    extraction whatever its quality score). It costs one extra LLM call per
    conversation up to the gate's length threshold. Off, cases are admitted
    by everalgo's complexity and value filters, as before. Needs a restart.

    Env binding:
        EVEROS_AGENT_MEMORY__CASE_GATE_ENABLED
    """

    case_gate_enabled: bool = False


class ClusteringSettings(_Section):
    """Geometry-clustering tunables.

    Env binding:
        EVEROS_CLUSTERING__THRESHOLD
        EVEROS_CLUSTERING__TIME_WINDOW_DAYS
    """

    threshold: float = Field(default=0.65, gt=0, le=1)
    time_window_days: float = Field(default=7.0, gt=0)


class LanceDBSettings(_Section):
    """LanceDB tunables.

    ``read_consistency_seconds``:
      ``None`` (omitted) → no consistency check (highest performance).
      ``0``              → strict consistency (every read).
      ``>0``             → eventual (interval between checks).

    ``index_cache_size_bytes``:
      Upper bound on LanceDB's global *index* cache (``GlobalIndexCache``
      in lance crate). Each cached entry is one opened FTS / vector /
      scalar index reader and **holds the file descriptors of its on-disk
      ``_indices/<uuid>/...`` files**.

      LanceDB's own default is ``None`` (unbounded), which on a long-
      running daemon means every new index UUID created by an
      ``optimize()`` call adds a fresh reader to the cache, and its
      FDs are never released — they leak monotonically until
      ``EMFILE`` (os error 24). Verified locally: 30 optimize cycles
      take FD usage from 0 to ~960 against macOS's default ``ulimit -n``
      of 256 / Linux's 1024.

      Setting a byte cap turns the cache into a real LRU: when it
      exceeds the cap, the oldest readers are dropped, Rust ``Drop``
      runs ``close(fd)``, and the FD pressure resolves itself.

      Cap → steady-state FD upper bound (measured under 30 add+optimize
      cycles with the real ``Episode`` schema and 100-query stress):

      ===========  =================  ===================
      cap          FD upper bound     query latency (100q)
      ===========  =================  ===================
      ``2 MB``     ~45                ~5 ms
      ``4 MB``     ~52                ~3 ms
      ``8 MB``     ~140               ~2.4 ms
      ``16 MB``    ~290               ~2.3 ms   ← default
      ``32 MB``    ~630               ~1.4 ms
      ``unbound``  >960 (leaks)       ~1.3 ms
      ===========  =================  ===================

      EverOS's measured steady-state working set after a 12 h
      ``rebuild_indexes`` cycle is ~50-100 readers / 3-6 MB resident
      (5 tables × ~7 BM25 columns × ~10 part_N entries each), so
      ``16 MB`` gives ~3× headroom for burst traffic and stale-but-not-
      yet-evicted readers, while the FD ceiling (~290) stays well below
      common ulimits (macOS default 256 needs ``ulimit -n 1024`` first;
      Linux default 1024 is fine out of the box).

      Override via ``EVEROS_LANCEDB__INDEX_CACHE_SIZE_BYTES`` if your
      working set is much larger (heavier table count or much wider
      indexes) or if you hit a tighter ``ulimit -n`` (containers / dev
      boxes).

    ``metadata_cache_size_bytes``:
      Upper bound on LanceDB's session metadata cache. An earlier note here
      said this cache was free of FD pressure and could stay unbounded. That
      was wrong, and it is the setting that actually decides whether a
      long-running writer's descriptor count converges.

      Lance keys these entries by dataset *version* (``ManifestKey{version}``),
      so every commit inserts new ones, and ``cleanup`` deleting the files
      underneath does not evict them — upstream ``lance-format/lance#7718``.
      Each surviving entry keeps its index reader, and the reader keeps the
      descriptors of its ``_indices/<uuid>/`` files, which POSIX will not
      release on unlink.

      **This setting chooses where the descriptor count converges.** It is
      not a capacity knob. Descriptors track surviving cache entries, not
      bytes, so the plateau lands at roughly
      ``cap / bytes-per-entry x handles-per-entry``: quadruple the cap and
      the plateau quadruples with it, it just takes proportionally longer
      to get there. Measured on 7 tables with the index cache at 16 MB,
      counting one descriptor per numeric ``lsof`` row:

      ===============  =============  ==========================
      metadata cap     converges at   reached by
      ===============  =============  ==========================
      ``1 MB``         ~216           round 31          ← default
      ``4 MB``         814            round 125
      unbounded        never          607 and climbing at 80
      ===============  =============  ==========================

      So size it from the descriptor ceiling you can afford, not from how
      much room the cache "needs". 1 MB leaves roughly 4x headroom under
      Linux's default ``ulimit -n`` of 1024. Raising it for a deployment
      whose manifests genuinely do not fit means raising ``ulimit -n`` by
      the same factor.

      Query latency was flat across every cap tried (BM25 2.0-2.9 ms,
      vector 3.9-4.5 ms at equal data volume), so the tight cap costs
      nothing measurable — the thrash this note used to predict did not
      appear. Not measured: a corpus of tens of GB per table, or object
      storage, where a miss is a network round trip rather than a local
      read.
    """

    read_consistency_seconds: float | None = None
    index_cache_size_bytes: int = 16 * 1024 * 1024
    metadata_cache_size_bytes: int = 1024 * 1024
    vector_index_min_rows: int = Field(default=2000, ge=1)
    """Rows (with a non-null vector) a table needs before its vector columns
    get an ANN index. Below this a brute-force scan is cheaper than the
    index; above it the scan grows linearly with the table — 27k rows of
    1024-dim vectors was 112 MB and ~0.6 s per query on a laptop SSD, and
    every round of a multi-round search runs several of them. Applied by the
    cascade worker only: its first rebuild sweep after server start builds a
    missing index and the heavy maintenance beat keeps it healthy. The CLI never builds
    one (it would race the running server's commits)."""
    max_concurrent_queries: int = Field(default=32, ge=1)
    """Search queries (keyword, vector, filtered scan) one process runs on
    LanceDB at once; the rest wait for a slot. One multi-round round fans out
    to up to five of them per sub-query (episode BM25 and vector, picture BM25,
    vector and visual). Uncapped, a burst opened every data file of a table once
    per query (981 descriptors at peak against a 1024 soft limit, 2026-11) and
    turned all their results into Python objects on the event loop together."""


class CascadeSettings(_Section):
    """Cascade on/off switches and maintenance cadences.

    ``enabled``:
      Whether the server runs the cascade at all (watcher + scanner + worker).
      Turned off, md edits never reach the index -- only useful for a run that
      does not search, or one that drives ``everos cascade sync`` itself.

    ``watcher_enabled``:
      Whether the filesystem watcher runs. The scanner already re-derives the
      same truth from the filesystem every ``scan_interval_seconds``, so the
      only thing lost is latency: an md edit is picked up within one scan
      instead of immediately. That trade is forced on a shared host, where
      inotify watches are a per-user kernel resource -- an IDE indexing the
      workspace can hold 350k of the 524k ceiling by itself, and every EverOS
      server then dies at startup with ``OSError: [Errno 28] inotify watch
      limit reached``. Unlike ``enabled = false`` this keeps the worker, so md
      still reaches the index -- which an ingesting run cannot do without.

    The evaluation harness's own names turn both off as well:
    ``EVEROS_DISABLE_CASCADE`` and ``EVEROS_DISABLE_CASCADE_WATCHER`` (see
    :class:`_ReadOnlySwitchSource`).

    The cadences below are *how often* each background job runs, not how long
    it is allowed to take — the deadlines that bound a hung call stay as
    constants next to the code they guard, sized from measurement, because a
    wrong value there either masks a hang or manufactures failures.

    ``optimize_heartbeat_seconds``:
      Idle sweep that offers every kind to the optimizer, so an unindexed tail
      left by a crash is merged even without new writes.

    ``optimize_prune_interval_seconds``:
      How often the heavy beat runs: reclaim the files of superseded dataset
      versions. Raise it if the write-lock hold is disruptive, lower it if disk
      transients are.

    ``optimize_prune_retention_seconds``:
      Passed straight to LanceDB as ``cleanup_older_than`` — versions replaced
      longer ago than this become eligible for deletion. It only has to outlive
      an in-flight read (sub-second). Shorter shrinks the transient footprint of
      superseded data fragments, but note it also decides how long index files
      keep a manifest that names them: below LanceDB's 7-day unverified window,
      index files lose that reference and wait out the full 7 days.

    ``optimize_rebuild_interval_seconds``:
      Full index rebuild per kind, which collapses the active index fragment
      count that every ``optimize()`` grows. Bounded by rebuild cost, not
      correctness — a missed sweep only defers cleanup.

    ``scan_interval_seconds``:
      How often the scanner walks the memory root for what the watcher
      missed: files written while the daemon was down, editors that
      move-replace, and mounts that deliver no filesystem events at all (a
      Windows directory bound into WSL2 or a container). On such a mount this
      is the only path an md edit takes to the index, so it is the
      edit-to-searchable latency. Every sweep also ``stat``-s every md file,
      which over a slow mount is the cost to weigh against it.

    ``delete_fence_warn_entries``:
      A memory deletion fences the index rows it removed while earlier
      cascade runs are still in flight, so they cannot write them back. The
      fence empties as those runs end; holding more entries than this logs
      ``memory_delete_fence_large``, a sign that some run is not ending.
    """

    enabled: bool = True
    watcher_enabled: bool = True
    optimize_heartbeat_seconds: float = 60.0
    optimize_prune_interval_seconds: float = 300.0
    optimize_prune_retention_seconds: float = 60.0
    optimize_rebuild_interval_seconds: float = 12 * 60 * 60.0
    scan_interval_seconds: float = Field(default=30.0, gt=0)
    delete_fence_warn_entries: int = Field(default=100_000, gt=0)


class OmeSettings(_Section):
    """Offline memory engine (OME) tunables.

    Engine-wide knobs only. Per-strategy overrides (enable / gate / cron) live
    in ``<memory-root>/ome.toml``, a separate file the engine reads itself.

    ``enabled``:
      Whether the server starts the engine. Off is for a server that only
      searches a store built elsewhere: no strategy runs, and a memorize on it
      fails when it publishes its events (``emit: engine not started``) -- so
      never on a server that ingests. The root lock is taken either way: one
      server per memory root. ``EVEROS_DISABLE_OME`` (the evaluation harness's
      name) turns it off too, as ``EVEROS_DISABLE_CASCADE`` does the cascade.

    ``max_concurrent_runs``:
      Engine-wide cap on concurrent strategy invocations. The default suits an
      interactive install; a bulk ingest of many conversations is throttled by
      it long before the machine is, since each slot spends almost all its
      time waiting on a remote extraction call rather than using CPU.

    ``run_timeout_seconds``:
      Wall-clock ceiling on one strategy attempt; ``0`` (or ``None``) disables
      it -- TOML has no null, so ``0`` is the operator's off switch.
      Without a ceiling, a coroutine parked on an await that carries no
      deadline of its own (a lock held by another stuck coroutine, a
      connection-pool wait) keeps its slot forever: it never raises, so it
      never retries. The 30-minute default is a deadlock backstop, not a
      latency target.
    """

    enabled: bool = True
    max_concurrent_runs: int = Field(default=20, gt=0)
    run_timeout_seconds: float | None = Field(default=1800.0, ge=0)

    @field_validator("run_timeout_seconds")
    @classmethod
    def _zero_disables(cls, value: float | None) -> float | None:
        return value or None


class IndexSettings(_Section):
    """Rebuildable derived-index backend selection."""

    backend: Literal["lancedb", "milvus"] = "lancedb"


class MilvusSettings(_Section):
    """Remote Milvus Server or Zilliz Cloud connection settings."""

    uri: str = ""
    token: SecretStr = SecretStr("")
    db_name: str = ""
    consistency_level: Literal["Strong", "Bounded", "Session", "Eventually"] = "Session"
    collection_prefix: str = Field(default="everos", min_length=1)

    @field_validator("collection_prefix")
    @classmethod
    def _validate_collection_prefix(cls, value: str) -> str:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
            raise ValueError(
                "collection_prefix must start with a letter or underscore and "
                "contain only letters, digits, and underscores"
            )
        return value


class KnowledgeSearchSettings(_Section):
    """``[knowledge.search]`` — retrieval tuning for the knowledge module."""

    recall_n: int = 200
    rerank_n: int = 50
    mass_top_m: int = 50
    lam: float = Field(0.1, alias="lambda")
    top_k_cap: int = 100

    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class KnowledgeSettings(_Section):
    """``[knowledge]`` — knowledge module configuration."""

    max_upload_bytes: int = 52_428_800  # 50 MiB
    search: KnowledgeSearchSettings = KnowledgeSearchSettings()


class GroupsSettings(_Section):
    """Group memory (``[groups]``, docs/group-memory.md sections 5.2 and 7.4).

    ``implicit_create``: a ``/memory/add`` naming an unknown ``group_id``
    creates the group (its first message's time is its creation time).
    ``implicit_join``: a message from a sender who is not a member opens an
    interval at that message (``source = implicit``); off, such a batch is
    refused with 422.
    ``member_profile_every``: a member's profile in a group is updated after
    this many new segments in which they spoke, and at every flush of the
    group; episodes are searchable at once, only the profile lags.
    ``scenes``: build the group's scenes from its episodes (section 7.7) --
    one narrative call each time an episode joins a scene of two or more.
    Off by default: the design builds group scenes on 2.0's per-owner
    scenes, which do not exist yet, so until they do group scenes are an
    opt-in (a group gets no record type users do not have, and ingestion
    makes no extra reflector calls).
    ``asker_in_question``: a group scope's multi-round question names the
    asker, ``"<display name>: <question>"`` (section 8.2).
    ``group_profile``: keep the group profile (section 7.5) -- topics,
    decisions, dissent and roles, one model call per batch of segments.
    ``group_profile_batch``: a batch is read after this many new segments;
    ``group_profile_idle_seconds``: or after the group has had no new segment
    for this long; and at every flush of the group.
    """

    implicit_create: bool = True
    implicit_join: bool = True
    member_profile_every: int = Field(default=5, ge=1)
    scenes: bool = False
    asker_in_question: bool = True
    group_profile: bool = True
    group_profile_batch: int = Field(default=5, ge=1)
    group_profile_idle_seconds: int = Field(default=600, ge=2)


class GroupSearchSettings(_Section):
    """``[search]`` -- what a group search adds around its ranking (sections 8.3-8.5).

    ``expansion``: after each round of a group's multi-round search, follow the
    stored links of the round's new core (decision chains, shared topics, the
    members it is about, a scene's episodes) and show the linked records in
    the next round. ``link_slots``: at most this many linked records join a
    round. ``constraints``: how the constraint block is built -- ``rules``
    from verified group-profile items and the core's metadata, or ``model``,
    which adds one call that states applicability and scope with exact
    quotes. ``group_profile_cap``: current group-profile items attached per
    kind (topics, decisions, dissent, roles).

    Env binding:
        EVEROS_SEARCH__EXPANSION
        EVEROS_SEARCH__LINK_SLOTS
        EVEROS_SEARCH__CONSTRAINTS
        EVEROS_SEARCH__GROUP_PROFILE_CAP
    """

    expansion: bool = True
    link_slots: int = Field(default=5, ge=0)
    constraints: Literal["rules", "model"] = "rules"
    group_profile_cap: int = Field(default=20, ge=1)


class AccessSettings(_Section):
    """Who is asking (``[access]``, docs/group-memory.md section 9.2).

    EverOS trusts its caller. A deployment that serves several people puts a
    gateway in front and names the header it sets here; the asker is then read
    from that header and a request whose body ``asker_id`` disagrees is
    refused (403). ``None`` (default): the body's ``asker_id`` is the asker.
    """

    identity_header: str | None = None


class LoggingSettings(_Section):
    """``[logging]`` — the process's console log output.

    ``everos server start --log-level`` and ``EVEROS_LOG_LEVEL`` override
    ``level`` for the server; interactive commands log ``WARNING`` (``INFO``
    with ``--verbose``) whatever it says.

    Env binding:
        EVEROS_LOGGING__LEVEL
    """

    level: str = "INFO"


class ObservabilitySettings(_Section):
    """``[observability]`` — OpenTelemetry tracing export.

    Off by default. When ``enabled`` is true a ``TracerProvider`` is built
    once at startup and standard OTLP/HTTP spans are exported to
    ``endpoint``. The signal is pure OpenTelemetry — vendor-neutral — so it
    works with any OTLP backend (Langfuse, an OTel Collector, ...); EverOS
    does not depend on any vendor SDK.

    ``langfuse_*`` are convenience credentials for pushing recall-quality
    *scores* to Langfuse (a Langfuse-specific REST call, independent of the
    OTLP span stream). Leave unset for a pure vendor-neutral OTLP export.

    Env binding:
        EVEROS_OBSERVABILITY__ENABLED
        EVEROS_OBSERVABILITY__EXPORTER
        EVEROS_OBSERVABILITY__ENDPOINT
        EVEROS_OBSERVABILITY__SERVICE_NAME
        EVEROS_OBSERVABILITY__SAMPLE_RATE
        EVEROS_OBSERVABILITY__LANGFUSE_PUBLIC_KEY / __LANGFUSE_SECRET_KEY
        EVEROS_OBSERVABILITY__LANGFUSE_HOST
        EVEROS_OBSERVABILITY__EMIT_RECALL_SCORES
    """

    enabled: bool = False
    exporter: Literal["otlp_http", "none"] = "otlp_http"
    endpoint: str = ""
    headers: dict[str, str] = Field(default_factory=dict)
    service_name: str = "everos"
    sample_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    # Privacy: when False (default) spans carry metadata only — no query text,
    # extracted memory, or .md paths. Set True to also emit request/response
    # content as span input/output (redacted + truncated).
    capture_content: bool = False

    # Langfuse scores (recall-quality feedback) — optional, Langfuse-specific.
    langfuse_public_key: str | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_host: str | None = None
    emit_recall_scores: bool = True


_READ_ONLY_SWITCHES: dict[str, tuple[str, str]] = {
    "EVEROS_DISABLE_CASCADE": ("cascade", "enabled"),
    "EVEROS_DISABLE_CASCADE_WATCHER": ("cascade", "watcher_enabled"),
    "EVEROS_DISABLE_OME": ("ome", "enabled"),
}
"""Each read-only switch the evaluation harness exports, and the setting it clears."""

_TRUTHY: frozenset[str] = frozenset({"1", "true", "yes"})


class _ReadOnlySwitchSource(PydanticBaseSettingsSource):
    """The evaluation harness's read-only switches, as the settings they stand for.

    The exp tree's evaluation harness (its ``benchmarks/run.py`` and the launch
    scripts built on it) starts a search-only server with
    ``EVEROS_DISABLE_CASCADE=1`` and ``EVEROS_DISABLE_OME=1``, and every server
    with ``EVEROS_DISABLE_CASCADE_WATCHER=1``. This tree had replaced them with
    ``[cascade] enabled`` / ``watcher_enabled`` and had no OME switch at all, so
    the same launch here silently started both write-side subsystems on a
    read-only server -- the cascade's periodic scan, which starved search on a
    dense store, and the OME engine.

    A switch is on exactly when the exp tree reads it as on (``1`` / ``true`` /
    ``yes``, case- and space-insensitive) and then sets its setting to
    ``false``; any other value, or none, leaves the setting to the other
    sources, so ``EVEROS_CASCADE__ENABLED`` and the TOML keep working. It sits
    just below init arguments: a switch exported for a run wins over a file or
    another environment variable, as the legacy ``EVEROS_LLMMR_*`` names do.
    """

    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:
        """Unused: :meth:`__call__` builds the whole mapping at once."""
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        cleared: dict[str, dict[str, bool]] = {}
        for env, (section, key) in _READ_ONLY_SWITCHES.items():
            if os.getenv(env, "").strip().lower() in _TRUTHY:
                cleared.setdefault(section, {})[key] = False
        return cleared


IGNORED_KEYS: Mapping[str, str] = {
    "decider.code_pin": (
        "a label the evaluation launchers export (EVEROS_DECIDER__CODE_PIN) to "
        "record which EverOS build ran; EverOS does not read it"
    ),
    "decider.full_text": "removed: the decider always reads the full episode text",
    "decider.invalid_reply_retries": (
        "removed: an unusable decider reply is one failed attempt of [decider] retries"
    ),
    "knowledge.search.session_lock_timeout_seconds": (
        "placed here by an older `everos init` template and never read here; "
        "the lock timeout is [memorize] session_lock_timeout_seconds"
    ),
    "llm.search_timeout_seconds": (
        "not read by EverOS 2; a decider round's deadline is [decider] timeout_seconds"
    ),
    "memory.root": (
        "removed in EverOS 1.x: the root is --root, EVEROS_ROOT or ~/.everos"
    ),
    "multimodal.max_tokens": (
        "never read: each [multimodal] call sets its own max_tokens; benchmark "
        "launchers still export EVEROS_MULTIMODAL__MAX_TOKENS"
    ),
    "observability.recall_hit_threshold": "removed with the search hit verdict",
    "search.facts_per_episode": "removed with atomic facts",
    "search.vector_strategy": "removed in EverOS 1.x with the vector search method",
}
"""Keys accepted and ignored, each logged once with why (``config_key_ignored``).

Every other unknown key fails the load. These are the keys EverOS itself once
wrote or documented and that configurations in use still carry: ``everos init``
copied some of them into the ``everos.toml`` of every store built before 2.0's
clean-up, and benchmark launchers export ``EVEROS_LLM__SEARCH_TIMEOUT_SECONDS``
and the decider code-pin label. A retired key is safe to delete wherever it
appears.
"""


def _model_of(annotation: Any) -> type[BaseModel] | None:
    """The section model a field holds, or ``None`` for a value field."""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    return None


def _key_names(model: type[BaseModel]) -> dict[str, str]:
    """Every spelling a key of ``model`` may take (aliases too) -> its field."""
    names: dict[str, str] = {}
    for name, field in model.model_fields.items():
        names[name] = name
        for alias in (field.alias, field.validation_alias):
            if isinstance(alias, str):
                names[alias] = name
    return names


def _unknown_keys(
    model: type[BaseModel], data: Mapping[str, Any], path: tuple[str, ...] = ()
) -> list[tuple[str, ...]]:
    """The paths in ``data`` that ``model`` and its sections do not declare.

    A free-form field (``[llm] extra``, ``[observability] headers``) is a value,
    so nothing under it is checked.
    """
    names = _key_names(model)
    unknown: list[tuple[str, ...]] = []
    for key, value in data.items():
        if key not in names:
            unknown.append((*path, str(key)))
            continue
        section = _model_of(model.model_fields[names[key]].annotation)
        if section is not None and isinstance(value, Mapping):
            unknown.extend(_unknown_keys(section, value, (*path, str(key))))
    return unknown


def _drop_ignored(data: dict[str, Any]) -> None:
    """Remove the :data:`IGNORED_KEYS` from ``data``, naming each one found."""
    for dotted, why in IGNORED_KEYS.items():
        *parents, key = dotted.split(".")
        node: Any = data
        for part in parents:
            node = node.get(part) if isinstance(node, Mapping) else None
        if isinstance(node, dict) and key in node:
            node.pop(key)
            logger.warning(
                "config_key_ignored",
                key=_label((*parents, key), table=False),
                source=_source_of((*parents, key)),
                reason=why,
            )


def _label(path: tuple[str, ...], *, table: bool) -> str:
    """``[llm] model`` for a key, ``[serch]`` for a section."""
    if table:
        return f"[{'.'.join(path)}]"
    if len(path) == 1:
        return path[0]
    return f"[{'.'.join(path[:-1])}] {path[-1]}"


def _source_of(path: tuple[str, ...]) -> str:
    """Where a key was set: its ``EVEROS_*`` variable, or the TOML file."""
    variable = "EVEROS_" + "__".join(path).upper()
    if any(name.upper() == variable for name in os.environ):
        return variable
    for toml in (resolve_root() / "everos.toml", _DEFAULT_TOML_PATH):
        if _toml_has(toml, path):
            return str(toml)
    return "the settings arguments"


def _toml_has(toml: Path, path: tuple[str, ...]) -> bool:
    try:
        node: Any = tomllib.loads(toml.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return False
    for part in path:
        if not isinstance(node, Mapping) or part not in node:
            return False
        node = node[part]
    return True


def _suggestion(path: tuple[str, ...]) -> str:
    """`` -- did you mean [llm] model?`` when a valid key is close."""
    model: type[BaseModel] | None = Settings
    for part in path[:-1]:
        names = _key_names(model) if model is not None else {}
        field = model.model_fields.get(names.get(part, "")) if model else None
        model = _model_of(field.annotation) if field is not None else None
    if model is None:
        return ""
    close = difflib.get_close_matches(path[-1], list(_key_names(model)), n=1)
    if not close:
        return ""
    candidate = (*path[:-1], close[0])
    field = model.model_fields.get(_key_names(model)[close[0]])
    table = field is not None and _model_of(field.annotation) is not None
    return f" -- did you mean {_label(candidate, table=table)}?"


def _unknown_message(data: Mapping[str, Any], unknown: list[tuple[str, ...]]) -> str:
    lines = [
        "Unknown configuration keys. EverOS rejects a key it does not know, so "
        "a misspelt setting cannot fall back to its default without a word:"
    ]
    for path in unknown:
        node: Any = data
        for part in path:
            node = node.get(part) if isinstance(node, Mapping) else None
        label = _label(path, table=isinstance(node, Mapping))
        lines.append(f"  {label} (from {_source_of(path)}){_suggestion(path)}")
    lines.append(
        "Correct or remove them; `everos config show` prints the effective "
        "configuration."
    )
    return "\n".join(lines)


class Settings(BaseSettings):
    """Top-level application settings.

    Strict: a key no section declares fails the load with
    :class:`~everos.core.errors.UnknownConfigKeyError`, except the retired
    keys in :data:`IGNORED_KEYS`, which are logged and dropped.
    """

    memory: MemorySettings = MemorySettings()
    api: ApiSettings = ApiSettings()
    sqlite: SqliteSettings = SqliteSettings()
    lancedb: LanceDBSettings = LanceDBSettings()
    index: IndexSettings = IndexSettings()
    milvus: MilvusSettings = MilvusSettings()
    llm: LLMSettings = LLMSettings()
    decider: DeciderSettings = DeciderSettings()
    embedding: EmbeddingSettings = EmbeddingSettings()
    vision_embedding: VisionEmbeddingSettings = VisionEmbeddingSettings()
    rerank: RerankSettings = RerankSettings()
    boundary_detection: BoundaryDetectionSettings = BoundaryDetectionSettings()
    memorize: MemorizeSettings = MemorizeSettings()
    agent_memory: AgentMemorySettings = AgentMemorySettings()
    clustering: ClusteringSettings = ClusteringSettings()
    cascade: CascadeSettings = CascadeSettings()
    ome: OmeSettings = OmeSettings()
    multimodal: MultimodalSettings = MultimodalSettings()
    images: ImagesSettings = ImagesSettings()
    extraction: ExtractionSettings = ExtractionSettings()
    knowledge: KnowledgeSettings = KnowledgeSettings()
    groups: GroupsSettings = GroupsSettings()
    search: GroupSearchSettings = GroupSearchSettings()
    access: AccessSettings = AccessSettings()
    logging: LoggingSettings = LoggingSettings()
    observability: ObservabilitySettings = ObservabilitySettings()

    model_config = SettingsConfigDict(
        env_prefix="EVEROS_",
        env_nested_delimiter="__",
        toml_file=_DEFAULT_TOML_PATH,
        extra="forbid",
    )

    @model_validator(mode="before")
    @classmethod
    def _known_keys_only(cls, data: Any) -> Any:
        """Drop the retired keys; refuse every other key no section declares.

        Runs on the merged sources (init arguments, the read-only switches,
        ``EVEROS_*`` variables, ``everos.toml``, ``default.toml``). Raises
        :class:`UnknownConfigKeyError` itself rather than a ``ValueError``, so
        the message is not wrapped in a validation error that would echo the
        whole configuration -- keys included.
        """
        if not isinstance(data, dict):
            return data
        _drop_ignored(data)
        unknown = _unknown_keys(cls, data)
        if unknown:
            # Deferred: everos.core.errors imports everos.config.
            from everos.core.errors import UnknownConfigKeyError

            raise UnknownConfigKeyError(_unknown_message(data, unknown))
        return data

    def __init__(self, *, _everos_root: Path | None = None, **kwargs: object) -> None:
        """Initialise settings, optionally pinning the memory-root for testing.

        Args:
            _everos_root: Override the memory root used to locate
                ``everos.toml``. Intended for tests only; pass ``None``
                (the default) in production to use :func:`resolve_root`.
            **kwargs: Forwarded verbatim to :class:`pydantic_settings.BaseSettings`.
        """
        if _everos_root is not None:
            # Temporarily inject EVEROS_ROOT so that settings_customise_sources
            # (a classmethod that cannot access instance state) picks it up via
            # resolve_root().  We restore the original value after super().__init__
            # returns to avoid leaking the override into the process environment.
            _prev = os.environ.get("EVEROS_ROOT")
            os.environ["EVEROS_ROOT"] = str(_everos_root)
            try:
                super().__init__(**kwargs)
            finally:
                if _prev is None:
                    os.environ.pop("EVEROS_ROOT", None)
                else:
                    os.environ["EVEROS_ROOT"] = _prev
        else:
            super().__init__(**kwargs)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Source order: init_args > read-only switches > env_vars > everos.toml >
        default.toml (see :class:`_ReadOnlySwitchSource` for the switches)."""
        sources: list[PydanticBaseSettingsSource] = [
            init_settings,
            _ReadOnlySwitchSource(settings_cls),
            env_settings,
        ]
        # Attempt to load <root>/everos.toml if it exists.
        everos_toml = resolve_root() / "everos.toml"
        if everos_toml.is_file():
            sources.append(
                TomlConfigSettingsSource(settings_cls, toml_file=everos_toml)
            )
        sources.append(TomlConfigSettingsSource(settings_cls))  # default.toml
        return tuple(sources)


@cache
def load_settings() -> Settings:
    """Load settings from default.toml + environment variables (cached).

    Cached at the module level — every caller sees the same instance until
    something explicitly clears the cache (``load_settings.cache_clear()``).
    Tests that monkeypatch environment variables must call
    ``cache_clear`` after each mutation to pick the new env up.
    """
    return Settings()
