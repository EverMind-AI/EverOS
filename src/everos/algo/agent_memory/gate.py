"""Length-aware admission of agent conversations before Case extraction.

The host injects a judge and owns its transport, endpoint, credentials, and deadline. Thresholds, input sizing,
admission, probability validation, and the question text belong here. A length bypass is permission to extract
experience, never evidence that a task succeeded.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from everos.algo.agent_memory.prompts.case_gate import (
    CASE_GATE_FALSE_CRITERION,
    CASE_GATE_INSTRUCTIONS,
    CASE_GATE_TRUE_CRITERION,
)
from everos.algo.llm import LLMOutputError
from everos.algo.utils import InvalidInputError, count_tokens

if TYPE_CHECKING:
    from everos.algo.types import AgentCase


class CaseGateQuestion(BaseModel):
    """Provider-neutral binary question the judge answers with a probability of ``true``.

    Args:
        instructions: What the judge assesses in the conversation.
        true_criterion: What a ``true`` answer means.
        false_criterion: What a ``false`` answer means.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    instructions: str
    true_criterion: str
    false_criterion: str


class CaseJudge(Protocol):
    """Probability reader supplied to :class:`AgentCaseGate`; the host maps the question onto its own provider."""

    async def apredict(self, conversation: str, question: CaseGateQuestion) -> float:
        """Return the probability that ``question`` is true for ``conversation``.

        Args:
            conversation: OpenAI-wire JSON of the untrimmed conversation.
            question: The question to answer, in provider-neutral form.

        Returns:
            A probability in ``[0, 1]``.
        """
        ...


class CaseGateConfig(BaseModel):
    """Algorithm settings; token sizing uses the untrimmed conversation under ``o200k_base``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    success_threshold: float = Field(default=0.70, ge=0, le=1, allow_inf_nan=False)
    bypass_token_threshold: int = Field(default=50_000, gt=0)


class CaseGateDecision(BaseModel):
    """Serializable provenance kept independently of the extractor's original quality score."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    policy: Literal["decision_case_v1"] = "decision_case_v1"
    method: Literal["decision_api", "length_bypass"]
    input_tokens: int = Field(ge=0)
    bypass_token_threshold: int = Field(gt=0)
    success_threshold: float = Field(ge=0, le=1, allow_inf_nan=False)
    success_probability: float | None = Field(
        default=None, ge=0, le=1, allow_inf_nan=False
    )

    @model_validator(mode="after")
    def _check_method(self) -> CaseGateDecision:
        if self.method == "length_bypass":
            if (
                self.input_tokens <= self.bypass_token_threshold
                or self.success_probability is not None
            ):
                raise InvalidInputError(
                    "length_bypass requires input above the token threshold and no probability"
                )
        elif (
            self.input_tokens > self.bypass_token_threshold
            or self.success_probability is None
        ):
            raise InvalidInputError(
                "decision_api requires input at or below the token threshold and a probability"
            )
        return self

    @property
    def accepted(self) -> bool:
        """Whether this decision admits the conversation for extraction."""
        return self.method == "length_bypass" or (
            self.success_probability is not None
            and self.success_probability >= self.success_threshold
        )


class AgentCaseGate:
    """Judge conversations up to the token threshold; admit longer conversations without a model call.

    Args:
        judge: Host-supplied probability reader.
        config: Admission thresholds; defaults to :class:`CaseGateConfig`.
    """

    def __init__(
        self, *, judge: CaseJudge, config: CaseGateConfig | None = None
    ) -> None:
        self._judge = judge
        self._config = config or CaseGateConfig()

    async def aevaluate(self, conversation: str) -> CaseGateDecision:
        """Evaluate original input before trimming; errors propagate and never become a probability of zero.

        Args:
            conversation: OpenAI-wire JSON of the untrimmed conversation.

        Returns:
            The admission decision with its provenance.

        Raises:
            LLMOutputError: The judge returned something other than a finite probability in ``[0, 1]``.
        """
        tokens = count_tokens(conversation)
        long_input = tokens > self._config.bypass_token_threshold
        probability = None if long_input else await self._apredict(conversation)
        return CaseGateDecision(
            method="length_bypass" if long_input else "decision_api",
            input_tokens=tokens,
            bypass_token_threshold=self._config.bypass_token_threshold,
            success_threshold=self._config.success_threshold,
            success_probability=probability,
        )

    async def _apredict(self, conversation: str) -> float:
        question = CaseGateQuestion(
            instructions=CASE_GATE_INSTRUCTIONS,
            true_criterion=CASE_GATE_TRUE_CRITERION,
            false_criterion=CASE_GATE_FALSE_CRITERION,
        )
        probability: object = await self._judge.apredict(conversation, question)
        if (
            isinstance(probability, bool)
            or not isinstance(probability, (int, float))
            or not math.isfinite(probability)
            or not 0 <= probability <= 1
        ):
            raise LLMOutputError(
                "case judge probability must be a finite number between zero and one"
            )
        return float(probability)


def admitted_case(case: AgentCase) -> CaseGateDecision | None:
    """Read persisted admission without treating absent or rejected provenance as an accepted source."""
    raw = getattr(case, "case_gate", None)
    if raw is None:
        return None
    decision = CaseGateDecision.model_validate(raw)
    if not decision.accepted:
        raise InvalidInputError("a stored case must not carry a rejected gate decision")
    return decision


def should_extract_skill(
    quality_score: float,
    *,
    case_gate: dict[str, Any] | None = None,
    skip_quality_threshold: float = 0.2,
) -> bool:
    """Keep quality thresholds inside algo; admitted cases continue regardless of the diagnostic quality score."""
    if case_gate is not None:
        decision = CaseGateDecision.model_validate(case_gate)
        if not decision.accepted:
            raise InvalidInputError(
                "a case must pass admission before skill extraction"
            )
        return True
    return quality_score >= skip_quality_threshold
