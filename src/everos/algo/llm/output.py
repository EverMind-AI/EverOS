"""The base class for ``response_format`` models and the parsing of a reply against one."""

from __future__ import annotations

__all__ = ["StrictOutputModel", "describe_errors", "parse_reply"]


from pydantic import BaseModel, ConfigDict, ValidationError

from everos.algo.llm.errors import LLMOutputError


class StrictOutputModel(BaseModel):
    """Base for every ``response_format`` model and each of its sub-models; declare fields without defaults."""

    model_config = ConfigDict(strict=True, extra="forbid")


def parse_reply[OutputT: StrictOutputModel](
    model: type[OutputT], text: str, *, operator: str
) -> OutputT:
    """Validate a reply against the ``response_format`` model it was requested with.

    Args:
        model: The model the call passed to ``achat`` as ``response_format``.
        text: The reply text ``achat`` returned.
        operator: Names the call in the error message, e.g. ``"episode"`` or ``"boundary batch"``.

    Returns:
        The validated reply.

    Raises:
        LLMOutputError: If ``text`` is not valid JSON for ``model``, as ``"<operator> reply is not a valid <Model>:
            <errors>"`` with the errors as :func:`describe_errors` writes them. The reply's content never appears.
    """
    try:
        return model.model_validate_json(text)
    except ValidationError as exc:
        raise LLMOutputError(
            f"{operator} reply is not a valid {model.__name__}: {describe_errors(exc)}"
        ) from exc


def describe_errors(exc: ValidationError) -> str:
    """Describe each error as ``location: message``, joined by ``"; "``.

    The input values pydantic's own text quotes are left out, since they are the model's reply; a root-level error
    is located as ``<root>``.
    """
    return "; ".join(
        f"{'.'.join(str(part) for part in error['loc']) or '<root>'}: {error['msg']}"
        for error in exc.errors(include_url=False, include_input=False)
    )
