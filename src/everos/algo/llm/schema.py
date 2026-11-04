"""Strict-mode JSON Schema derivation for ``response_format`` models, and the request parameters built from it.

There is one ``to_<api>_response_format`` builder per mainstream API. Each translates the format: it returns what
that API expects in its request and leaves placing it there to the caller. Where an API takes only part of JSON
Schema, the other keywords move into the description, as the Anthropic SDK does; the reply is still validated
against the full model.
"""

from __future__ import annotations

__all__ = [
    "to_anthropic_response_format",
    "to_gemini_response_format",
    "to_ollama_response_format",
    "to_openai_response_format",
]

from typing import TYPE_CHECKING, Any, Final, Literal, cast

from pydantic import BaseModel, RootModel

if TYPE_CHECKING:
    from collections.abc import Callable

# What Anthropic structured outputs take, after the Anthropic SDK's ``transform_schema`` (anthropic 0.75.0
# ``lib/_parse/_transform.py``) and platform.claude.com/docs/en/build-with-claude/structured-outputs.
_ANTHROPIC_KEYWORDS: Final = frozenset(
    {
        "type",
        "properties",
        "required",
        "additionalProperties",
        "items",
        "enum",
        "const",
        "anyOf",
        "allOf",
        "$ref",
        "$defs",
        "definitions",
        "description",
        "title",
        "format",
        "minItems",
    }
)
_ANTHROPIC_STRING_FORMATS: Final = frozenset(
    {
        "date-time",
        "time",
        "date",
        "duration",
        "email",
        "hostname",
        "uri",
        "ipv4",
        "ipv6",
        "uuid",
    }
)

# What Gemini ``responseJsonSchema`` takes, as listed in google-genai 1.53.0 ``GenerateContentConfig``, less the
# ``oneOf`` the strict rewrite never leaves in a schema.
_GEMINI_KEYWORDS: Final = frozenset(
    {
        "$id",
        "$defs",
        "$ref",
        "$anchor",
        "type",
        "format",
        "title",
        "description",
        "enum",
        "items",
        "prefixItems",
        "minItems",
        "maxItems",
        "minimum",
        "maximum",
        "anyOf",
        "properties",
        "additionalProperties",
        "required",
        "propertyOrdering",
    }
)

_SUB_SCHEMA_MAPS: Final = frozenset({"properties", "$defs", "definitions"})
_SUB_SCHEMA_LISTS: Final = frozenset({"anyOf", "allOf", "prefixItems"})


def _to_strict_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Render ``model`` as the JSON Schema that OpenAI-style strict structured outputs accept.

    The result is ``model.model_json_schema()`` rewritten as openai SDK 2.35.1's ``to_strict_json_schema`` rewrites
    it: every object without a declared ``additionalProperties`` forbids extra keys and requires all its
    properties, a ``$ref`` with sibling keys is inlined, a single-arm ``allOf`` is flattened and ``None``
    defaults are dropped. Beyond the SDK, which passes them through, a discriminated union's ``oneOf`` becomes
    ``anyOf`` and its ``discriminator`` is dropped: strict mode rejects ``oneOf``, and ``discriminator`` is OpenAPI
    vocabulary no JSON Schema validator reads. A structure strict mode cannot express raises here rather than failing
    upstream with a 400.

    Args:
        model: The ``response_format`` model; its top level must be an object.

    Returns:
        The schema itself, without the ``{"type": "json_schema", ...}`` request envelope.

    Raises:
        TypeError: If ``model`` is a ``RootModel``.
        ValueError: If the schema lies outside the strict subset: a non-object top level, an open object, a
            dynamic-key mapping, a multi-arm ``allOf``, a ``oneOf`` beside an ``anyOf`` or a non-local ``$ref``. The
            message names the field path and, where there is one, the alternative to use.
    """
    if issubclass(model, RootModel):
        raise TypeError(
            f"{model.__name__}: a RootModel cannot be a response_format — its top level is not an object, so "
            f"a truncated reply may still parse; wrap the value in a field of a StrictOutputModel instead"
        )
    schema = model.model_json_schema()
    top_level_type = _top_level_type(schema, path=model.__name__)
    if top_level_type != "object":
        raise ValueError(
            f"{model.__name__}: the top level of a response_format schema must be an object, got "
            f"{top_level_type!r}; wrap the value in a field of a StrictOutputModel instead"
        )
    return _make_strict(schema, root=schema, path=model.__name__)


def to_openai_response_format(
    model: type[BaseModel],
    mode: Literal["json_schema", "json_object", "prompt_only"] = "json_schema",
) -> dict[str, Any] | None:
    """Build the value of the OpenAI Chat Completions ``response_format`` parameter for a reply shaped like ``model``.

    Serves any OpenAI-compatible endpoint. The caller places the result under ``response_format`` itself; nothing
    else in the request is touched.

    Args:
        model: The ``response_format`` model of the call.
        mode: ``json_schema`` asks for the strict schema of ``model``, ``json_object`` for JSON mode, and
            ``prompt_only`` for nothing beyond the prompt.

    Returns:
        The parameter value, or ``None`` when ``mode`` is ``prompt_only`` and the parameter is left out.

    Raises:
        TypeError: If ``mode`` is ``json_schema`` and ``model`` is a ``RootModel``.
        ValueError: If ``mode`` is ``json_schema`` and ``model`` lies outside the strict subset in any other way.
    """
    match mode:
        case "json_schema":
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": model.__name__,
                    "schema": _to_strict_json_schema(model),
                    "strict": True,
                },
            }
        case "json_object":
            return {"type": "json_object"}
        case "prompt_only":
            return None


def to_anthropic_response_format(
    model: type[BaseModel], mode: Literal["json_schema", "prompt_only"] = "json_schema"
) -> dict[str, Any] | None:
    """Build the value of the Anthropic Messages ``output_config.format`` parameter for a reply shaped like ``model``.

    The caller places the result under ``output_config.format`` itself, next to whatever else ``output_config`` holds
    (``effort``, for one). Anthropic has no JSON-mode level, so neither does ``mode``.

    As the Anthropic SDK's ``transform_schema`` does, the schema keeps only what Anthropic structured outputs take:
    other keywords (numeric bounds, string length, ``pattern``, a ``minItems`` above 1, an unlisted string format)
    move into the description, where the model can still read them.

    Args:
        model: The ``response_format`` model of the call.
        mode: ``json_schema`` asks for the strict schema of ``model``; ``prompt_only`` for nothing beyond the prompt.

    Returns:
        The parameter value, or ``None`` when ``mode`` is ``prompt_only`` and the parameter is left out.

    Raises:
        TypeError: If ``mode`` is ``json_schema`` and ``model`` is a ``RootModel``.
        ValueError: If ``mode`` is ``json_schema`` and ``model`` lies outside the strict subset in any other way.
    """
    match mode:
        case "json_schema":
            schema = _restrict(
                _to_strict_json_schema(model),
                keywords=_ANTHROPIC_KEYWORDS,
                keeps=_anthropic_keeps,
            )
            return {"type": "json_schema", "schema": schema}
        case "prompt_only":
            return None


def to_gemini_response_format(
    model: type[BaseModel],
    mode: Literal["json_schema", "json_object", "prompt_only"] = "json_schema",
) -> dict[str, Any]:
    """Build the ``generationConfig`` entries of the Gemini ``generateContent`` API for a reply shaped like ``model``.

    Gemini spreads this setting over two sibling fields, so both come back, under their REST names (the
    ``google-genai`` SDK accepts them as well). Add them to ``generationConfig`` with ``update``: no other entry
    there shares their keys. Keywords ``responseJsonSchema`` does not list (``pattern``, string length,
    ``multipleOf``, ``const`` and the like) move into the description.

    Args:
        model: The ``response_format`` model of the call.
        mode: ``json_schema`` asks for the strict schema of ``model``, ``json_object`` for JSON output without a
            schema, and ``prompt_only`` for nothing beyond the prompt.

    Returns:
        ``responseMimeType`` and, for ``json_schema``, ``responseJsonSchema``; empty for ``prompt_only``.

    Raises:
        TypeError: If ``mode`` is ``json_schema`` and ``model`` is a ``RootModel``.
        ValueError: If ``mode`` is ``json_schema`` and ``model`` lies outside the strict subset in any other way.
    """
    match mode:
        case "json_schema":
            schema = _restrict(_to_strict_json_schema(model), keywords=_GEMINI_KEYWORDS)
            return {
                "responseMimeType": "application/json",
                "responseJsonSchema": schema,
            }
        case "json_object":
            return {"responseMimeType": "application/json"}
        case "prompt_only":
            return {}


def to_ollama_response_format(
    model: type[BaseModel],
    mode: Literal["json_schema", "json_object", "prompt_only"] = "json_schema",
) -> dict[str, Any] | str | None:
    """Build the value of the Ollama ``format`` parameter (``/api/chat``, ``/api/generate``) for ``model``.

    Ollama takes the schema itself, with no envelope around it.

    Args:
        model: The ``response_format`` model of the call.
        mode: ``json_schema`` asks for the strict schema of ``model``, ``json_object`` for JSON mode, and
            ``prompt_only`` for nothing beyond the prompt.

    Returns:
        The schema, ``"json"`` for JSON mode, or ``None`` when ``mode`` is ``prompt_only`` and the parameter is left
        out.

    Raises:
        TypeError: If ``mode`` is ``json_schema`` and ``model`` is a ``RootModel``.
        ValueError: If ``mode`` is ``json_schema`` and ``model`` lies outside the strict subset in any other way.
    """
    match mode:
        case "json_schema":
            return _to_strict_json_schema(model)
        case "json_object":
            return "json"
        case "prompt_only":
            return None


def _top_level_type(schema: dict[str, Any], *, path: str) -> object:
    """Return the JSON type of the schema's top level, following the bare ``$ref`` a recursive model emits."""
    if "type" not in schema and "$ref" in schema:
        return _resolve_ref(root=schema, ref=schema["$ref"], path=path).get("type")
    return schema.get("type")


def _make_strict(value: object, *, root: dict[str, Any], path: str) -> dict[str, Any]:
    """Rewrite the schema node ``value`` in place for strict mode, recursing into every sub-schema.

    ``root`` is the whole schema, for resolving ``$ref``; ``path`` locates ``value`` in error messages.
    """
    node = _as_dict(value)
    if node is None:
        raise TypeError(
            f"{path}: expected a JSON Schema object, got {type(value).__name__}"
        )

    _check_additional_properties(node, path=path)
    _make_definitions_strict(node, root=root)
    if node.get("type") == "object" and "additionalProperties" not in node:
        node["additionalProperties"] = False
    _rewrite_discriminated_union(node, path=path)
    _make_children_strict(node, root=root, path=path)
    if "default" in node and node["default"] is None:
        # Nullability already lives in the type; a null default only adds a key strict mode rejects.
        del node["default"]
    return _inline_decorated_ref(node, root=root, path=path)


def _check_additional_properties(node: dict[str, Any], *, path: str) -> None:
    """Reject an ``additionalProperties`` other than ``false``; rewriting it would change what the model accepts."""
    if "additionalProperties" not in node:
        return
    additional = node["additionalProperties"]
    if additional is False:
        return
    if additional is True:
        raise ValueError(
            f"{path}: an open object (additionalProperties: true) is outside the strict subset — it comes from "
            f"extra='allow' or dict[str, Any]; inherit StrictOutputModel and declare every field explicitly"
        )
    raise ValueError(
        f"{path}: a dynamic-key mapping such as dict[str, X] is outside the strict subset — strict mode needs "
        f"every key listed as a property; use list[SubModel] with the key as one of SubModel's fields"
    )


def _rewrite_discriminated_union(node: dict[str, Any], *, path: str) -> None:
    """Rename the node's ``oneOf`` to ``anyOf`` and drop its ``discriminator``; reject a ``oneOf`` beside an ``anyOf``.

    pydantic emits ``oneOf`` only for a discriminated union. With a field-name discriminator each arm fixes that field
    to its own ``const``, so at most one arm matches and ``anyOf`` accepts the same replies; with a callable
    ``Discriminator`` the arms may overlap, and local validation against the model, which picks the arm by the
    discriminator, is what tells them apart.
    """
    node.pop("discriminator", None)
    if "oneOf" not in node:
        return
    if "anyOf" in node:
        raise ValueError(
            f"{path}: a oneOf beside an anyOf is outside the strict subset — strict mode takes anyOf only, and merging "
            f"the two would change what the node accepts; express the field as a single union"
        )
    node["anyOf"] = node.pop("oneOf")


def _make_definitions_strict(node: dict[str, Any], *, root: dict[str, Any]) -> None:
    """Rewrite every entry of the node's ``$defs`` / ``definitions`` sections."""
    for section in ("$defs", "definitions"):
        definitions = _as_dict(node.get(section))
        if definitions is not None:
            for name, definition in definitions.items():
                _make_strict(definition, root=root, path=name)


def _make_children_strict(
    node: dict[str, Any], *, root: dict[str, Any], path: str
) -> None:
    """Rewrite the node's properties, array items and ``anyOf`` / ``allOf`` arms; reject a multi-arm ``allOf``."""
    properties = _as_dict(node.get("properties"))
    if properties is not None:
        node["required"] = list(properties)
        node["properties"] = {
            key: _make_strict(child, root=root, path=f"{path}.{key}")
            for key, child in properties.items()
        }

    items = _as_dict(node.get("items"))
    if items is not None:
        node["items"] = _make_strict(items, root=root, path=f"{path}[]")

    any_of = _as_list(node.get("anyOf"))
    if any_of is not None:
        node["anyOf"] = [_make_strict(arm, root=root, path=path) for arm in any_of]

    all_of = _as_list(node.get("allOf"))
    if all_of is not None:
        if len(all_of) != 1:
            raise ValueError(
                f"{path}: an allOf with {len(all_of)} arms is outside the strict subset; express the field as "
                f"a single type or sub-model"
            )
        node.update(_make_strict(all_of[0], root=root, path=path))
        del node["allOf"]


def _inline_decorated_ref(
    node: dict[str, Any], *, root: dict[str, Any], path: str
) -> dict[str, Any]:
    """Inline a ``$ref`` that carries sibling keys, the node's own keys winning; a bare ``$ref`` is kept."""
    ref = node.get("$ref")
    if not ref or len(node) == 1:
        return node
    resolved = _resolve_ref(root=root, ref=ref, path=path)
    node.update({**resolved, **node})
    del node["$ref"]
    return _make_strict(node, root=root, path=path)


def _resolve_ref(*, root: dict[str, Any], ref: str, path: str) -> dict[str, Any]:
    """Return the schema a document-local pointer such as ``#/$defs/Inner`` names."""
    if not ref.startswith("#/"):
        raise ValueError(f"{path}: only document-local $ref is supported, got {ref!r}")
    resolved: Any = root
    for key in ref[2:].split("/"):
        resolved = resolved[key]
    schema = _as_dict(resolved)
    if schema is None:
        raise TypeError(f"{path}: $ref {ref!r} does not resolve to a schema object")
    return schema


def _restrict(
    schema: dict[str, Any],
    *,
    keywords: frozenset[str],
    keeps: Callable[[str, object], bool] = lambda _key, _value: True,
) -> dict[str, Any]:
    """Return ``schema`` holding only ``keywords`` at every level; the rest is appended to the description.

    ``keeps`` can still turn a listed keyword's value away. The note follows the Anthropic SDK's format:
    ``{key: value, ...}`` after a blank line.
    """
    kept: dict[str, Any] = {}
    moved: dict[str, Any] = {}
    for key, value in schema.items():
        if key in keywords and keeps(key, value):
            kept[key] = _restrict_children(key, value, keywords=keywords, keeps=keeps)
        else:
            moved[key] = value
    if moved:
        note = "{" + ", ".join(f"{key}: {value}" for key, value in moved.items()) + "}"
        kept["description"] = (
            f"{kept['description']}\n\n{note}" if "description" in kept else note
        )
    return kept


def _restrict_children(
    key: str,
    value: object,
    *,
    keywords: frozenset[str],
    keeps: Callable[[str, object], bool],
) -> object:
    """Apply ``_restrict`` to the sub-schemas ``value`` holds under ``key``, and return anything else unchanged."""
    if key in _SUB_SCHEMA_MAPS and (mapping := _as_dict(value)) is not None:
        return {
            name: _restrict_node(sub, keywords=keywords, keeps=keeps)
            for name, sub in mapping.items()
        }
    if key in _SUB_SCHEMA_LISTS and (arms := _as_list(value)) is not None:
        return [_restrict_node(arm, keywords=keywords, keeps=keeps) for arm in arms]
    if key == "items":
        return _restrict_node(value, keywords=keywords, keeps=keeps)
    return value


def _restrict_node(
    value: object, *, keywords: frozenset[str], keeps: Callable[[str, object], bool]
) -> object:
    node = _as_dict(value)
    return (
        _restrict(node, keywords=keywords, keeps=keeps) if node is not None else value
    )


def _anthropic_keeps(key: str, value: object) -> bool:
    """Whether Anthropic takes this value of a keyword it lists: ``minItems`` of 0 or 1, a listed string format."""
    if key == "minItems":
        return value in (0, 1)
    if key == "format":
        return value in _ANTHROPIC_STRING_FORMATS
    return True


def _as_dict(value: object) -> dict[str, Any] | None:
    """Return ``value`` as a schema mapping (JSON, so its keys are strings), or ``None`` if it is not a dict."""
    return cast("dict[str, Any]", value) if isinstance(value, dict) else None


def _as_list(value: object) -> list[object] | None:
    """Return ``value`` as a list of schema nodes, or ``None`` if it is not a list."""
    return cast("list[object]", value) if isinstance(value, list) else None
