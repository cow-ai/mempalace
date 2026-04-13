"""ChromaDB where-clause → Qdrant Filter translator.

Translates the ChromaDB ``where`` / ``where_document`` filter DSL into
``qdrant_client.models.Filter`` objects suitable for Qdrant searches
and deletes.

Supported metadata operators: ``$eq``, ``$ne``, ``$in``, ``$nin``,
``$and``, ``$or``, ``$gt``, ``$gte``, ``$lt``, ``$lte``.

``$contains`` is explicitly unsupported (Qdrant has no native substring
filter on payload fields) and raises
:class:`~mempalace.backends.base.UnsupportedFilterError`.
"""

import logging
from typing import Optional

from qdrant_client import models as qmodels

from .base import UnsupportedFilterError

logger = logging.getLogger(__name__)

# Operators accepted inside a field-level condition dict.
_FIELD_OPERATORS = frozenset(
    {"$eq", "$ne", "$in", "$nin", "$gt", "$gte", "$lt", "$lte", "$contains"}
)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def translate_where(where: Optional[dict]) -> Optional[qmodels.Filter]:
    """Translate a ChromaDB metadata *where* clause to a Qdrant ``Filter``.

    Returns ``None`` when *where* is ``None`` (no filter).

    Raises:
        UnsupportedFilterError: for ``$contains`` or unknown operators.
    """
    if where is None:
        return None
    return _translate_clause(where)


def translate_where_document(where: Optional[dict]) -> Optional[qmodels.Filter]:
    """Translate a ChromaDB *where_document* clause to a Qdrant ``Filter``.

    Returns ``None`` when *where* is ``None``.

    Raises:
        UnsupportedFilterError: ``$contains`` (and therefore all
            document-level filters) are unsupported by the Qdrant backend.
    """
    if where is None:
        return None
    _scan_for_contains(where)
    # If we somehow reach here (e.g. only $and/$or with no leaf operators),
    # return an empty filter that matches everything.
    return qmodels.Filter()


# ---------------------------------------------------------------------------
# Document-clause validation
# ---------------------------------------------------------------------------


def _scan_for_contains(clause: dict) -> None:
    """Walk the clause tree and raise on ``$contains``."""
    for key, value in clause.items():
        if key == "$contains":
            raise UnsupportedFilterError(
                "$contains is not supported by the Qdrant backend "
                "(Qdrant has no native substring filter)"
            )
        if key in ("$and", "$or") and isinstance(value, list):
            for sub in value:
                if isinstance(sub, dict):
                    _scan_for_contains(sub)


# ---------------------------------------------------------------------------
# Metadata-clause translation
# ---------------------------------------------------------------------------


def _translate_clause(clause: dict) -> qmodels.Filter:
    """Translate a single where-clause dict into a Qdrant ``Filter``."""
    must: list[qmodels.Condition] = []
    must_not: list[qmodels.Condition] = []

    for key, value in clause.items():
        if key == "$and":
            if not isinstance(value, list):
                raise UnsupportedFilterError("$and requires a list of clauses")
            sub_filters = [_translate_clause(sub) for sub in value]
            # All sub-filters must match → nest them in ``must``.
            must.extend(sub_filters)
        elif key == "$or":
            if not isinstance(value, list):
                raise UnsupportedFilterError("$or requires a list of clauses")
            sub_filters = [_translate_clause(sub) for sub in value]
            # At least one must match → use ``should``.
            # Return immediately: ``should`` semantics cannot be mixed with
            # sibling ``must`` entries in the same Filter.
            return qmodels.Filter(should=sub_filters)
        elif key.startswith("$"):
            raise UnsupportedFilterError(
                f"unknown operator {key!r} in metadata where clause"
            )
        else:
            # ``key`` is a metadata field name.
            field_filter = _translate_field(key, value)
            must.extend(field_filter.must)
            must_not.extend(field_filter.must_not)

    return qmodels.Filter(must=must, must_not=must_not)


def _translate_field(field: str, value: object) -> qmodels.Filter:
    """Translate a field-level condition into a Qdrant ``Filter``.

    Handles both the shorthand form (``{"field": "value"}`` → ``$eq``)
    and the explicit operator form (``{"field": {"$gt": 5}}``).
    """
    if not isinstance(value, dict):
        # Shorthand: ``{"field": "value"}`` is equivalent to ``$eq``.
        return qmodels.Filter(
            must=[qmodels.FieldCondition(key=field, match=qmodels.MatchValue(value=value))]
        )

    must: list[qmodels.Condition] = []
    must_not: list[qmodels.Condition] = []
    range_kwargs: dict[str, float] = {}
    seen_range = False

    for op, op_value in value.items():
        if op == "$contains":
            raise UnsupportedFilterError(
                "$contains is not supported by the Qdrant backend "
                "(Qdrant has no native substring filter)"
            )
        if op not in _FIELD_OPERATORS and op not in ("$and", "$or"):
            raise UnsupportedFilterError(
                f"operator {op!r} not supported by Qdrant backend"
            )
        if op in ("$and", "$or"):
            raise UnsupportedFilterError(
                f"logical operator {op!r} cannot appear inside a field condition"
            )

        if op == "$eq":
            must.append(
                qmodels.FieldCondition(key=field, match=qmodels.MatchValue(value=op_value))
            )
        elif op == "$ne":
            must_not.append(
                qmodels.FieldCondition(key=field, match=qmodels.MatchValue(value=op_value))
            )
        elif op == "$in":
            if not isinstance(op_value, list):
                raise UnsupportedFilterError("$in requires a list value")
            must.append(
                qmodels.FieldCondition(key=field, match=qmodels.MatchAny(any=op_value))
            )
        elif op == "$nin":
            if not isinstance(op_value, list):
                raise UnsupportedFilterError("$nin requires a list value")
            must_not.append(
                qmodels.FieldCondition(key=field, match=qmodels.MatchAny(any=op_value))
            )
        elif op in ("$gt", "$gte", "$lt", "$lte"):
            if not isinstance(op_value, (int, float)):
                raise UnsupportedFilterError(
                    f"range operator {op} requires a numeric value"
                )
            param_name = op[1:]  # "$gt" → "gt", "$gte" → "gte"
            range_kwargs[param_name] = float(op_value)
            seen_range = True

    if seen_range:
        must.append(qmodels.FieldCondition(key=field, range=qmodels.Range(**range_kwargs)))

    return qmodels.Filter(must=must, must_not=must_not)
