"""Stable observation ranking, independent of classification threshold."""

from copy import deepcopy
from math import ceil, isfinite
from numbers import Integral, Real

from bank_marketing.schemas import DomainError


def _capacity(capacity: dict, n: int) -> int:
    if not isinstance(capacity, dict) or set(capacity) not in ({"k"}, {"q"}):
        raise DomainError("CAPACITY_INVALID", "Specify exactly one of k or q.")
    if "k" in capacity:
        k = capacity["k"]
        if isinstance(k, bool) or not isinstance(k, Integral) or not 0 <= k <= n:
            raise DomainError("CAPACITY_INVALID", "k must be an integer between 0 and batch size.")
        return int(k)
    q = capacity["q"]
    if isinstance(q, bool) or not isinstance(q, Real) or not isfinite(q) or not 0 <= q <= 1:
        raise DomainError("CAPACITY_INVALID", "q must be finite and between 0 and 1.")
    return ceil(q * n)


def rank_records(scores: dict, capacity: dict) -> dict:
    """Copy scores, sort p_yes descending then input_row_id ascending, select K."""
    if not isinstance(scores, dict) or not isinstance(scores.get("rows"), list):
        raise DomainError("SCHEMA_INVALID", "Scores must contain a rows list.")
    rows = scores["rows"]
    k = _capacity(capacity, len(rows))
    ids = set()
    for row in rows:
        if not isinstance(row, dict):
            raise DomainError("ROW_INVALID", "Each score row must be a mapping.")
        row_id, p = row.get("input_row_id"), row.get("p_yes")
        if isinstance(row_id, bool) or not isinstance(row_id, Integral) or row_id < 0 or row_id in ids:
            raise DomainError("ROW_INVALID", "input_row_id must be unique nonnegative integers.")
        if isinstance(p, bool) or not isinstance(p, Real) or not isfinite(p) or not 0 <= p <= 1:
            raise DomainError("ROW_INVALID", "p_yes must be finite and between 0 and 1.")
        ids.add(row_id)
    result = deepcopy(scores)
    result["rows"].sort(key=lambda row: (-row["p_yes"], row["input_row_id"]))
    for rank, row in enumerate(result["rows"], 1):
        row["rank"] = rank
        row["selected_for_contact"] = rank <= k
    result["capacity_k"] = k
    result["call_fraction"] = k / len(rows) if rows else 0.0
    return result
