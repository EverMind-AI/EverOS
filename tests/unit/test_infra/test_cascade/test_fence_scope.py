"""A purge's scope entries in the delete fence: whole scopes, kept until lifted."""

from __future__ import annotations

import pytest

from everos.infra.cascade import DeleteFence


class _Row:
    def __init__(self, app_id: str, project_id: str, row_id: str = "x") -> None:
        self.app_id, self.project_id, self.id = app_id, project_id, row_id


# (blocked scope, row's scope) -> blocked?  Written out by hand.
CASES = [
    (("a1", "p1"), ("a1", "p1"), True),
    (("a1", "p1"), ("a1", "p2"), False),
    (("a1", "p1"), ("a2", "p1"), False),
    (("a1", None), ("a1", "p1"), True),
    (("a1", None), ("a1", "p2"), True),
    (("a1", None), ("a2", "p1"), False),
    (("*", None), ("a2", "p1"), True),
]


@pytest.mark.parametrize(("blocked", "row", "expected"), CASES)
def test_scope_entry_blocks_rows(
    blocked: tuple[str, str | None], row: tuple[str, str], expected: bool
) -> None:
    fence = DeleteFence()
    fence.block_scope(*blocked)
    assert fence.blocks(_Row(*row)) is expected
    fence.unblock_scope(*blocked)
    assert fence.blocks(_Row(*row)) is False


def test_scope_entry_survives_no_runs() -> None:
    fence = DeleteFence()
    fence.block_scope("a1", "p1")
    token = fence.register_run()
    fence.end_run(token)  # prunes row entries; a scope entry stays
    fence.add([("a2", "p1", "y")])  # no run in flight: a row entry is dropped
    assert fence.blocks(_Row("a1", "p1"))
    assert not fence.blocks(_Row("a2", "p1", "y"))


def test_in_flight_lists_registered_runs() -> None:
    fence = DeleteFence()
    first, second = fence.register_run(), fence.register_run()
    assert fence.in_flight() == {first, second}
    fence.end_run(first)
    assert fence.in_flight() == {second}
