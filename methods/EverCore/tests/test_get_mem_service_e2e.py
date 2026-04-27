"""
V1 Memory GET API Test Suite
Tests POST /api/v1/memories/get with all filter, pagination, sorting, and error scenarios.

Prerequisites:
    - Server running: uv run python src/run.py --port 1995
    - Seed data inserted: python my_docs/temp/seed_v1_data.py

Usage:
    # Run all tests
    PYTHONPATH=src pytest tests/test_memory_get.py

    # Run a single test
    PYTHONPATH=src pytest tests/test_memory_get.py::TestEpisodicFilters::test_7a_filter_by_user_id

    # Run only episodic / profile / error / pagination groups
    PYTHONPATH=src pytest tests/test_memory_get.py::TestEpisodicFilters
    PYTHONPATH=src pytest tests/test_memory_get.py::TestProfileFilters
    PYTHONPATH=src pytest tests/test_memory_get.py::TestErrorCases
    PYTHONPATH=src pytest tests/test_memory_get.py::TestPaginationAndSorting

Seed data summary (see seed_v1_data.py header for full table):
    Episodic(10): user_01=5, user_02=3, user_03=2 | group_01=6, group_02=2, group_03=2
    Profiles(7):  user_01=3, user_02=2, user_03=2 | group_01=3, group_02=2, group_03=2
    Timestamps:   t_old(7d), t1(3h), t2(2h), t3(1h), t4(10min)
    Sessions:     session_01(group_01), session_02(group_02), session_03(group_01), session_04(group_03)
"""

import os
import logging

import pytest
import requests

logger = logging.getLogger(__name__)

BASE_URL = os.environ.get("TEST_BASE_URL", "http://localhost:1995")
API_URL = f"{BASE_URL}/api/v1/memories/get"
TIMEOUT = 30


def post_memories(payload: dict) -> requests.Response:
    """Send POST request to the memories/get endpoint."""
    return requests.post(
        API_URL,
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=TIMEOUT,
    )


def assert_success(resp: requests.Response) -> dict:
    """Assert HTTP 200 and return parsed JSON body."""
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    body = resp.json()
    assert "data" in body, f"Response missing 'data' key: {body}"
    return body


def assert_error(resp: requests.Response) -> dict:
    """Assert response contains an error object (status may vary).

    Accepts both custom error format {"error": {...}} and
    Pydantic validation format {"detail": [...]}.
    """
    body = resp.json()
    assert "error" in body or "detail" in body, f"Expected error response, got: {body}"
    return body


# ================================================================
# Episodic memory filter tests (7a - 7e, 7k)
# ================================================================
@pytest.mark.integration
class TestEpisodicFilters:
    """Test episodic memory filtering: user_id, group_id, session_id, in operator."""

    def test_7a_filter_by_user_id(self):
        """7a. user_01 should have 5 episodes across 3 groups."""
        resp = post_memories(
            {"memory_type": "episodic_memory", "filters": {"user_id": "user_01"}}
        )
        body = assert_success(resp)
        data = body["data"]

        assert data["total_count"] == 5
        assert data["count"] == 5
        assert len(data["episodes"]) == 5
        assert data["profiles"] == []

        user_ids = {ep["user_id"] for ep in data["episodes"]}
        assert user_ids == {"user_01"}

        subjects = {ep["subject"] for ep in data["episodes"]}
        expected_subjects = {
            "Search API Design Discussion",
            "Dashboard Design Review",
            "Sprint Planning",
            "Project Kickoff Meeting",
            "Security Audit Review",
        }
        assert subjects == expected_subjects

    def test_7b_filter_by_user_id_and_group_id(self):
        """7b. user_01 + group_01 should have 3 episodes."""
        resp = post_memories(
            {
                "memory_type": "episodic_memory",
                "filters": {"user_id": "user_01", "group_id": "group_01"},
            }
        )
        body = assert_success(resp)
        data = body["data"]

        assert data["total_count"] == 3
        for ep in data["episodes"]:
            assert ep["user_id"] == "user_01"
            assert ep["group_id"] == "group_01"

        subjects = {ep["subject"] for ep in data["episodes"]}
        assert subjects == {
            "Search API Design Discussion",
            "Sprint Planning",
            "Project Kickoff Meeting",
        }

    def test_7c_filter_by_group_id(self):
        """7c. group_01 should have 6 episodes from 3 users."""
        resp = post_memories(
            {"memory_type": "episodic_memory", "filters": {"group_id": "group_01"}}
        )
        body = assert_success(resp)
        data = body["data"]

        assert data["total_count"] == 6
        user_ids = {ep["user_id"] for ep in data["episodes"]}
        assert user_ids == {"user_01", "user_02", "user_03"}

        subjects = {ep["subject"] for ep in data["episodes"]}
        assert "Search API Design Discussion" in subjects
        assert "Sprint Planning" in subjects
        assert "Project Kickoff Meeting" in subjects

    def test_7d_filter_by_user_id_and_session_id(self):
        """7d. user_01 + session_01 should have 2 episodes."""
        resp = post_memories(
            {
                "memory_type": "episodic_memory",
                "filters": {"user_id": "user_01", "session_id": "session_01"},
            }
        )
        body = assert_success(resp)
        data = body["data"]

        assert data["total_count"] == 2
        for ep in data["episodes"]:
            assert ep["session_id"] == "session_01"

        subjects = {ep["subject"] for ep in data["episodes"]}
        assert subjects == {"Search API Design Discussion", "Project Kickoff Meeting"}

    def test_7e_group_id_in_operator(self):
        """7e. group_id in [group_01, group_02] should have 8 episodes."""
        resp = post_memories(
            {
                "memory_type": "episodic_memory",
                "filters": {"group_id": {"in": ["group_01", "group_02"]}},
            }
        )
        body = assert_success(resp)
        data = body["data"]

        assert data["total_count"] == 8
        group_ids = {ep["group_id"] for ep in data["episodes"]}
        assert group_ids == {"group_01", "group_02"}

    def test_7k_no_matching_data(self):
        """7k. Non-existent user should return empty result."""
        resp = post_memories(
            {
                "memory_type": "episodic_memory",
                "filters": {"user_id": "user_nonexistent"},
            }
        )
        body = assert_success(resp)
        data = body["data"]

        assert data["total_count"] == 0
        assert data["count"] == 0
        assert data["episodes"] == []

    def test_7s_session_id_in_operator(self):
        """7s. group_id in + session_id in should have 6 episodes."""
        resp = post_memories(
            {
                "memory_type": "episodic_memory",
                "filters": {
                    "group_id": {"in": ["group_01", "group_02"]},
                    "session_id": {"in": ["session_01", "session_02"]},
                },
            }
        )
        body = assert_success(resp)
        data = body["data"]

        assert data["total_count"] == 6
        session_ids = {ep["session_id"] for ep in data["episodes"]}
        assert "session_03" not in session_ids

    def test_user_id_in_operator(self):
        """user_id in [user_01, user_02] should return 8 episodes (5+3)."""
        resp = post_memories(
            {
                "memory_type": "episodic_memory",
                "filters": {"user_id": {"in": ["user_01", "user_02"]}},
            }
        )
        body = assert_success(resp)
        data = body["data"]

        assert data["total_count"] == 8
        user_ids = {ep["user_id"] for ep in data["episodes"]}
        assert user_ids == {"user_01", "user_02"}

    def test_user_id_eq_and_group_id_in(self):
        """user_01 + group_id in [group_01, group_02] should return 4 episodes (3+1)."""
        resp = post_memories(
            {
                "memory_type": "episodic_memory",
                "filters": {
                    "user_id": "user_01",
                    "group_id": {"in": ["group_01", "group_02"]},
                },
            }
        )
        body = assert_success(resp)
        data = body["data"]

        assert data["total_count"] == 4
        for ep in data["episodes"]:
            assert ep["user_id"] == "user_01"
        group_ids = {ep["group_id"] for ep in data["episodes"]}
        assert group_ids == {"group_01", "group_02"}

    def test_user_id_eq_and_session_id_in(self):
        """user_01 + session_id in [session_01, session_03] should return 3 episodes (2+1)."""
        resp = post_memories(
            {
                "memory_type": "episodic_memory",
                "filters": {
                    "user_id": "user_01",
                    "session_id": {"in": ["session_01", "session_03"]},
                },
            }
        )
        body = assert_success(resp)
        data = body["data"]

        assert data["total_count"] == 3
        session_ids = {ep["session_id"] for ep in data["episodes"]}
        assert session_ids == {"session_01", "session_03"}

    def test_group_id_eq_and_session_id_eq(self):
        """group_01 + session_01 should return 4 episodes (ep_01,02,06,07)."""
        resp = post_memories(
            {
                "memory_type": "episodic_memory",
                "filters": {"group_id": "group_01", "session_id": "session_01"},
            }
        )
        body = assert_success(resp)
        data = body["data"]

        assert data["total_count"] == 4
        for ep in data["episodes"]:
            assert ep["group_id"] == "group_01"
            assert ep["session_id"] == "session_01"

    def test_or_combinator(self):
        """user_01 + OR[group_01, group_03] should return 4 episodes (3+1)."""
        resp = post_memories(
            {
                "memory_type": "episodic_memory",
                "filters": {
                    "user_id": "user_01",
                    "OR": [{"group_id": "group_01"}, {"group_id": "group_03"}],
                },
            }
        )
        body = assert_success(resp)
        data = body["data"]

        assert data["total_count"] == 4
        for ep in data["episodes"]:
            assert ep["user_id"] == "user_01"
        group_ids = {ep["group_id"] for ep in data["episodes"]}
        assert group_ids == {"group_01", "group_03"}


# ================================================================
# Pagination and sorting tests (7f - 7j, 7t)
# ================================================================
@pytest.mark.integration
class TestPaginationAndSorting:
    """Test pagination (page/page_size) and sorting (rank_by/rank_order)."""

    def test_7f_pagination_page1(self):
        """7f. page=1, page_size=2 for user_01: 2 of 5 episodes."""
        resp = post_memories(
            {
                "memory_type": "episodic_memory",
                "page": 1,
                "page_size": 2,
                "filters": {"user_id": "user_01"},
            }
        )
        body = assert_success(resp)
        data = body["data"]

        assert data["total_count"] == 5
        assert data["count"] == 2
        assert len(data["episodes"]) == 2

    def test_7g_pagination_last_page(self):
        """7g. page=3, page_size=2 for user_01: last page with 1 episode."""
        resp = post_memories(
            {
                "memory_type": "episodic_memory",
                "page": 3,
                "page_size": 2,
                "filters": {"user_id": "user_01"},
            }
        )
        body = assert_success(resp)
        data = body["data"]

        assert data["total_count"] == 5
        assert data["count"] == 1
        assert len(data["episodes"]) == 1

    def test_7f_7g_pagination_no_overlap(self):
        """7f+7g combined: page 1 and page 3 should not overlap."""
        resp1 = post_memories(
            {
                "memory_type": "episodic_memory",
                "page": 1,
                "page_size": 2,
                "filters": {"user_id": "user_01"},
            }
        )
        resp3 = post_memories(
            {
                "memory_type": "episodic_memory",
                "page": 3,
                "page_size": 2,
                "filters": {"user_id": "user_01"},
            }
        )
        ids_page1 = {ep["id"] for ep in resp1.json()["data"]["episodes"]}
        ids_page3 = {ep["id"] for ep in resp3.json()["data"]["episodes"]}
        assert ids_page1.isdisjoint(ids_page3), "Pages should not overlap"

    def test_7h_sort_asc(self):
        """7h. rank_order=asc: oldest first (t_old -> t1 -> t2 -> t3 -> t4)."""
        resp = post_memories(
            {
                "memory_type": "episodic_memory",
                "rank_by": "timestamp",
                "rank_order": "asc",
                "filters": {"user_id": "user_01"},
            }
        )
        body = assert_success(resp)
        episodes = body["data"]["episodes"]

        expected_order = [
            "Project Kickoff Meeting",
            "Search API Design Discussion",
            "Dashboard Design Review",
            "Sprint Planning",
            "Security Audit Review",
        ]
        actual_order = [ep["subject"] for ep in episodes]
        assert actual_order == expected_order, f"ASC order mismatch: {actual_order}"

    def test_7i_sort_desc(self):
        """7i. rank_order=desc: newest first (t4 -> t3 -> t2 -> t1 -> t_old)."""
        resp = post_memories(
            {
                "memory_type": "episodic_memory",
                "rank_order": "desc",
                "filters": {"user_id": "user_01"},
            }
        )
        body = assert_success(resp)
        episodes = body["data"]["episodes"]

        expected_order = [
            "Security Audit Review",
            "Sprint Planning",
            "Dashboard Design Review",
            "Search API Design Discussion",
            "Project Kickoff Meeting",
        ]
        actual_order = [ep["subject"] for ep in episodes]
        assert actual_order == expected_order, f"DESC order mismatch: {actual_order}"

    def test_7j_timestamp_range_with_and(self):
        """7j. Timestamp range captures only t2 (2026-01-15T10:00:00Z) episode."""
        # Fixed: t2 = 2026-01-15T10:00:00Z, window = 09:30 ~ 10:30
        t_gte = 1768469400000  # 2026-01-15T09:30:00Z in millis
        t_lt = 1768473000000  # 2026-01-15T10:30:00Z in millis

        resp = post_memories(
            {
                "memory_type": "episodic_memory",
                "filters": {
                    "user_id": "user_01",
                    "AND": [{"timestamp": {"gte": t_gte}}, {"timestamp": {"lt": t_lt}}],
                },
            }
        )
        body = assert_success(resp)
        data = body["data"]

        assert data["total_count"] == 1
        assert data["episodes"][0]["subject"] == "Dashboard Design Review"

    def test_7t_page_beyond_data(self):
        """7t. page=99 should return empty episodes but correct total_count."""
        resp = post_memories(
            {
                "memory_type": "episodic_memory",
                "page": 99,
                "page_size": 20,
                "filters": {"user_id": "user_01"},
            }
        )
        body = assert_success(resp)
        data = body["data"]

        assert data["total_count"] == 5
        assert data["count"] == 0
        assert data["episodes"] == []


# ================================================================
# Profile filter tests (7l - 7n)
# ================================================================
@pytest.mark.integration
class TestProfileFilters:
    """Test profile memory filtering."""

    def test_7l_profile_by_user_id(self):
        """7l. user_01 should have 3 profiles across 3 groups."""
        resp = post_memories(
            {"memory_type": "profile", "filters": {"user_id": "user_01"}}
        )
        body = assert_success(resp)
        data = body["data"]

        assert data["total_count"] == 3
        assert data["count"] == 3
        assert len(data["profiles"]) == 3
        assert data["episodes"] == []

        for pf in data["profiles"]:
            assert pf["user_id"] == "user_01"

        roles = [pf["profile_data"]["explicit_info"]["Role"] for pf in data["profiles"]]
        roles_text = " | ".join(roles)
        assert "Tech Lead" in roles_text
        assert "Frontend Lead" in roles_text
        assert "Security Lead" in roles_text

    def test_7m_profile_by_user_id_and_group_id(self):
        """7m. user_01 + group_01 should have 1 profile (Tech Lead)."""
        resp = post_memories(
            {
                "memory_type": "profile",
                "filters": {"user_id": "user_01", "group_id": "group_01"},
            }
        )
        body = assert_success(resp)
        data = body["data"]

        assert data["total_count"] == 1
        pf = data["profiles"][0]
        assert pf["user_id"] == "user_01"
        assert pf["group_id"] == "group_01"
        assert "Tech Lead" in pf["profile_data"]["explicit_info"]["Role"]
        assert pf["scenario"] == "team"

    def test_7n_profile_by_group_id(self):
        """7n. group_01 should have 3 profiles (user_01, user_02, user_03)."""
        resp = post_memories(
            {"memory_type": "profile", "filters": {"group_id": "group_01"}}
        )
        body = assert_success(resp)
        data = body["data"]

        assert data["total_count"] == 3
        user_ids = {pf["user_id"] for pf in data["profiles"]}
        assert user_ids == {"user_01", "user_02", "user_03"}

    def test_profile_group_id_in(self):
        """group_id in [group_01, group_02] should return 5 profiles (3+2)."""
        resp = post_memories(
            {
                "memory_type": "profile",
                "filters": {"group_id": {"in": ["group_01", "group_02"]}},
            }
        )
        body = assert_success(resp)
        data = body["data"]

        assert data["total_count"] == 5
        group_ids = {pf["group_id"] for pf in data["profiles"]}
        assert group_ids == {"group_01", "group_02"}

    def test_profile_user_id_in(self):
        """user_id in [user_01, user_02] should return 5 profiles (3+2)."""
        resp = post_memories(
            {
                "memory_type": "profile",
                "filters": {"user_id": {"in": ["user_01", "user_02"]}},
            }
        )
        body = assert_success(resp)
        data = body["data"]

        assert data["total_count"] == 5
        user_ids = {pf["user_id"] for pf in data["profiles"]}
        assert user_ids == {"user_01", "user_02"}


# ================================================================
# Error case tests (7o - 7r)
# ================================================================
@pytest.mark.integration
class TestErrorCases:
    """Test validation error responses."""

    def test_7o_missing_scope(self):
        """7o. Empty filters (no user_id or group_id) should return error."""
        resp = post_memories({"memory_type": "episodic_memory", "filters": {}})
        body = assert_error(resp)
        if "error" in body:
            error = body["error"]
            assert error["code"] == "InvalidParameter"
            msg = error["message"].lower()
        else:
            # Pydantic validation error format
            msg = body["detail"][0]["msg"].lower()
        assert "user_id" in msg or "group_id" in msg

    def test_7p_missing_memory_type(self):
        """7p. Missing memory_type should return error."""
        resp = post_memories({"filters": {"user_id": "user_01"}})
        body = resp.json()
        # Could be error object or validation error depending on framework
        assert resp.status_code != 200 or "error" in body

    def test_7q_invalid_memory_type(self):
        """7q. Invalid memory_type value should return error."""
        resp = post_memories(
            {"memory_type": "invalid_type", "filters": {"user_id": "user_01"}}
        )
        body = resp.json()
        assert resp.status_code != 200 or "error" in body

    def test_7r_missing_filters(self):
        """7r. Missing filters key should return error."""
        resp = post_memories({"memory_type": "episodic_memory"})
        body = resp.json()
        assert resp.status_code != 200 or "error" in body


# ================================================================
# Cross-validation / additional coverage
# ================================================================
@pytest.mark.integration
class TestAdditionalCoverage:
    """Extra tests for completeness beyond the spec 7a-7t."""

    def test_user_02_episodic_count(self):
        """user_02 should have 3 episodes."""
        resp = post_memories(
            {"memory_type": "episodic_memory", "filters": {"user_id": "user_02"}}
        )
        data = assert_success(resp)["data"]
        assert data["total_count"] == 3

    def test_user_03_episodic_count(self):
        """user_03 should have 2 episodes."""
        resp = post_memories(
            {"memory_type": "episodic_memory", "filters": {"user_id": "user_03"}}
        )
        data = assert_success(resp)["data"]
        assert data["total_count"] == 2

    def test_group_02_episodic_count(self):
        """group_02 should have 2 episodes."""
        resp = post_memories(
            {"memory_type": "episodic_memory", "filters": {"group_id": "group_02"}}
        )
        data = assert_success(resp)["data"]
        assert data["total_count"] == 2

    def test_group_03_episodic_count(self):
        """group_03 should have 2 episodes."""
        resp = post_memories(
            {"memory_type": "episodic_memory", "filters": {"group_id": "group_03"}}
        )
        data = assert_success(resp)["data"]
        assert data["total_count"] == 2

    def test_user_02_profile_count(self):
        """user_02 should have 2 profiles."""
        resp = post_memories(
            {"memory_type": "profile", "filters": {"user_id": "user_02"}}
        )
        data = assert_success(resp)["data"]
        assert data["total_count"] == 2

    def test_user_03_profile_count(self):
        """user_03 should have 2 profiles."""
        resp = post_memories(
            {"memory_type": "profile", "filters": {"user_id": "user_03"}}
        )
        data = assert_success(resp)["data"]
        assert data["total_count"] == 2

    def test_group_02_profile_count(self):
        """group_02 should have 2 profiles."""
        resp = post_memories(
            {"memory_type": "profile", "filters": {"group_id": "group_02"}}
        )
        data = assert_success(resp)["data"]
        assert data["total_count"] == 2

    def test_group_03_profile_count(self):
        """group_03 should have 2 profiles."""
        resp = post_memories(
            {"memory_type": "profile", "filters": {"group_id": "group_03"}}
        )
        data = assert_success(resp)["data"]
        assert data["total_count"] == 2

    def test_all_groups_in_operator(self):
        """group_id in all 3 groups should return all 10 episodes."""
        resp = post_memories(
            {
                "memory_type": "episodic_memory",
                "filters": {"group_id": {"in": ["group_01", "group_02", "group_03"]}},
            }
        )
        data = assert_success(resp)["data"]
        assert data["total_count"] == 10

    def test_episode_fields_present(self):
        """Verify episode response includes all expected fields."""
        resp = post_memories(
            {
                "memory_type": "episodic_memory",
                "page_size": 1,
                "filters": {"user_id": "user_01"},
            }
        )
        data = assert_success(resp)["data"]
        ep = data["episodes"][0]

        required_fields = [
            "id",
            "user_id",
            "group_id",
            "session_id",
            "timestamp",
            "participants",
            "summary",
            "subject",
            "episode",
            "type",
            "parent_type",
            "parent_id",
        ]
        for field in required_fields:
            assert field in ep, f"Episode missing field: {field}"

        # Must NOT contain vector or audit fields
        assert "vector" not in ep
        assert "created_at" not in ep
        assert "updated_at" not in ep

    def test_profile_fields_present(self):
        """Verify profile response includes all expected fields."""
        resp = post_memories(
            {
                "memory_type": "profile",
                "page_size": 1,
                "filters": {"user_id": "user_01"},
            }
        )
        data = assert_success(resp)["data"]
        pf = data["profiles"][0]

        required_fields = [
            "id",
            "user_id",
            "group_id",
            "profile_data",
            "scenario",
            "memcell_count",
        ]
        for field in required_fields:
            assert field in pf, f"Profile missing field: {field}"

        # Must NOT contain audit fields
        assert "created_at" not in pf
        assert "updated_at" not in pf

    def test_default_pagination(self):
        """Default page=1, page_size=20 should work without explicit params."""
        resp = post_memories(
            {
                "memory_type": "episodic_memory",
                "filters": {"group_id": {"in": ["group_01", "group_02", "group_03"]}},
            }
        )
        data = assert_success(resp)["data"]
        assert data["total_count"] == 10
        assert data["count"] == 10


# ================================================================
# Soft delete filtering
# ================================================================
@pytest.mark.integration
class TestSoftDeleteFiltering:
    """Verify soft-deleted records are automatically excluded from results.

    Seed data includes:
    - 2 soft-deleted episodic memories (ep_del_01: user_01/group_01, ep_del_02: user_02/group_02)
    - 1 soft-deleted profile (pf_del_01: user_01/group_01)
    These should NEVER appear in query results.
    """

    def test_deleted_episodes_excluded_from_user_count(self):
        """user_01 has 5 active + 1 deleted episode. Should return 5."""
        resp = post_memories(
            {"memory_type": "episodic_memory", "filters": {"user_id": "user_01"}}
        )
        data = assert_success(resp)["data"]
        assert data["total_count"] == 5

    def test_deleted_episodes_excluded_from_group_count(self):
        """group_01 has 6 active + 1 deleted episode. Should return 6."""
        resp = post_memories(
            {"memory_type": "episodic_memory", "filters": {"group_id": "group_01"}}
        )
        data = assert_success(resp)["data"]
        assert data["total_count"] == 6

    def test_deleted_episodes_not_in_results(self):
        """No episode with subject 'Deleted Episode' should appear."""
        resp = post_memories(
            {"memory_type": "episodic_memory", "filters": {"user_id": "user_01"}}
        )
        data = assert_success(resp)["data"]
        subjects = [ep["subject"] for ep in data["episodes"]]
        assert "Deleted Episode" not in subjects

    def test_deleted_episodes_excluded_from_group_02(self):
        """group_02 has 2 active + 1 deleted episode. Should return 2."""
        resp = post_memories(
            {"memory_type": "episodic_memory", "filters": {"group_id": "group_02"}}
        )
        data = assert_success(resp)["data"]
        assert data["total_count"] == 2

    def test_deleted_profiles_excluded_from_user_count(self):
        """user_01 has 3 active + 1 deleted profile. Should return 3."""
        resp = post_memories(
            {"memory_type": "profile", "filters": {"user_id": "user_01"}}
        )
        data = assert_success(resp)["data"]
        assert data["total_count"] == 3

    def test_deleted_profiles_excluded_from_group_count(self):
        """group_01 has 3 active + 1 deleted profile. Should return 3."""
        resp = post_memories(
            {"memory_type": "profile", "filters": {"group_id": "group_01"}}
        )
        data = assert_success(resp)["data"]
        assert data["total_count"] == 3

    def test_total_episodes_excludes_all_deleted(self):
        """All 3 groups combined: 10 active episodes, 2 deleted. Should return 10."""
        resp = post_memories(
            {
                "memory_type": "episodic_memory",
                "filters": {"group_id": {"in": ["group_01", "group_02", "group_03"]}},
            }
        )
        data = assert_success(resp)["data"]
        assert data["total_count"] == 10


# ================================================================
# Agent memory GET tests (agent_case / agent_skill)
# ================================================================
#
# These tests use real data created by demo/search_agent_demo.py
# (user_id=demo_user, 4 agent_cases, 3 agent_skills).
#
# Prerequisites: demo data must exist in MongoDB.
# ================================================================

AGENT_USER_ID = "demo_user"


@pytest.mark.integration
class TestAgentCaseGet:
    """Test GET agent_case with real data from demo runs."""

    def test_get_agent_cases_by_user(self):
        """Get all agent cases for demo_user."""
        resp = post_memories(
            {"memory_type": "agent_case", "filters": {"user_id": AGENT_USER_ID}}
        )
        data = assert_success(resp)["data"]
        assert data["total_count"] >= 1
        assert len(data["agent_cases"]) >= 1
        assert data["episodes"] == []
        assert data["profiles"] == []
        assert data["agent_skills"] == []

    def test_agent_case_fields_present(self):
        """Verify agent_case response includes all expected fields."""
        resp = post_memories(
            {
                "memory_type": "agent_case",
                "page_size": 1,
                "filters": {"user_id": AGENT_USER_ID},
            }
        )
        data = assert_success(resp)["data"]
        case = data["agent_cases"][0]

        required_fields = [
            "id",
            "user_id",
            "group_id",
            "session_id",
            "task_intent",
            "approach",
            "quality_score",
            "timestamp",
            "parent_type",
            "parent_id",
        ]
        for field in required_fields:
            assert field in case, f"Agent case missing field: {field}"

    def test_agent_case_pagination(self):
        """Pagination should work for agent_case."""
        resp = post_memories(
            {
                "memory_type": "agent_case",
                "page": 1,
                "page_size": 2,
                "filters": {"user_id": AGENT_USER_ID},
            }
        )
        data = assert_success(resp)["data"]
        assert data["count"] <= 2
        assert data["total_count"] >= data["count"]

    def test_agent_case_sort_desc(self):
        """Agent cases should be sortable by timestamp desc."""
        resp = post_memories(
            {
                "memory_type": "agent_case",
                "rank_by": "timestamp",
                "rank_order": "desc",
                "filters": {"user_id": AGENT_USER_ID},
            }
        )
        data = assert_success(resp)["data"]
        cases = data["agent_cases"]
        if len(cases) >= 2:
            assert cases[0]["timestamp"] >= cases[1]["timestamp"]

    def test_agent_case_filter_by_session(self):
        """Filter agent cases by session_id."""
        resp = post_memories(
            {"memory_type": "agent_case", "filters": {"user_id": AGENT_USER_ID}}
        )
        all_cases = assert_success(resp)["data"]["agent_cases"]
        if not all_cases:
            pytest.skip("No agent cases in database")

        session_id = all_cases[0]["session_id"]
        resp2 = post_memories(
            {
                "memory_type": "agent_case",
                "filters": {"user_id": AGENT_USER_ID, "session_id": session_id},
            }
        )
        data2 = assert_success(resp2)["data"]
        for case in data2["agent_cases"]:
            assert case["session_id"] == session_id


@pytest.mark.integration
class TestAgentSkillGet:
    """Test GET agent_skill with real data from demo runs."""

    def test_get_agent_skills_by_user(self):
        """Get all agent skills for demo_user."""
        resp = post_memories(
            {"memory_type": "agent_skill", "filters": {"user_id": AGENT_USER_ID}}
        )
        data = assert_success(resp)["data"]
        assert data["total_count"] >= 1
        assert len(data["agent_skills"]) >= 1
        assert data["episodes"] == []
        assert data["profiles"] == []
        assert data["agent_cases"] == []

    def test_agent_skill_fields_present(self):
        """Verify agent_skill response includes all expected fields."""
        resp = post_memories(
            {
                "memory_type": "agent_skill",
                "page_size": 1,
                "filters": {"user_id": AGENT_USER_ID},
            }
        )
        data = assert_success(resp)["data"]
        skill = data["agent_skills"][0]

        required_fields = [
            "id",
            "user_id",
            "group_id",
            "cluster_id",
            "name",
            "description",
            "content",
            "confidence",
            "maturity_score",
        ]
        for field in required_fields:
            assert field in skill, f"Agent skill missing field: {field}"

    def test_agent_skill_pagination(self):
        """Pagination should work for agent_skill."""
        resp = post_memories(
            {
                "memory_type": "agent_skill",
                "page": 1,
                "page_size": 2,
                "filters": {"user_id": AGENT_USER_ID},
            }
        )
        data = assert_success(resp)["data"]
        assert data["count"] <= 2
        assert data["total_count"] >= data["count"]

    def test_agent_skill_sort_by_updated_at(self):
        """Agent skills default sort should use updated_at (no timestamp field)."""
        resp = post_memories(
            {
                "memory_type": "agent_skill",
                "rank_by": "timestamp",
                "rank_order": "desc",
                "filters": {"user_id": AGENT_USER_ID},
            }
        )
        # Should not error - timestamp fallback to updated_at
        assert_success(resp)
