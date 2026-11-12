"""Domain event contracts and the payloads older builds persisted."""

from __future__ import annotations

from everos.memory.events import EpisodeExtracted


def test_an_episode_event_without_an_owner_type_reads_as_a_user_s() -> None:
    """A payload persisted before the group fields replays as a user's episode."""
    event = EpisodeExtracted.model_validate_json(
        '{"memcell_id": "mc_1", "episode_entry_id": "ep_1", "episode_text": "t",'
        ' "episode_timestamp_ms": 1, "owner_id": "u_ann"}'
    )

    assert (event.owner_type, event.speaker_ids, event.group_key) == ("user", [], None)


def test_an_episode_event_persisted_with_reflection_keys_still_replays() -> None:
    """An earlier 2.0 build persisted ``source`` and ``reflection``; both drop."""
    event = EpisodeExtracted.model_validate_json(
        '{"memcell_id": "mc_1", "episode_entry_id": "ep_1", "episode_text": "t",'
        ' "episode_timestamp_ms": 1, "owner_id": "u_ann", "source": "pipeline",'
        ' "reflection": null}'
    )

    assert (event.owner_id, event.episode_entry_id) == ("u_ann", "ep_1")
    assert "source" not in event.model_dump()
    assert "reflection" not in event.model_dump()


def test_a_group_episode_event_round_trips_with_its_group_key() -> None:
    event = EpisodeExtracted(
        memcell_id="mc_1",
        episode_entry_id="gep_1",
        episode_text="t",
        episode_timestamp_ms=1,
        owner_id="trip",
        owner_type="group",
        app_id="a",
        project_id="p",
        speaker_ids=["u_ann", "u_bob"],
    )

    again = EpisodeExtracted.model_validate_json(event.model_dump_json())

    assert again == event
    assert again.group_key == "a/p/trip"
