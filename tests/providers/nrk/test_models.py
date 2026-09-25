"""Unit tests for NRK parsing helpers."""

from datetime import datetime

from music_assistant.providers.nrk.models import (
    link_href,
    parse_datetime,
    parse_iso_duration,
    path_tail,
    pick_image_url,
)


def test_parse_iso_duration() -> None:
    assert parse_iso_duration("PT15M3S") == 903
    assert parse_iso_duration("PT57M") == 3420
    assert parse_iso_duration("PT1H2M3S") == 3723
    assert parse_iso_duration(None) == 0
    assert parse_iso_duration("not-a-duration") == 0


def test_parse_datetime() -> None:
    assert parse_datetime("2026-02-10T08:00:00+01:00") == datetime.fromisoformat(
        "2026-02-10T08:00:00+01:00"
    )
    assert parse_datetime(None) is None
    assert parse_datetime("nonsense") is None


def test_link_and_path_helpers() -> None:
    assert link_href("/podcasts/foo") == "/podcasts/foo"
    assert link_href({"href": "/podcasts/foo"}) == "/podcasts/foo"
    assert path_tail("/podcasts/foo/episodes/bar?x=1") == "bar"


def test_pick_image_prefers_first_width_at_or_above_target() -> None:
    images = [
        {"url": "https://example/300.jpg", "width": 300},
        {"url": "https://example/960.jpg", "width": 960},
        {"url": "https://example/1920.jpg", "width": 1920},
    ]
    assert pick_image_url(images) == "https://example/960.jpg"
