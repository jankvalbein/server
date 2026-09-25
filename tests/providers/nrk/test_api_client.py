"""Tests for NRK API normalization and stream selection."""

from __future__ import annotations

import logging
from typing import Any, Self
from unittest.mock import Mock

import pytest

from music_assistant.providers.nrk.api_client import NRKAPIClient
from music_assistant.providers.nrk.models import NRKNotPlayableError


class FakeResponse:
    """Minimal aiohttp response double."""

    def __init__(self, payload: dict[str, Any], status: int = 200) -> None:
        self.payload = payload
        self.status = status
        self.headers: dict[str, str] = {}

    async def json(self) -> dict[str, Any]:
        return self.payload

    def raise_for_status(self) -> None:
        if self.status >= 400:
            raise RuntimeError(self.status)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        return None


def make_client(payload: dict[str, Any]) -> NRKAPIClient:
    session = Mock()
    session.get = Mock(return_value=FakeResponse(payload))
    return NRKAPIClient(session, logging.getLogger("nrk-test"))


async def test_resolve_manifest_prefers_unencrypted_hls() -> None:
    client = make_client(
        {
            "playability": "playable",
            "sourceMedium": "audio",
            "playable": {
                "duration": "PT12M44S",
                "assets": [
                    {
                        "url": "https://example/audio.mp3",
                        "format": "MP3",
                        "mimeType": "audio/mpeg",
                        "encrypted": False,
                        "encryptionScheme": "none",
                    },
                    {
                        "url": "https://example/muxed.m3u8",
                        "format": "HLS",
                        "mimeType": "application/vnd.apple.mpegurl",
                        "encrypted": False,
                        "encryptionScheme": "none",
                    },
                ],
            },
        }
    )

    stream = await client.resolve_manifest("/playback/manifest/program/ABC")
    assert stream.url == "https://example/muxed.m3u8"
    assert stream.format == "HLS"
    assert stream.duration == 764
    assert stream.source_medium == "audio"


async def test_resolve_manifest_rejects_encrypted_only_assets() -> None:
    client = make_client(
        {
            "playability": "playable",
            "playable": {
                "assets": [
                    {
                        "url": "https://example/encrypted.m3u8",
                        "format": "HLS",
                        "encrypted": True,
                        "encryptionScheme": "sample-aes",
                    }
                ]
            },
        }
    )

    with pytest.raises(NRKNotPlayableError):
        await client.resolve_manifest("/playback/manifest/program/ABC")


async def test_radio_pages_are_normalized() -> None:
    client = make_client(
        {
            "pages": [
                {
                    "id": "podkast",
                    "title": "Podkast",
                    "imageSquare": {
                        "webImages": [
                            {"uri": "https://example/300.jpg", "width": 300},
                            {"uri": "https://example/960.jpg", "width": 960},
                        ]
                    },
                }
            ]
        }
    )

    pages = await client.get_radio_pages()
    assert len(pages) == 1
    assert pages[0].page_id == "podkast"
    assert pages[0].title == "Podkast"
    assert pages[0].image_url == "https://example/960.jpg"


async def test_catalog_episode_is_normalized() -> None:
    client = make_client({})
    episode = client._parse_episode(  # noqa: SLF001 - focused parser unit test
        {
            "episodeId": "l_123",
            "titles": {"title": "Episode", "subtitle": "Subtitle"},
            "duration": "PT15M2S",
            "durationInSeconds": 902,
            "date": "2026-02-10T08:00:00+01:00",
            "availability": {"status": "available"},
            "image": [{"url": "https://example/960.jpg", "width": 960}],
        },
        kind="podcast",
        parent_id="show",
    )

    assert episode.episode_id == "l_123"
    assert episode.parent_id == "show"
    assert episode.duration == 902
    assert episode.published == "2026-02-10T08:00:00+01:00"
    assert episode.available is True


async def test_search_normalizes_channels_and_series() -> None:
    client = make_client(
        {
            "results": {
                "channels": {
                    "results": [
                        {
                            "id": "nrk-p1",
                            "type": "channel",
                            "title": "NRK P1",
                            "images": [{"url": "https://example/p1.jpg", "width": 960}],
                        }
                    ]
                },
                "series": {
                    "results": [
                        {
                            "id": "x",
                            "seriesId": "abels-taarn",
                            "type": "podcast",
                            "title": "Abels tårn",
                            "description": "Vitenskap på øret.",
                            "images_1_1": [
                                {"url": "https://example/abel.jpg", "width": 960}
                            ],
                        },
                        {
                            "id": "y",
                            "seriesId": "studio-2",
                            "type": "series",
                            "title": "Studio 2",
                            "description": "Kulturprogram.",
                            "images": [{"url": "https://example/studio2.jpg", "width": 960}],
                        },
                    ]
                },
            }
        }
    )

    channels, shows = await client.search("test", limit=5)

    assert [channel.channel_id for channel in channels] == ["nrk-p1"]
    assert [(show.kind, show.show_id) for show in shows] == [
        ("podcast", "abels-taarn"),
        ("series", "studio-2"),
    ]


async def test_program_payload_uses_temporal_titles_and_duration_object() -> None:
    client = make_client({})
    program = {
        "id": "program-internal-id",
        "episodeId": "ABC123",
        "temporalTitles": {
            "titles": [],
            "defaultTitles": {
                "mainTitle": "Konsert fra arkivet",
                "subtitle": "Live fra Rockefeller",
            },
        },
        "duration": {"seconds": 3600, "iso8601": "PT1H"},
        "date": "2026-01-01T20:00:00+01:00",
        "availability": {"status": "available"},
        "image": [{"url": "https://example/program.jpg", "width": 960}],
    }

    show = client._program_to_show(program, "ABC123")  # noqa: SLF001
    episode = client._parse_episode(  # noqa: SLF001
        program,
        kind="program",
        parent_id="ABC123",
        fallback_id="ABC123",
    )

    assert show.title == "Konsert fra arkivet"
    assert show.subtitle == "Live fra Rockefeller"
    assert episode.title == "Konsert fra arkivet"
    assert episode.duration == 3600


async def test_page_plug_metadata_uses_nrk_human_readable_titles() -> None:
    client = make_client({})

    podcast = client.show_from_plug(
        {
            "type": "podcast",
            "title": "Fallback title",
            "tagline": "Nyheter og aktualitet.",
            "image": {"webImages": [{"uri": "https://example/plug.jpg", "width": 960}]},
            "podcast": {
                "podcastId": "desken_brenner",
                "podcastTitle": "Desken brenner",
                "numberOfEpisodes": 42,
            },
        }
    )
    assert podcast is not None
    assert podcast.show_id == "desken_brenner"
    assert podcast.title == "Desken brenner"
    assert podcast.subtitle == "Nyheter og aktualitet."
    assert podcast.total_episodes == 42
    assert podcast.image_url == "https://example/plug.jpg"

    series = client.show_from_plug(
        {
            "type": "series",
            "series": {
                "seriesId": "studio-2",
                "seriesTitle": "Studio 2",
                "numberOfEpisodes": 12,
            },
        }
    )
    assert series is not None
    assert series.show_id == "studio-2"
    assert series.title == "Studio 2"

    program = client.show_from_plug(
        {
            "type": "standaloneProgram",
            "standaloneProgram": {
                "programId": "ABC123",
                "programTitle": "Konsert fra arkivet",
            },
        }
    )
    assert program is not None
    assert program.kind == "program"
    assert program.show_id == "ABC123"
    assert program.title == "Konsert fra arkivet"


async def test_page_plug_metadata_normalizes_channel_and_episode_titles() -> None:
    client = make_client({})

    channel = client.channel_from_plug(
        {
            "type": "channel",
            "channel": {
                "channelId": "nrk-p1",
                "channelTitle": "NRK P1",
            },
        }
    )
    assert channel is not None
    assert channel.channel_id == "nrk-p1"
    assert channel.title == "NRK P1"

    podcast_episode = client.episode_from_plug(
        {
            "type": "podcastEpisode",
            "podcastEpisode": {
                "episodeId": "episode-1",
                "podcastId": "sentralbordet",
                "podcastTitle": "Sentralbordet på NRK",
                "podcastEpisodeTitle": "Kan du høre meg gjennom gulvet?",
                "duration": "PT31M",
                "imageUrl": "https://example/episode.jpg",
            },
        }
    )
    assert podcast_episode is not None
    assert podcast_episode.parent_id == "sentralbordet"
    assert podcast_episode.episode_id == "episode-1"
    assert podcast_episode.title == "Kan du høre meg gjennom gulvet?"
    assert podcast_episode.duration == 1860

    series_episode = client.episode_from_plug(
        {
            "type": "episode",
            "episode": {
                "programId": "MUHR12345678",
                "seriesId": "studio-2",
                "seriesTitle": "Studio 2",
                "episodeTitle": "Dagens sending",
                "duration": "PT57M",
            },
        }
    )
    assert series_episode is not None
    assert series_episode.kind == "series"
    assert series_episode.parent_id == "studio-2"
    assert series_episode.episode_id == "MUHR12345678"
    assert series_episode.title == "Dagens sending"
