"""Tests for the NRK Music Assistant provider."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from music_assistant_models.errors import MediaNotFoundError, ResourceTemporarilyUnavailable
from music_assistant_models.media_items import Podcast, ProviderMapping, Radio

from music_assistant.providers.nrk.models import NRKEpisode, NRKShow
from music_assistant.providers.nrk.provider import NRKProvider


def _create_provider() -> NRKProvider:
    """Create an NRK provider with only the dependencies needed by these unit tests."""
    with patch.object(NRKProvider, "__init__", lambda *_args, **_kwargs: None):
        provider = NRKProvider.__new__(NRKProvider)

    provider.config = MagicMock()
    provider.config.instance_id = "nrk"
    provider.manifest = MagicMock()
    provider.manifest.domain = "nrk"
    provider.mass = MagicMock()
    provider.mass.music.podcasts.get_library_items_by_prov_id = AsyncMock(return_value=[])
    provider.mass.music.radio.get_library_items_by_prov_id = AsyncMock(return_value=[])
    provider.logger = MagicMock()
    return provider


def _library_podcast(provider_item_id: str, name: str = "NRK programme") -> Podcast:
    """Create the shape of an NRK podcast already stored in the MA library."""
    mapping = ProviderMapping(
        item_id=provider_item_id,
        provider_domain="nrk",
        provider_instance="nrk",
    )
    return Podcast(
        item_id="123",
        provider="library",
        name=name,
        provider_mappings={mapping},
    )


async def test_library_podcasts_refresh_existing_nrk_items() -> None:
    """A user-added NRK podcast is refreshed from NRK instead of disappearing on sync."""
    provider = _create_provider()
    stored = _library_podcast("podcast/oppdatert", "Old title")
    provider.mass.music.podcasts.get_library_items_by_prov_id = AsyncMock(
        return_value=[stored]
    )
    fresh = provider._podcast_item(  # noqa: SLF001 - focused provider unit test
        NRKShow(
            kind="podcast",
            show_id="oppdatert",
            title="Oppdatert",
            subtitle="Fresh metadata",
            total_episodes=12,
        )
    )
    provider.get_podcast = AsyncMock(return_value=fresh)

    result = [item async for item in provider.get_library_podcasts()]

    assert result == [fresh]
    provider.get_podcast.assert_awaited_once_with("podcast/oppdatert")


async def test_library_podcasts_keep_item_when_nrk_lookup_fails() -> None:
    """A transient/missing NRK lookup must not silently delete a user-added library item."""
    provider = _create_provider()
    stored = _library_podcast("tv_series/ages-reise", "Åges reise")
    mapping = next(iter(stored.provider_mappings))
    provider.mass.music.podcasts.get_library_items_by_prov_id = AsyncMock(
        return_value=[stored]
    )
    provider.get_podcast = AsyncMock(
        side_effect=MediaNotFoundError("NRK item temporarily unavailable")
    )

    result = [item async for item in provider.get_library_podcasts()]

    assert len(result) == 1
    assert result[0].item_id == "tv_series/ages-reise"
    assert result[0].name == "Åges reise"
    assert result[0].provider_mappings == {mapping}


async def test_tv_series_episodes_keep_stable_library_ids_and_parent() -> None:
    """NRK TV audio episodes use stable podcast/episode IDs across refreshes."""
    provider = _create_provider()
    show = NRKShow(
        kind="tv_series",
        show_id="ages-reise",
        title="Åges reise",
        subtitle="Musikkdokumentar",
    )
    episodes = [
        NRKEpisode(
            kind="tv_series",
            parent_id="ages-reise",
            episode_id="MUHU26000126",
            title="9. mai 2025",
            duration=3120,
            published="2025-05-09T20:00:00+02:00",
        ),
        NRKEpisode(
            kind="tv_series",
            parent_id="ages-reise",
            episode_id="MUHU26000226",
            title="16. mai 2025",
            duration=3180,
            published="2025-05-16T20:00:00+02:00",
        ),
    ]
    provider._get_show = AsyncMock(return_value=show)  # type: ignore[method-assign]
    provider._get_episodes = AsyncMock(return_value=episodes)  # type: ignore[method-assign]

    result = [
        episode
        async for episode in provider.get_podcast_episodes("tv_series/ages-reise")
    ]

    assert [episode.item_id for episode in result] == [
        "tv_series/ages-reise/MUHU26000126",
        "tv_series/ages-reise/MUHU26000226",
    ]
    assert all(episode.podcast.item_id == "tv_series/ages-reise" for episode in result)
    assert all(episode.podcast.name == "Åges reise" for episode in result)
    assert [episode.duration for episode in result] == [3120, 3180]


async def test_library_radios_refresh_existing_nrk_items() -> None:
    """A user-added NRK radio station is refreshed from current NRK metadata."""
    provider = _create_provider()
    mapping = ProviderMapping(
        item_id="radio/nrk-p1",
        provider_domain="nrk",
        provider_instance="nrk",
    )
    stored = Radio(
        item_id="321",
        provider="library",
        name="Old P1 title",
        provider_mappings={mapping},
    )
    provider.mass.music.radio.get_library_items_by_prov_id = AsyncMock(return_value=[stored])
    fresh = Radio(
        item_id="radio/nrk-p1",
        provider="nrk",
        name="NRK P1",
        provider_mappings={mapping},
    )
    provider.get_radio = AsyncMock(return_value=fresh)

    result = [item async for item in provider.get_library_radios()]

    assert result == [fresh]
    provider.get_radio.assert_awaited_once_with("radio/nrk-p1")


async def test_library_radios_keep_item_when_nrk_lookup_fails() -> None:
    """A transient NRK lookup failure must not silently delete a saved radio station."""
    provider = _create_provider()
    mapping = ProviderMapping(
        item_id="radio/nrk-p2",
        provider_domain="nrk",
        provider_instance="nrk",
    )
    stored = Radio(
        item_id="654",
        provider="library",
        name="NRK P2",
        provider_mappings={mapping},
    )
    provider.mass.music.radio.get_library_items_by_prov_id = AsyncMock(return_value=[stored])
    provider.get_radio = AsyncMock(
        side_effect=ResourceTemporarilyUnavailable("temporary failure")
    )

    result = [item async for item in provider.get_library_radios()]

    assert len(result) == 1
    assert result[0].item_id == "radio/nrk-p2"
    assert result[0].name == "NRK P2"
    assert result[0].provider_mappings == {mapping}


async def test_library_sync_config_is_hidden_and_enabled() -> None:
    """NRK's MA-local podcast and radio sync stays enabled without confusing toggles."""
    provider = _create_provider()

    entries = await provider.get_config_entries()

    assert len(entries) == 2
    assert all(entry.hidden is True for entry in entries)
    assert all(entry.default_value is True for entry in entries)
