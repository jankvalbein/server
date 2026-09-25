"""Async client for the public NRK playback and radio catalogue endpoints."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Final, cast

import aiohttp
from music_assistant_models.errors import RateLimited, ResourceTemporarilyUnavailable

from music_assistant.helpers.throttle_retry import (
    ThrottlerManager,
    parse_retry_after,
    throttle_with_retries,
)

from .models import (
    NRKChannel,
    NRKEpisode,
    NRKNotFoundError,
    NRKNotPlayableError,
    NRKPage,
    NRKSection,
    NRKShow,
    NRKShowKind,
    NRKStream,
    link_href,
    parse_iso_duration,
    path_tail,
    pick_image_url,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

API_BASE: Final = "https://psapi.nrk.no"
HTTP_TIMEOUT: Final = aiohttp.ClientTimeout(total=15)
PAGE_SIZE: Final = 50

# Conservative client-side rate limit. Archive browsing can fan out into several requests,
# so keep burst traffic modest even when the user navigates quickly.
THROTTLER = ThrottlerManager(
    rate_limit=4,
    period=1,
    retry_attempts=3,
    initial_backoff=2,
)

HEADERS: Final = {
    "Accept": "application/json",
    "User-Agent": "MusicAssistant-NRK/0.1 (+https://music-assistant.io)",
}


class NRKAPIClient:
    """Small async client for NRK PSAPI."""

    domain = "nrk"
    throttler = THROTTLER

    def __init__(self, session: aiohttp.ClientSession, logger: logging.Logger) -> None:
        """Initialize the API client with Music Assistant's shared HTTP session."""
        self._session = session
        self.logger = logger

    @throttle_with_retries
    async def _get(
        self,
        path: str,
        *,
        params: dict[str, str | int] | None = None,
    ) -> dict[str, Any]:
        """GET JSON from PSAPI and normalize transport errors."""
        url = path if path.startswith("http") else f"{API_BASE}{path}"
        try:
            async with self._session.get(
                url,
                params=params,
                headers=HEADERS,
                timeout=HTTP_TIMEOUT,
            ) as response:
                if response.status == 404:
                    raise NRKNotFoundError(path)
                if response.status == 429:
                    backoff = parse_retry_after(response.headers.get("Retry-After"))
                    raise RateLimited("NRK rate limit", backoff_time=backoff)
                if response.status in (500, 502, 503, 504):
                    raise ResourceTemporarilyUnavailable(
                        f"NRK PSAPI returned HTTP {response.status}",
                        backoff_time=30,
                    )
                response.raise_for_status()
                payload = await response.json()
        except NRKNotFoundError:
            raise
        except (RateLimited, ResourceTemporarilyUnavailable):
            raise
        except (aiohttp.ClientError, TimeoutError, ValueError) as err:
            raise ResourceTemporarilyUnavailable("NRK PSAPI unavailable") from err

        if not isinstance(payload, dict):
            raise ResourceTemporarilyUnavailable("NRK PSAPI returned invalid JSON")
        return cast("dict[str, Any]", payload)


    async def search(
        self, query: str, limit: int = 10
    ) -> tuple[list[NRKChannel], list[NRKShow]]:
        """Search the NRK Radio catalogue for live channels and programme series."""
        data = await self._get(
            "/radio/search/search",
            params={
                "q": query,
                "take": max(1, limit),
                "skip": 0,
                "page": 1,
            },
        )
        results = data.get("results")
        if not isinstance(results, dict):
            return [], []

        channels: list[NRKChannel] = []
        channel_group = results.get("channels")
        raw_channels = channel_group.get("results", []) if isinstance(channel_group, dict) else []
        for raw in raw_channels:
            if not isinstance(raw, dict):
                continue
            channel_id = raw.get("id")
            title = raw.get("title")
            if not isinstance(channel_id, str) or not isinstance(title, str):
                continue
            channels.append(
                NRKChannel(
                    channel_id=channel_id,
                    title=title,
                    image_url=pick_image_url(raw.get("images_1_1"))
                    or pick_image_url(raw.get("images")),
                )
            )

        shows: list[NRKShow] = []
        series_group = results.get("series")
        raw_series = series_group.get("results", []) if isinstance(series_group, dict) else []
        for raw in raw_series:
            if not isinstance(raw, dict):
                continue
            result_type = raw.get("type")
            if result_type not in {"podcast", "series"}:
                continue
            show_id = raw.get("seriesId")
            title = raw.get("title")
            if not isinstance(show_id, str) or not isinstance(title, str):
                continue
            description = raw.get("description")
            shows.append(
                NRKShow(
                    kind=result_type,
                    show_id=show_id,
                    title=title,
                    subtitle=description if isinstance(description, str) and description else None,
                    image_url=pick_image_url(raw.get("images_1_1"))
                    or pick_image_url(raw.get("images")),
                )
            )

        return channels[:limit], shows[:limit]

    async def get_radio_pages(self) -> list[NRKPage]:
        """Return the browse pages exposed by NRK Radio."""
        data = await self._get("/radio/pages")
        pages: list[NRKPage] = []
        for raw in data.get("pages", []):
            if not isinstance(raw, dict):
                continue
            page_id = raw.get("id")
            title = raw.get("title")
            if not isinstance(page_id, str) or not isinstance(title, str):
                continue
            pages.append(
                NRKPage(
                    page_id=page_id,
                    title=title,
                    image_url=pick_image_url(raw.get("imageSquare"))
                    or pick_image_url(raw.get("image")),
                )
            )
        return pages

    async def get_radio_page(self, page_id: str) -> list[NRKSection]:
        """Return all sections and plugs for one NRK Radio page."""
        data = await self._get(f"/radio/pages/{page_id}")
        sections: list[NRKSection] = []
        for raw in data.get("sections", []):
            if not isinstance(raw, dict):
                continue
            container = raw.get("included") or raw.get("placeholder")
            if not isinstance(container, dict):
                continue
            plugs = tuple(item for item in container.get("plugs", []) if isinstance(item, dict))
            title = container.get("title")
            sections.append(
                NRKSection(
                    title=title if isinstance(title, str) and title else "NRK",
                    plugs=plugs,
                )
            )
        return sections

    async def get_tv_pages(self) -> list[NRKPage]:
        """Return browse pages exposed by NRK TV."""
        data = await self._get("/tv/pages")
        pages: list[NRKPage] = [
            NRKPage(page_id="frontpage", title="Forsiden"),
        ]
        seen = {"frontpage"}
        raw_pages = data.get("pageListItems", [])
        if not isinstance(raw_pages, list):
            raw_pages = []
        for raw in raw_pages:
            if not isinstance(raw, dict):
                continue
            page_id = raw.get("id")
            if not isinstance(page_id, str) or not page_id:
                page_id = path_tail(link_href(raw.get("_links", {}).get("self")))
            title = raw.get("title") or raw.get("displayValue")
            if (
                not isinstance(page_id, str)
                or not page_id
                or page_id in seen
                or not isinstance(title, str)
                or not title
            ):
                continue
            seen.add(page_id)
            pages.append(
                NRKPage(
                    page_id=page_id,
                    title=title,
                    image_url=pick_image_url(raw.get("image")),
                )
            )
        return pages

    async def get_tv_page(self, page_id: str) -> list[NRKSection]:
        """Return all sections and plugs for one NRK TV page."""
        data = await self._get(f"/tv/pages/{page_id}")
        sections: list[NRKSection] = []
        for raw in data.get("sections", []):
            if not isinstance(raw, dict):
                continue
            container = raw.get("included") or raw.get("placeholder")
            if not isinstance(container, dict):
                continue
            plugs = tuple(
                item for item in container.get("plugs", []) if isinstance(item, dict)
            )
            title = container.get("title")
            if plugs:
                sections.append(
                    NRKSection(
                        title=title if isinstance(title, str) and title else "NRK TV",
                        plugs=plugs,
                    )
                )
        return sections

    async def get_show(self, kind: NRKShowKind, show_id: str) -> NRKShow:
        """Fetch podcast/radio/TV programme metadata."""
        if kind == "program":
            data = await self._get(f"/radio/catalog/programs/{show_id}")
            return self._program_to_show(data, show_id)

        if kind == "tv_program":
            data = await self._get(f"/tv/catalog/programs/{show_id}")
            return self._tv_program_to_show(data, show_id)

        if kind == "tv_series":
            data = await self._get(f"/tv/catalog/series/{show_id}")
            return self._tv_series_to_show(data, show_id)

        data = await self._get(f"/radio/catalog/{kind}/{show_id}")
        series = data.get("series")
        if not isinstance(series, dict):
            series = data.get("sequential")
        if not isinstance(series, dict):
            series = data

        titles = series.get("titles") if isinstance(series.get("titles"), dict) else {}
        title = titles.get("title") or series.get("title") or show_id
        subtitle = titles.get("subtitle") or series.get("subtitle")
        image_url = (
            pick_image_url(series.get("squareImage"))
            or pick_image_url(series.get("image"))
            or pick_image_url(series.get("posterImage"))
        )

        total = data.get("episodeCount") or series.get("episodeCount")
        return NRKShow(
            kind=kind,
            show_id=show_id,
            title=str(title),
            subtitle=str(subtitle) if subtitle else None,
            image_url=image_url,
            total_episodes=int(total) if isinstance(total, (int, float)) else None,
        )

    async def iter_show_episodes(
        self, kind: NRKShowKind, show_id: str
    ) -> AsyncGenerator[NRKEpisode]:
        """Iterate every episode for a podcast, radio series or NRK TV series."""
        if kind == "program":
            data = await self._get(f"/radio/catalog/programs/{show_id}")
            yield self._parse_episode(
                data, kind="program", parent_id=show_id, fallback_id=show_id
            )
            return

        if kind == "tv_program":
            data = await self._get(f"/tv/catalog/programs/{show_id}")
            yield self._parse_tv_program(
                data, kind="tv_program", parent_id=show_id, fallback_id=show_id
            )
            return

        if kind == "tv_series":
            series_data = await self._get(f"/tv/catalog/series/{show_id}")
            links = series_data.get("_links", {})
            seasons = links.get("seasons", []) if isinstance(links, dict) else []
            for season in seasons:
                if not isinstance(season, dict):
                    continue
                season_name = season.get("name")
                if not isinstance(season_name, str) or not season_name:
                    continue
                season_data = await self._get(
                    f"/tv/catalog/series/{show_id}/seasons/{season_name}"
                )
                embedded = season_data.get("_embedded", {})
                if not isinstance(embedded, dict):
                    continue
                raw_episodes = embedded.get("episodes")
                if not isinstance(raw_episodes, list):
                    raw_episodes = embedded.get("instalments", [])
                if not isinstance(raw_episodes, list):
                    continue
                for raw in raw_episodes:
                    if isinstance(raw, dict):
                        yield self._parse_episode(
                            raw, kind="tv_series", parent_id=show_id
                        )
            return

        page = 1
        while True:
            data = await self._get(
                f"/radio/catalog/{kind}/{show_id}/episodes",
                params={"page": page, "pageSize": PAGE_SIZE, "sort": "desc"},
            )
            embedded = data.get("_embedded")
            raw_episodes = embedded.get("episodes", []) if isinstance(embedded, dict) else []
            episodes = [item for item in raw_episodes if isinstance(item, dict)]
            for raw in episodes:
                yield self._parse_episode(raw, kind=kind, parent_id=show_id)

            if len(episodes) < PAGE_SIZE:
                break
            page += 1

    async def get_episode(
        self, kind: NRKShowKind, parent_id: str, episode_id: str
    ) -> NRKEpisode:
        """Fetch one podcast, radio or TV episode."""
        if kind == "podcast":
            data = await self._get(
                f"/radio/catalog/podcast/{parent_id}/episodes/{episode_id}"
            )
            return self._parse_episode(
                data, kind=kind, parent_id=parent_id, fallback_id=episode_id
            )

        if kind in {"tv_series", "tv_program"}:
            data = await self._get(f"/tv/catalog/programs/{episode_id}")
            return self._parse_tv_program(
                data, kind=kind, parent_id=parent_id, fallback_id=episode_id
            )

        data = await self._get(f"/radio/catalog/programs/{episode_id}")
        return self._parse_episode(
            data,
            kind=kind,
            parent_id=parent_id,
            fallback_id=episode_id,
        )

    async def resolve_manifest(self, manifest_path: str) -> NRKStream:
        """Resolve the best unencrypted playable asset from an NRK playback manifest."""
        data = await self._get(manifest_path)
        if data.get("playability") not in (None, "playable"):
            raise NRKNotPlayableError(str(data.get("playability")))

        playable = data.get("playable")
        if not isinstance(playable, dict):
            raise NRKNotPlayableError("NRK returned no playable payload")

        assets = [item for item in playable.get("assets", []) if isinstance(item, dict)]
        assets = [
            asset
            for asset in assets
            if not asset.get("encrypted") and str(asset.get("encryptionScheme") or "none") == "none"
        ]
        if not assets:
            raise NRKNotPlayableError("NRK returned no unencrypted playable asset")

        def asset_rank(asset: dict[str, Any]) -> int:
            fmt = str(asset.get("format") or "").upper()
            if fmt == "HLS":
                return 0
            if fmt in {"MP3", "AAC"}:
                return 1
            return 2

        asset = sorted(assets, key=asset_rank)[0]
        url = asset.get("url")
        if not isinstance(url, str) or not url:
            raise NRKNotPlayableError("NRK playback asset has no URL")

        duration = parse_iso_duration(playable.get("duration"))
        if not duration:
            scores = data.get("statistics", {}).get("scores", {})
            score_duration = (
                scores.get("springStreamDuration") if isinstance(scores, dict) else None
            )
            duration = parse_iso_duration(score_duration)

        return NRKStream(
            url=url,
            format=str(asset.get("format") or ""),
            mime_type=str(asset.get("mimeType")) if asset.get("mimeType") else None,
            duration=duration or None,
            source_medium=str(data.get("sourceMedium")) if data.get("sourceMedium") else None,
        )

    async def resolve_channel(self, channel_id: str) -> NRKStream:
        """Resolve a live NRK channel."""
        return await self.resolve_manifest(f"/playback/manifest/channel/{channel_id}")

    async def resolve_podcast_episode(self, podcast_id: str, episode_id: str) -> NRKStream:
        """Resolve an NRK podcast episode."""
        return await self.resolve_manifest(
            f"/playback/manifest/podcast/{podcast_id}/{episode_id}"
        )

    async def resolve_program(self, program_id: str) -> NRKStream:
        """Resolve an on-demand NRK programme.

        The same playback endpoint is used by radio programmes and NRK TV programmes.
        The provider currently exposes only the verified radio catalogue.
        """
        return await self.resolve_manifest(f"/playback/manifest/program/{program_id}")

    def tv_show_from_plug(self, plug: dict[str, Any]) -> NRKShow | None:
        """Convert an NRK TV page series/program plug into a Podcast-like show."""
        target_type = plug.get("targetType")
        content = plug.get("displayContractContent")
        if not isinstance(content, dict):
            content = {}

        if target_type == "series":
            raw = plug.get("series")
            if not isinstance(raw, dict):
                return None
            show_id = raw.get("seriesId")
            if not isinstance(show_id, str) or not show_id:
                show_id = path_tail(link_href(plug.get("_links", {}).get("series")))
            if not show_id:
                return None
            title = content.get("contentTitle") or raw.get("seriesTitle") or show_id
            return NRKShow(
                kind="tv_series",
                show_id=show_id,
                title=str(title),
                subtitle=(
                    str(content["contentDescription"])
                    if isinstance(content.get("contentDescription"), str)
                    else None
                ),
                image_url=pick_image_url(content.get("displayContractImage"))
                or pick_image_url(raw.get("image")),
            )

        if target_type in {"program", "standaloneProgram"}:
            raw = plug.get("program") or plug.get("standaloneProgram")
            if not isinstance(raw, dict):
                return None
            show_id = raw.get("programId") or raw.get("prfId")
            if not isinstance(show_id, str) or not show_id:
                return None
            title = content.get("contentTitle") or raw.get("programTitle") or show_id
            return NRKShow(
                kind="tv_program",
                show_id=show_id,
                title=str(title),
                image_url=pick_image_url(content.get("displayContractImage"))
                or pick_image_url(raw.get("image")),
                total_episodes=1,
            )

        return None

    def tv_episode_from_plug(self, plug: dict[str, Any]) -> NRKEpisode | None:
        """Convert a directly playable NRK TV page episode plug."""
        if plug.get("targetType") != "episode":
            return None
        raw = plug.get("episode")
        if not isinstance(raw, dict):
            return None
        program_id = raw.get("programId") or raw.get("prfId")
        if not isinstance(program_id, str) or not program_id:
            return None
        series_id = raw.get("seriesId")
        kind: NRKShowKind
        if isinstance(series_id, str) and series_id:
            kind = "tv_series"
            parent_id = series_id
        else:
            kind = "tv_program"
            parent_id = program_id
        content = plug.get("displayContractContent")
        if not isinstance(content, dict):
            content = {}
        title = content.get("contentTitle") or raw.get("episodeTitle") or program_id
        return NRKEpisode(
            kind=kind,
            parent_id=parent_id,
            episode_id=program_id,
            title=str(title),
            image_url=pick_image_url(content.get("displayContractImage"))
            or pick_image_url(raw.get("image")),
            available=True,
        )

    def channel_from_plug(self, plug: dict[str, Any]) -> NRKChannel | None:
        """Convert a radio page channel plug into a normalized channel."""
        raw = plug.get("channel")
        if not isinstance(raw, dict):
            return None

        channel_id = raw.get("channelId")
        if not isinstance(channel_id, str) or not channel_id:
            link = link_href(plug.get("_links", {}).get("channel"))
            channel_id = path_tail(link)
        if not channel_id:
            return None

        titles = raw.get("titles") if isinstance(raw.get("titles"), dict) else {}
        title = (
            raw.get("channelTitle")
            or titles.get("title")
            or raw.get("title")
            or plug.get("title")
            or channel_id
        )
        return NRKChannel(
            channel_id=channel_id,
            title=str(title),
            image_url=pick_image_url(raw.get("image"))
            or pick_image_url(plug.get("image")),
        )

    def show_from_plug(self, plug: dict[str, Any]) -> NRKShow | None:
        """Convert a podcast/series/program plug into a show."""
        plug_type = plug.get("type")
        if plug_type == "podcast":
            kind: NRKShowKind = "podcast"
            raw = plug.get("podcast")
            id_key = "podcastId"
            title_key = "podcastTitle"
            link_key = "podcast"
        elif plug_type == "series":
            kind = "series"
            raw = plug.get("series")
            id_key = "seriesId"
            title_key = "seriesTitle"
            link_key = "series"
        elif plug_type == "standaloneProgram":
            kind = "program"
            raw = plug.get("standaloneProgram") or plug.get("program")
            id_key = "programId"
            title_key = "programTitle"
            link_key = "program"
        else:
            return None

        if not isinstance(raw, dict):
            return None

        show_id = raw.get(id_key)
        if not isinstance(show_id, str) or not show_id:
            show_id = path_tail(link_href(plug.get("_links", {}).get(link_key)))
        if not show_id and raw.get("id"):
            show_id = str(raw["id"])
        if not show_id:
            return None

        titles = raw.get("titles") if isinstance(raw.get("titles"), dict) else {}
        title = (
            raw.get(title_key)
            or titles.get("title")
            or raw.get("title")
            or plug.get("title")
            or show_id
        )
        subtitle = titles.get("subtitle") or raw.get("subtitle") or plug.get("tagline")
        total = raw.get("numberOfEpisodes") or raw.get("episodeCount")
        return NRKShow(
            kind=kind,
            show_id=show_id,
            title=str(title),
            subtitle=str(subtitle) if subtitle else None,
            image_url=pick_image_url(raw.get("image"))
            or pick_image_url(raw.get("imageUrl"))
            or pick_image_url(plug.get("image")),
            total_episodes=int(total) if isinstance(total, (int, float)) else None,
        )

    def episode_from_plug(self, plug: dict[str, Any]) -> NRKEpisode | None:
        """Convert a directly playable page plug into an episode."""
        plug_type = plug.get("type")
        if plug_type == "podcastEpisode":
            raw = plug.get("podcastEpisode")
            if not isinstance(raw, dict):
                return None
            links = plug.get("_links", {})
            parent_id = raw.get("podcastId")
            episode_id = raw.get("episodeId")
            if not isinstance(parent_id, str) or not parent_id:
                parent_id = path_tail(link_href(links.get("podcast")))
            if not isinstance(episode_id, str) or not episode_id:
                episode_id = path_tail(link_href(links.get("podcastEpisode")))
            if not parent_id or not episode_id:
                return None
            return self._parse_episode(
                raw,
                kind="podcast",
                parent_id=parent_id,
                fallback_id=episode_id,
            )

        if plug_type == "episode":
            raw = plug.get("episode")
            if not isinstance(raw, dict):
                return None
            episode_id = raw.get("programId") or raw.get("episodeId")
            if not isinstance(episode_id, str) or not episode_id:
                episode_id = path_tail(link_href(plug.get("_links", {}).get("episode")))
            if not episode_id:
                return None
            parent_id = raw.get("seriesId")
            if not isinstance(parent_id, str) or not parent_id:
                series = raw.get("series")
                if isinstance(series, dict):
                    parent_id = series.get("id")
                    if not parent_id:
                        parent_id = path_tail(link_href(series.get("_links", {}).get("self")))
            if not isinstance(parent_id, str) or not parent_id:
                # A direct episode without a resolvable parent is represented as a one-off show.
                parent_id = episode_id
                kind: NRKShowKind = "program"
            else:
                kind = "series"
            return self._parse_episode(
                raw,
                kind=kind,
                parent_id=parent_id,
                fallback_id=episode_id,
            )

        return None

    def _tv_series_to_show(self, data: dict[str, Any], series_id: str) -> NRKShow:
        """Convert an NRK TV series payload into a Podcast-like show."""
        series_type = data.get("seriesType")
        body = data.get(series_type) if isinstance(series_type, str) else None
        if not isinstance(body, dict):
            for key in ("sequential", "standard", "news"):
                candidate = data.get(key)
                if isinstance(candidate, dict):
                    body = candidate
                    break
        if not isinstance(body, dict):
            body = data
        title, subtitle = self._title_parts(body, fallback=series_id)
        return NRKShow(
            kind="tv_series",
            show_id=series_id,
            title=title,
            subtitle=subtitle,
            image_url=pick_image_url(body.get("image")),
        )

    def _tv_program_to_show(self, data: dict[str, Any], program_id: str) -> NRKShow:
        """Convert an NRK TV programme payload into a one-episode show."""
        info = data.get("programInformation")
        if not isinstance(info, dict):
            info = data
        title, subtitle = self._title_parts(info, fallback=program_id)
        return NRKShow(
            kind="tv_program",
            show_id=program_id,
            title=title,
            subtitle=subtitle,
            image_url=pick_image_url(info.get("image")),
            total_episodes=1,
        )

    def _parse_tv_program(
        self,
        data: dict[str, Any],
        *,
        kind: NRKShowKind,
        parent_id: str,
        fallback_id: str,
    ) -> NRKEpisode:
        """Normalize the NRK TV program detail payload."""
        info = data.get("programInformation")
        if not isinstance(info, dict):
            info = data
        title, subtitle = self._title_parts(info, fallback=fallback_id)

        more = data.get("moreInformation")
        duration = 0
        published = None
        if isinstance(more, dict):
            duration_data = more.get("duration")
            if isinstance(duration_data, dict):
                seconds = duration_data.get("seconds")
                if isinstance(seconds, (int, float)):
                    duration = int(seconds)
            transmissions = more.get("transmissions")
            if isinstance(transmissions, dict):
                first = transmissions.get("first")
                if isinstance(first, dict):
                    candidate = first.get("date") or first.get("displayValue")
                    if isinstance(candidate, str):
                        published = candidate

        availability = info.get("availability")
        available = True
        if isinstance(availability, dict):
            status = availability.get("status")
            if isinstance(status, str):
                available = status in {"available", "expires"}

        return NRKEpisode(
            kind=kind,
            parent_id=parent_id,
            episode_id=fallback_id,
            title=title,
            subtitle=subtitle,
            duration=max(0, duration),
            published=published,
            image_url=pick_image_url(info.get("image")),
            available=available,
        )

    def _program_to_show(self, data: dict[str, Any], program_id: str) -> NRKShow:
        """Convert a programme catalogue payload into a one-episode show."""
        title, subtitle = self._title_parts(data, fallback=program_id)
        return NRKShow(
            kind="program",
            show_id=program_id,
            title=title,
            subtitle=subtitle,
            image_url=pick_image_url(data.get("squareImage"))
            or pick_image_url(data.get("image")),
            total_episodes=1,
        )

    def _parse_episode(
        self,
        data: dict[str, Any],
        *,
        kind: NRKShowKind,
        parent_id: str,
        fallback_id: str | None = None,
    ) -> NRKEpisode:
        """Normalize a catalogue episode payload."""
        # Some detail endpoints wrap the actual episode/program object.
        raw = data
        for key in ("episode", "program", "podcastEpisode"):
            nested = data.get(key)
            if isinstance(nested, dict):
                raw = nested
                break

        episode_id = raw.get("episodeId") or raw.get("prfId") or raw.get("id") or fallback_id
        if not isinstance(episode_id, str) or not episode_id:
            raise NRKNotFoundError("Episode payload has no stable id")

        title, subtitle = self._title_parts(raw, fallback=episode_id)

        duration = raw.get("durationInSeconds")
        if not isinstance(duration, (int, float)):
            duration_payload = raw.get("duration")
            if isinstance(duration_payload, dict):
                duration = duration_payload.get("seconds")
                if not isinstance(duration, (int, float)):
                    duration = parse_iso_duration(duration_payload.get("iso8601"))
            else:
                duration = parse_iso_duration(duration_payload)

        availability = raw.get("availability")
        available = True
        if isinstance(availability, dict):
            status = availability.get("status")
            if isinstance(status, str):
                available = status in {"available", "expires"}

        published = raw.get("date")
        if not isinstance(published, str):
            published = None

        return NRKEpisode(
            kind=kind,
            parent_id=parent_id,
            episode_id=episode_id,
            title=title,
            subtitle=subtitle,
            duration=max(0, int(duration or 0)),
            published=published,
            image_url=pick_image_url(raw.get("squareImage"))
            or pick_image_url(raw.get("image"))
            or pick_image_url(raw.get("imageUrl")),
            available=available,
        )

    @staticmethod
    def _title_parts(data: dict[str, Any], *, fallback: str) -> tuple[str, str | None]:
        """Extract title/subtitle from both episode and standalone-program payloads."""
        titles = data.get("titles")
        if isinstance(titles, dict):
            title = titles.get("title")
            subtitle = titles.get("subtitle")
            if isinstance(title, str) and title:
                return title, subtitle if isinstance(subtitle, str) and subtitle else None

        temporal = data.get("temporalTitles")
        if isinstance(temporal, dict):
            defaults = temporal.get("defaultTitles")
            if isinstance(defaults, dict):
                title = defaults.get("mainTitle")
                subtitle = defaults.get("subtitle")
                if isinstance(title, str) and title:
                    return title, subtitle if isinstance(subtitle, str) and subtitle else None
            dynamic_titles = temporal.get("titles")
            if isinstance(dynamic_titles, list):
                usable = [item for item in dynamic_titles if isinstance(item, str) and item]
                if usable:
                    return " - ".join(usable), None

        subtitle = data.get("subtitle")
        for key in (
            "podcastEpisodeTitle",
            "episodeTitle",
            "programTitle",
            "podcastTitle",
            "seriesTitle",
            "title",
        ):
            title = data.get(key)
            if isinstance(title, str) and title:
                return title, subtitle if isinstance(subtitle, str) and subtitle else None
        return fallback, subtitle if isinstance(subtitle, str) and subtitle else None
