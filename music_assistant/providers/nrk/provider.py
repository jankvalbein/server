"""Music Assistant provider implementation for NRK."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Sequence
from typing import TYPE_CHECKING
from urllib.parse import quote, unquote

from music_assistant_models.enums import (
    ContentType,
    ImageType,
    MediaType,
    ProviderFeature,
    StreamType,
)
from music_assistant_models.errors import (
    MediaNotFoundError,
    UnplayableMediaError,
)
from music_assistant_models.media_items import (
    AudioFormat,
    BrowseFolder,
    ItemMapping,
    MediaItemImage,
    MediaItemType,
    Podcast,
    PodcastEpisode,
    ProviderMapping,
    Radio,
    SearchResults,
)
from music_assistant_models.streamdetails import StreamDetails

from music_assistant.controllers.cache import use_cache
from music_assistant.helpers.podcast_parsers import rank_episodes_by_date
from music_assistant.models.music_provider import MusicProvider

from .api_client import NRKAPIClient
from .models import (
    NRKChannel,
    NRKEpisode,
    NRKNotFoundError,
    NRKNotPlayableError,
    NRKPage,
    NRKSection,
    NRKShow,
    NRKShowKind,
    parse_datetime,
)

if TYPE_CHECKING:
    from music_assistant_models.config_entries import ConfigEntry, ProviderConfig
    from music_assistant_models.provider import ProviderManifest

    from music_assistant.mass import MusicAssistant


SUPPORTED_FEATURES = {
    ProviderFeature.BROWSE,
    ProviderFeature.SEARCH,
}

BROWSE_RADIO = "radio"


class NRKProvider(MusicProvider):
    """NRK Radio and archive provider."""

    _client: NRKAPIClient

    @property
    def supported_media_types(self) -> set[MediaType]:
        """Return the media types served by the provider."""
        return {MediaType.RADIO, MediaType.PODCAST, MediaType.PODCAST_EPISODE}

    async def get_config_entries(self) -> tuple[ConfigEntry, ...]:
        """NRK Radio playback does not require configuration."""
        return ()

    async def handle_async_init(self) -> None:
        """Initialize the NRK API client."""
        self._client = NRKAPIClient(self.mass.http_session, self.logger)

    async def browse(self, path: str) -> Sequence[MediaItemType | ItemMapping | BrowseFolder]:
        """Browse NRK Radio pages, sections and programmes."""
        subpath = path.split("://", 1)[1] if "://" in path else ""
        parts = [unquote(part) for part in subpath.split("/") if part]

        if not parts:
            return [
                BrowseFolder(
                    item_id=BROWSE_RADIO,
                    provider=self.instance_id,
                    path=f"{self.instance_id}://{BROWSE_RADIO}",
                    name="NRK Radio",
                )
            ]

        if parts == [BROWSE_RADIO]:
            return [self._page_folder(page) for page in await self._get_pages()]

        if len(parts) == 3 and parts[0] == BROWSE_RADIO and parts[1] == "page":
            page_id = parts[2]
            sections = await self._get_page(page_id)
            return [
                BrowseFolder(
                    item_id=f"{page_id}:{idx}",
                    provider=self.instance_id,
                    path=(
                        f"{self.instance_id}://{BROWSE_RADIO}/page/"
                        f"{quote(page_id, safe='')}/section/{idx}"
                    ),
                    name=section.title,
                )
                for idx, section in enumerate(sections)
                if section.plugs
            ]

        if (
            len(parts) == 5
            and parts[0] == BROWSE_RADIO
            and parts[1] == "page"
            and parts[3] == "section"
        ):
            page_id = parts[2]
            try:
                section_index = int(parts[4])
            except ValueError as err:
                raise KeyError(path) from err
            sections = await self._get_page(page_id)
            if section_index < 0 or section_index >= len(sections):
                raise KeyError(path)
            return self._render_section(sections[section_index])

        raise KeyError(path)

    async def search(
        self, search_query: str, media_types: list[MediaType], limit: int = 5
    ) -> SearchResults:
        """Search NRK Radio channels, podcasts and radio programme series."""
        results = SearchResults()
        query = search_query.strip()
        if not query:
            return results
        if not ({MediaType.RADIO, MediaType.PODCAST} & set(media_types)):
            return results

        channels, shows = await self._client.search(query, limit=limit)
        if MediaType.RADIO in media_types:
            results.radio = [self._radio_item(channel) for channel in channels][:limit]
        if MediaType.PODCAST in media_types:
            results.podcasts = [self._podcast_item(show) for show in shows][:limit]
        return results

    async def get_radio(self, prov_radio_id: str) -> Radio:
        """Return one live radio channel."""
        channel_id = self._parse_radio_id(prov_radio_id)
        if channel := await self._find_channel(channel_id):
            return self._radio_item(channel)
        # A channel can remain playable after it drops out of a curated page. Keep the id
        # resolvable; get_stream_details will make the authoritative playback check.
        return self._radio_item(NRKChannel(channel_id=channel_id, title=channel_id))

    async def get_podcast(self, prov_podcast_id: str) -> Podcast:
        """Return one NRK podcast/radio-series abstraction."""
        kind, show_id = self._parse_show_id(prov_podcast_id)
        try:
            show = await self._get_show(kind, show_id)
        except NRKNotFoundError as err:
            raise MediaNotFoundError(f"NRK programme {prov_podcast_id} not found") from err
        return self._podcast_item(show)

    async def get_podcast_episodes(
        self, prov_podcast_id: str
    ) -> AsyncGenerator[PodcastEpisode]:
        """Yield every episode of an NRK podcast or radio programme."""
        kind, show_id = self._parse_show_id(prov_podcast_id)
        try:
            show = await self._get_show(kind, show_id)
            episodes = await self._get_episodes(kind, show_id)
        except NRKNotFoundError as err:
            raise MediaNotFoundError(f"NRK programme {prov_podcast_id} not found") from err

        podcast = self._podcast_mapping(show)
        positions = rank_episodes_by_date([episode.published for episode in episodes])
        for episode, position in zip(episodes, positions, strict=True):
            yield self._episode_item(episode, podcast, position)

    async def get_podcast_episode(self, prov_episode_id: str) -> PodcastEpisode:
        """Return one NRK podcast/radio episode."""
        kind, parent_id, episode_id = self._parse_episode_id(prov_episode_id)
        try:
            episode = await self._client.get_episode(kind, parent_id, episode_id)
            show = await self._get_show(kind, parent_id)
        except NRKNotFoundError as err:
            raise MediaNotFoundError(f"NRK episode {prov_episode_id} not found") from err
        return self._episode_item(episode, self._podcast_mapping(show), position=0)

    async def get_stream_details(self, item_id: str, media_type: MediaType) -> StreamDetails:
        """Resolve an NRK live or on-demand stream just before playback."""
        try:
            if media_type == MediaType.RADIO:
                channel_id = self._parse_radio_id(item_id)
                stream = await self._client.resolve_channel(channel_id)
                seekable = False
            elif media_type == MediaType.PODCAST_EPISODE:
                kind, parent_id, episode_id = self._parse_episode_id(item_id)
                if kind == "podcast":
                    stream = await self._client.resolve_podcast_episode(parent_id, episode_id)
                else:
                    stream = await self._client.resolve_program(episode_id)
                seekable = True
            else:
                raise UnplayableMediaError(f"NRK does not serve {media_type}")
        except NRKNotFoundError as err:
            raise MediaNotFoundError(f"NRK media {item_id} not found") from err
        except NRKNotPlayableError as err:
            raise UnplayableMediaError(f"NRK media {item_id} is not playable: {err}") from err

        is_hls = stream.format.upper() == "HLS" or ".m3u8" in stream.url.lower()
        stream_type = StreamType.HLS if is_hls else StreamType.HTTP
        if is_hls:
            content_type = ContentType.AAC
        else:
            content_type = ContentType.try_parse(stream.url)

        return StreamDetails(
            provider=self.instance_id,
            item_id=item_id,
            media_type=media_type,
            stream_type=stream_type,
            path=stream.url,
            audio_format=AudioFormat(content_type=content_type),
            duration=stream.duration,
            can_seek=seekable,
            allow_seek=seekable,
        )

    @use_cache(3600 * 6, base_class=NRKPage)
    async def _get_pages(self) -> list[NRKPage]:
        """Cache the relatively static NRK Radio page list."""
        return await self._client.get_radio_pages()

    @use_cache(1800, base_class=NRKSection)
    async def _get_page(self, page_id: str) -> list[NRKSection]:
        """Cache one curated page briefly because its contents can change."""
        return await self._client.get_radio_page(page_id)

    @use_cache(3600 * 6, base_class=NRKShow)
    async def _get_show(self, kind: NRKShowKind, show_id: str) -> NRKShow:
        """Cache show metadata."""
        return await self._client.get_show(kind, show_id)

    @use_cache(900, base_class=NRKEpisode)
    async def _get_episodes(self, kind: NRKShowKind, show_id: str) -> list[NRKEpisode]:
        """Cache one complete episode listing for a short period."""
        return [episode async for episode in self._client.iter_show_episodes(kind, show_id)]

    async def _find_channel(self, channel_id: str) -> NRKChannel | None:
        """Find channel metadata in the curated NRK Radio pages."""
        for page in await self._get_pages():
            for section in await self._get_page(page.page_id):
                for plug in section.plugs:
                    channel = self._client.channel_from_plug(plug)
                    if channel and channel.channel_id == channel_id:
                        return channel
        return None

    def _render_section(
        self, section: NRKSection
    ) -> list[MediaItemType | ItemMapping | BrowseFolder]:
        """Convert NRK page plugs to Music Assistant media items."""
        items: list[MediaItemType | ItemMapping | BrowseFolder] = []
        for plug in section.plugs:
            if channel := self._client.channel_from_plug(plug):
                items.append(self._radio_item(channel))
                continue
            if show := self._client.show_from_plug(plug):
                items.append(self._podcast_item(show))
                continue
            if episode := self._client.episode_from_plug(plug):
                parent_title = self._episode_parent_title(plug, episode)
                parent = ItemMapping(
                    media_type=MediaType.PODCAST,
                    item_id=self._show_id(episode.kind, episode.parent_id),
                    provider=self.instance_id,
                    name=parent_title,
                )
                items.append(self._episode_item(episode, parent, position=0))
                continue
            self.logger.debug("Skipping unsupported NRK page plug type: %s", plug.get("type"))
        return items

    def _page_folder(self, page: NRKPage) -> BrowseFolder:
        """Build a BrowseFolder for an NRK Radio page."""
        folder = BrowseFolder(
            item_id=page.page_id,
            provider=self.instance_id,
            path=(
                f"{self.instance_id}://{BROWSE_RADIO}/page/"
                f"{quote(page.page_id, safe='')}"
            ),
            name=page.title,
        )
        if page.image_url:
            folder.image = self._image(page.image_url)
        return folder

    def _radio_item(self, channel: NRKChannel) -> Radio:
        """Build a Radio item."""
        item_id = self._radio_id(channel.channel_id)
        radio = Radio(
            name=channel.title,
            item_id=item_id,
            provider=self.instance_id,
            provider_mappings={
                ProviderMapping(
                    item_id=item_id,
                    provider_domain=self.domain,
                    provider_instance=self.instance_id,
                )
            },
        )
        if channel.image_url:
            radio.metadata.add_image(self._image(channel.image_url))
        return radio

    def _podcast_item(self, show: NRKShow) -> Podcast:
        """Represent an NRK long-form programme as a Podcast in Music Assistant."""
        item_id = self._show_id(show.kind, show.show_id)
        podcast = Podcast(
            name=show.title,
            item_id=item_id,
            provider=self.instance_id,
            total_episodes=show.total_episodes,
            provider_mappings={
                ProviderMapping(
                    item_id=item_id,
                    provider_domain=self.domain,
                    provider_instance=self.instance_id,
                )
            },
        )
        if show.subtitle:
            podcast.metadata.description = show.subtitle
        if show.image_url:
            podcast.metadata.add_image(self._image(show.image_url))
        return podcast

    def _podcast_mapping(self, show: NRKShow) -> ItemMapping:
        """Build the parent mapping required by PodcastEpisode."""
        return ItemMapping(
            media_type=MediaType.PODCAST,
            item_id=self._show_id(show.kind, show.show_id),
            provider=self.instance_id,
            name=show.title,
        )

    def _episode_item(
        self,
        episode: NRKEpisode,
        podcast: ItemMapping,
        position: int,
    ) -> PodcastEpisode:
        """Build a PodcastEpisode item."""
        item_id = self._episode_id(episode.kind, episode.parent_id, episode.episode_id)
        item = PodcastEpisode(
            name=episode.title,
            item_id=item_id,
            provider=self.instance_id,
            position=position,
            duration=episode.duration,
            podcast=podcast,
            is_playable=episode.available,
            provider_mappings={
                ProviderMapping(
                    item_id=item_id,
                    provider_domain=self.domain,
                    provider_instance=self.instance_id,
                    available=episode.available,
                )
            },
        )
        if episode.subtitle:
            item.metadata.description = episode.subtitle
        if episode.image_url:
            item.metadata.add_image(self._image(episode.image_url))
        if published := parse_datetime(episode.published):
            item.metadata.release_date = published
        return item

    def _image(self, url: str) -> MediaItemImage:
        """Build an externally reachable NRK image."""
        return MediaItemImage(
            type=ImageType.THUMB,
            path=url,
            provider=self.instance_id,
            remotely_accessible=True,
        )

    def _episode_parent_title(self, plug: dict, episode: NRKEpisode) -> str:
        """Find a useful parent title for an episode exposed directly on a page."""
        if episode.kind == "podcast":
            raw = plug.get("podcastEpisode")
            if isinstance(raw, dict):
                podcast = raw.get("podcast")
                if isinstance(podcast, dict):
                    titles = podcast.get("titles")
                    if isinstance(titles, dict) and isinstance(titles.get("title"), str):
                        return titles["title"]
        if episode.kind == "series":
            raw = plug.get("episode")
            if isinstance(raw, dict):
                series = raw.get("series")
                if isinstance(series, dict):
                    titles = series.get("titles")
                    if isinstance(titles, dict) and isinstance(titles.get("title"), str):
                        return titles["title"]
        return episode.title

    @staticmethod
    def _radio_id(channel_id: str) -> str:
        return f"channel/{channel_id}"

    @staticmethod
    def _parse_radio_id(item_id: str) -> str:
        prefix = "channel/"
        if not item_id.startswith(prefix) or not item_id[len(prefix) :]:
            raise MediaNotFoundError(f"Invalid NRK radio id: {item_id}")
        return item_id[len(prefix) :]

    @staticmethod
    def _show_id(kind: NRKShowKind, show_id: str) -> str:
        return f"{kind}/{show_id}"

    @staticmethod
    def _episode_id(kind: NRKShowKind, parent_id: str, episode_id: str) -> str:
        return f"{kind}/{parent_id}/{episode_id}"

    @staticmethod
    def _parse_show_id(item_id: str) -> tuple[NRKShowKind, str]:
        parts = item_id.split("/", 1)
        if len(parts) != 2 or parts[0] not in {"podcast", "series", "program"} or not parts[1]:
            raise MediaNotFoundError(f"Invalid NRK programme id: {item_id}")
        return parts[0], parts[1]  # type: ignore[return-value]

    @staticmethod
    def _parse_episode_id(item_id: str) -> tuple[NRKShowKind, str, str]:
        parts = item_id.split("/", 2)
        if (
            len(parts) != 3
            or parts[0] not in {"podcast", "series", "program"}
            or not parts[1]
            or not parts[2]
        ):
            raise MediaNotFoundError(f"Invalid NRK episode id: {item_id}")
        return parts[0], parts[1], parts[2]  # type: ignore[return-value]
