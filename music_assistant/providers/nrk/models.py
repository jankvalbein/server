"""Data models and parsing helpers for the NRK provider."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import re
from typing import Any, Literal

from mashumaro.mixins.dict import DataClassDictMixin

NRKShowKind = Literal["podcast", "series", "program"]


class NRKError(Exception):
    """Base error raised by the NRK API client."""


class NRKNotFoundError(NRKError):
    """Requested NRK resource does not exist."""


class NRKNotPlayableError(NRKError):
    """Requested NRK resource exists but cannot be played."""


@dataclass(slots=True, frozen=True)
class NRKPage(DataClassDictMixin):
    """A top-level NRK Radio page."""

    page_id: str
    title: str
    image_url: str | None = None


@dataclass(slots=True, frozen=True)
class NRKSection(DataClassDictMixin):
    """A section on an NRK Radio page."""

    title: str
    plugs: tuple[dict[str, Any], ...]


@dataclass(slots=True, frozen=True)
class NRKShow(DataClassDictMixin):
    """A podcast, radio series or one-off radio programme."""

    kind: NRKShowKind
    show_id: str
    title: str
    subtitle: str | None = None
    image_url: str | None = None
    total_episodes: int | None = None


@dataclass(slots=True, frozen=True)
class NRKEpisode(DataClassDictMixin):
    """One playable NRK podcast/radio episode."""

    kind: NRKShowKind
    parent_id: str
    episode_id: str
    title: str
    subtitle: str | None = None
    duration: int = 0
    published: str | None = None
    image_url: str | None = None
    available: bool = True


@dataclass(slots=True, frozen=True)
class NRKChannel:
    """A live NRK radio channel."""

    channel_id: str
    title: str
    image_url: str | None = None


@dataclass(slots=True, frozen=True)
class NRKStream:
    """Resolved stream information from the NRK Playback API."""

    url: str
    format: str
    mime_type: str | None
    duration: int | None
    source_medium: str | None


def parse_iso_duration(value: str | int | float | None) -> int:
    """Convert an ISO-8601 duration used by NRK to whole seconds."""
    if isinstance(value, (int, float)):
        return max(0, int(value))
    if not value or not isinstance(value, str):
        return 0
    match = re.fullmatch(
        r"P(?:(?P<days>\d+)D)?T"
        r"(?:(?P<hours>\d+)H)?"
        r"(?:(?P<minutes>\d+)M)?"
        r"(?:(?P<seconds>\d+(?:\.\d+)?)S)?",
        value,
    )
    if not match:
        return 0
    parts = match.groupdict(default="0")
    return int(
        int(parts["days"]) * 86400
        + int(parts["hours"]) * 3600
        + int(parts["minutes"]) * 60
        + float(parts["seconds"])
    )


def parse_datetime(value: str | None) -> datetime | None:
    """Parse an NRK ISO timestamp without raising on malformed input."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def link_href(value: Any) -> str | None:
    """Extract href from either an NRK link object or a direct link string."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        href = value.get("href")
        if isinstance(href, str):
            return href
    return None


def path_tail(value: str | None) -> str | None:
    """Return the last non-empty segment of a URL/path."""
    if not value:
        return None
    parts = [part for part in value.split("?")[0].split("/") if part]
    return parts[-1] if parts else None


def pick_image_url(value: Any, preferred_width: int = 960) -> str | None:
    """Return a useful image URL from the different NRK image payload shapes."""
    if isinstance(value, str):
        return value if value.startswith(("http://", "https://")) else None

    if isinstance(value, dict):
        for key in ("uri", "url", "imageUrl"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.startswith(("http://", "https://")):
                return candidate
        for key in ("webImages", "images", "items", "image"):
            if key in value:
                candidate = pick_image_url(value[key], preferred_width)
                if candidate:
                    return candidate
        return None

    if not isinstance(value, list) or not value:
        return None

    candidates: list[tuple[int, str]] = []
    for item in value:
        if isinstance(item, str):
            if item.startswith(("http://", "https://")):
                candidates.append((preferred_width, item))
            continue
        if not isinstance(item, dict):
            continue
        url = None
        for key in ("uri", "url", "imageUrl"):
            raw = item.get(key)
            if isinstance(raw, str) and raw.startswith(("http://", "https://")):
                url = raw
                break
        if url:
            width = item.get("width")
            candidates.append((int(width) if isinstance(width, (int, float)) else 0, url))
            continue
        nested = pick_image_url(item, preferred_width)
        if nested:
            candidates.append((0, nested))

    if not candidates:
        return None

    at_or_above = [item for item in candidates if item[0] >= preferred_width]
    if at_or_above:
        return min(at_or_above, key=lambda item: item[0])[1]
    return max(candidates, key=lambda item: item[0])[1]
