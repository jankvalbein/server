"""NRK music provider for Music Assistant."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .provider import SUPPORTED_FEATURES, NRKProvider

if TYPE_CHECKING:
    from music_assistant_models.config_entries import ProviderConfig
    from music_assistant_models.provider import ProviderManifest

    from music_assistant.mass import MusicAssistant
    from music_assistant.models import ProviderInstanceType


async def setup(
    mass: MusicAssistant, manifest: ProviderManifest, config: ProviderConfig
) -> ProviderInstanceType:
    """Initialize the NRK provider."""
    return NRKProvider(mass, manifest, config, SUPPORTED_FEATURES)
