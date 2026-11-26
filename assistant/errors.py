"""Exception hierarchy.

Every error may carry a ``user_message`` that is safe to speak aloud; anything
without one gets a generic canned message from :mod:`assistant.messages`.
"""
from __future__ import annotations


class AssistantError(Exception):
    user_message: str | None = None

    def __init__(self, message: str = "", *, user_message: str | None = None) -> None:
        super().__init__(message)
        if user_message is not None:
            self.user_message = user_message


class ConfigError(AssistantError):
    """Invalid or missing configuration."""


class AudioError(AssistantError):
    """Microphone / speaker problem (usually transient → reopen and retry)."""


class AudioUnavailable(AudioError):
    """Audio backend is missing entirely (not worth retrying)."""


class STTError(AssistantError):
    """A speech recogniser failed (as opposed to hearing nothing)."""


class TranscriptionEmpty(STTError):
    """The recogniser ran fine but found no usable speech (noise, silence)."""


class LLMError(AssistantError):
    """All / one LLM backend failed."""


class TTSError(AssistantError):
    """Speech synthesis or playback failed."""


class SkillError(AssistantError):
    """A skill could not complete; ``user_message`` explains why."""
