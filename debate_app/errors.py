class DebateError(Exception):
    """Base class for expected application errors."""


class ConfigurationError(DebateError):
    """Raised when required runtime configuration is missing or invalid."""


class SafetyError(DebateError):
    """Raised when input or model output violates a safety constraint."""


class AgentCallError(DebateError):
    """Raised when a bounded Agent call cannot produce valid output."""

