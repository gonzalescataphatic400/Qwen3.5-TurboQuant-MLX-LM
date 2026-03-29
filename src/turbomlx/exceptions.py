"""Package-local exceptions."""


class TurboMLXError(Exception):
    """Base TurboMLX exception."""


class MissingDependencyError(TurboMLXError):
    """Raised when an optional runtime dependency is unavailable."""


class UnsupportedConfigurationError(TurboMLXError):
    """Raised when a requested TurboQuant configuration is invalid."""


class UnsupportedRuntimeVersionError(TurboMLXError):
    """Raised when installed MLX runtime packages are outside the tested range."""


class PromptCacheSerializationError(TurboMLXError):
    """Raised when a TurboMLX-owned prompt-cache archive cannot be serialized or restored."""
