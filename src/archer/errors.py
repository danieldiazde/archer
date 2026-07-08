class ArcherError(Exception):
    """Base class for expected Archer errors."""


class ConfigError(ArcherError):
    """Raised when configuration is invalid."""


class FetchError(ArcherError):
    """Raised when all data sources fail."""


class MissingDataError(ArcherError):
    """Raised when expected data is missing."""


class IntegrityError(ArcherError):
    """Raised when stored data fails integrity verification."""