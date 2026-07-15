"""Market-data ingestion, persistence, quality control, and reference calendars.

The data package deliberately exposes small, immutable boundary objects.  Research
code consumes a :class:`~edgestack.data.sources.SourceBatch` rather than a vendor
response, which makes provenance and availability timestamps impossible to drop by
accident.
"""

from edgestack.data.cache import ContentAddressedRawStore, DataCache
from edgestack.data.calendars import FOMCCalendarSource, NYSECalendar
from edgestack.data.factors import (
    FREDCSVSource,
    KenFrenchDailyFactorsSource,
    ReferenceDataCache,
)
from edgestack.data.sources import (
    BAR_COLUMNS,
    BarRequest,
    DailyBarSource,
    FallbackDailyBarSource,
    Quote,
    QuoteSource,
    RawPayload,
    SourceBatch,
    SourceCapabilities,
    StooqBulkArchiveDailyBarSource,
    StooqDailyBarSource,
    TiingoDailyBarSource,
    TiingoQuoteSource,
    YahooDailyBarSource,
    YahooQuoteSource,
)
from edgestack.data.universe import LIQUID_ETFS, WikipediaSP500UniverseSource

__all__ = [
    "BAR_COLUMNS",
    "LIQUID_ETFS",
    "BarRequest",
    "ContentAddressedRawStore",
    "DailyBarSource",
    "DataCache",
    "FOMCCalendarSource",
    "FREDCSVSource",
    "FallbackDailyBarSource",
    "KenFrenchDailyFactorsSource",
    "NYSECalendar",
    "Quote",
    "QuoteSource",
    "RawPayload",
    "ReferenceDataCache",
    "SourceBatch",
    "SourceCapabilities",
    "StooqBulkArchiveDailyBarSource",
    "StooqDailyBarSource",
    "TiingoDailyBarSource",
    "TiingoQuoteSource",
    "WikipediaSP500UniverseSource",
    "YahooDailyBarSource",
    "YahooQuoteSource",
]
