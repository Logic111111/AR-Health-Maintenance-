"""Source interface: anything that can deliver raw NOJA recloser data interval by interval.

Phase 1 uses :class:`ar_gen.sources.simulated.SimulatedNojaSource`. A real source
(DNP3 / SCADA historian / CMS log import) only has to implement this interface and
return rows shaped like the raw tables in :mod:`ar_gen.db.schema`; the variable and
Health Index code does not change.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

import sqlalchemy as sa

from ..sim.out import Chunk

RawBatch = Chunk
"""rows (table -> list of row dicts), asset_updates, ground_truth, last_ms, next_ids."""


class RecloserSource(ABC):
    name = "abstract"

    @abstractmethod
    def last_collected_ms(self) -> int:
        """End (UTC ms) of the last interval whose raw data is complete in the database."""

    @abstractmethod
    def collect(self, until_ms: int, chunk_intervals: int = 96) -> Iterator[RawBatch]:
        """Yield raw batches covering (last_collected_ms, until_ms]."""

    @abstractmethod
    def commit(self, batch: RawBatch) -> None:
        """Called after ``batch`` was committed to the database."""

    def recover(self, engine: sa.Engine) -> None:
        """Remove partially written data after a crash (optional)."""
