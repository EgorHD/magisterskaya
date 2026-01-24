from __future__ import annotations
from dataclasses import dataclass


@dataclass(slots=True)
class IntegrityReport:
    ok: bool
    broken_pages: list[int]
    broken_spans: dict[int, list[int]]  # page_index -> list of span indices
    message: str
