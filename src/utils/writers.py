"""
Output writers. JSONL is the canonical machine-readable output (one
validated Pydantic record per line); CSV export mirrors the "Google
Sheets, 6 tabs" deliverable shape for local/offline demonstration.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel


def write_jsonl(records: Iterable[BaseModel], path: str) -> int:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(r.model_dump_json() + "\n")
            n += 1
    return n


def write_rejects(rejects: Iterable[dict], path: str) -> int:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in rejects:
            f.write(json.dumps(r, default=str) + "\n")
            n += 1
    return n


def flatten(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten(v, new_key, sep))
        elif isinstance(v, list):
            items[new_key] = "; ".join(str(x) for x in v)
        else:
            items[new_key] = v
    return items


def write_csv_tab(records: list[BaseModel], path: str) -> int:
    """Write one 'tab' (CSV file) matching the Google Sheets deliverable structure."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not records:
        Path(path).write_text("")
        return 0
    rows = [flatten(json.loads(r.model_dump_json())) for r in records]
    fieldnames: list[str] = []
    for row in rows:
        for k in row.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)
