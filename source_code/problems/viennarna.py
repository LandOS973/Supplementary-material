from __future__ import annotations

import csv
from pathlib import Path
from urllib.request import urlopen

import numpy as np


RNA_ALPHABET = np.array(list("ACGU"))
ETERNA100_TSV_URL = (
    "https://raw.githubusercontent.com/eternagame/eterna100-benchmarking/master/data/eterna100_puzzles.tsv"
)


def is_dotbracket(value: str) -> bool:
    if not value:
        return False
    allowed = set(".()[]{}<>&")
    return all(ch in allowed for ch in value)


def normalize_target_struct(value: str) -> str:
    target = str(value).strip().replace(" ", "")
    if not target:
        raise ValueError("Target structure is empty.")
    if not is_dotbracket(target):
        raise ValueError(f"Invalid target structure `{target}`. Expected dot-bracket symbols only.")
    return target


def extract_dotbracket_from_row(row: dict) -> str | None:
    preferred_keys = (
        "target_structure",
        "target_struct",
        "structure",
        "secstruct",
        "secondary_structure",
        "dotbracket",
        "dot_bracket",
    )
    for key in preferred_keys:
        value = row.get(key)
        if value and is_dotbracket(str(value).strip()):
            return str(value).strip()
    for value in row.values():
        if value and is_dotbracket(str(value).strip()):
            return str(value).strip()
    return None


def row_matches_name(row: dict, target_name: str) -> bool:
    ref = target_name.strip().lower()
    if not ref:
        return False
    name_keys = ("name", "puzzle_name", "title", "display_name", "id")
    for key in name_keys:
        value = row.get(key)
        if value and str(value).strip().lower() == ref:
            return True
    for value in row.values():
        if isinstance(value, str) and value.strip().lower() == ref:
            return True
    return False


def read_text_from_source(source: str) -> str:
    source = str(source).strip()
    if source.startswith("http://") or source.startswith("https://"):
        with urlopen(source, timeout=20) as response:
            return response.read().decode("utf-8")
    return Path(source).read_text(encoding="utf-8")


def load_target_from_eterna100(
    target_name: str,
    source: str,
    fallback_target: str,
    verbose: bool = True,
) -> tuple[str, str]:
    try:
        raw = read_text_from_source(source)
        reader = csv.DictReader(raw.splitlines(), delimiter="\t")
        rows = list(reader)
        for row in rows:
            if row_matches_name(row, target_name):
                candidate = extract_dotbracket_from_row(row)
                if candidate:
                    target = normalize_target_struct(candidate)
                    if verbose:
                        print(f"[ViennaRNA] target loaded from benchmark: `{target_name}` (len={len(target)})")
                    return target, target_name
        if verbose:
            print(
                f"[ViennaRNA][WARN] target `{target_name}` not found in benchmark source; "
                "using fallback default target."
            )
    except Exception as exc:
        if verbose:
            print(f"[ViennaRNA][WARN] failed loading benchmark target ({exc}); using fallback default target.")
    return normalize_target_struct(fallback_target), "fallback_default"


def tokens_to_rna_strings(tokens) -> list[str]:
    arr = np.asarray(tokens, dtype=np.int64)
    if arr.ndim == 1:
        arr = arr[None, :]
    arr = np.clip(arr, 0, len(RNA_ALPHABET) - 1)
    return ["".join(RNA_ALPHABET[row]) for row in arr]
