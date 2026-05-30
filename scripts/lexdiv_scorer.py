#!/usr/bin/env python3
"""Reproduced competition Blind-A scorers: LexDiv + CatalogDiv (CPU, offline).

LexDiv = corpus-pooled Distinct-2 over all predicted_response texts:
  lowercase, whitespace .split(), pool bigrams across ALL responses, skip
  responses with <2 tokens, NO stopword filter, NO punctuation strip.
  distinct2 = unique bigrams / total bigrams.

Verified EXACT (0.0000 error, 4 decimals) against 7 scored Blind-A submissions:
  R92 p11 0.8720, R84c 0.8720, R78 0.8845, R77 0.8821, R74 0.8719,
  R73 0.8536, R63c 0.8438.

CatalogDiv = unique recommended track ids across all cases / catalog size (47071):
  len(set(all predicted_track_ids)) / 47071. Verified: p11 -> 0.0301.

This is the canonical OFFLINE gate. R86 burned a slot by optimizing a
stopword-filtered Distinct-2, which is NOT the competition metric — use THIS.
"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

CATALOG_SIZE = 47071


def lexical_diversity(responses: list[str]) -> float:
    total = 0
    seen: set[tuple[str, str]] = set()
    for r in responses:
        toks = (r or "").lower().split()
        if len(toks) < 2:
            continue
        for i in range(len(toks) - 1):
            seen.add((toks[i], toks[i + 1]))
            total += 1
    return len(seen) / total if total else 0.0


def catalog_diversity(track_id_lists: list[list[str]], catalog_size: int = CATALOG_SIZE) -> float:
    uniq: set[str] = set()
    for lst in track_id_lists:
        uniq.update(lst)
    return len(uniq) / catalog_size


def score_zip(path: Path) -> dict:
    with zipfile.ZipFile(path) as zf:
        name = "prediction.json" if "prediction.json" in zf.namelist() else zf.namelist()[0]
        rows = json.loads(zf.read(name))
    responses = [r.get("predicted_response", "") for r in rows]
    tracks = [r.get("predicted_track_ids", []) for r in rows]
    return {
        "n_cases": len(rows),
        "lexical_diversity": round(lexical_diversity(responses), 6),
        "catalog_diversity": round(catalog_diversity(tracks), 6),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("zip", type=Path, help="submission .zip with prediction.json")
    args = ap.parse_args()
    s = score_zip(args.zip)
    print(f"{args.zip.name}: LexDiv={s['lexical_diversity']:.4f}  "
          f"CatalogDiv={s['catalog_diversity']:.4f}  n={s['n_cases']}")
