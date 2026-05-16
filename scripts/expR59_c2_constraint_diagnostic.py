#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301
"""R59 Phase A — C2 entity/constraint parser diagnostic (heuristic regex).

Dev-only, no training, no LLM, no API. Per `docs/r59_candidates/c2_entity_constraint_parser.md` §5.

Question we are answering:
  If we extract structural constraints (year, duration, tag, artist exclusion)
  from user queries via heuristic regex and apply them as hard filters, would
  they admit GT in cases where the current pool@300 misses GT? Establishes a
  ceiling for the C2 mechanism without building any new retrieval.

What it does:
  1. Rebuild baseline state: pool@300 via weighted_rrf, GT membership in pool
     and in source union. No CV5 LR retrain (not needed for this diagnostic).
  2. Bucket assignment: HIT (GT in pool top-20 via base LR proxy), DEMOTED,
     POOL_MISS, UNREACHABLE. Note: HIT/DEMOTED split here is approximated by
     "GT in pool@300" since we are not running LR. Refined buckets only matter
     for HIT/DEMOTED accounting; POOL_MISS / UNREACHABLE are exact.
  3. For each dev case, run regex extractor on user_query and produce a
     {year_min, year_max, duration_max_ms, exclude_played_artists, required_tags}
     constraint dict. Confidence is per-extraction; cases with confidence >= 0.7
     have an active constraint.
  4. For each case, check whether GT satisfies the extracted constraints.
  5. Compute admission metrics:
     - Constraint extraction rate (Metric 1)
     - GT compliance rate (Metric 2)
     - Hypothetical admission rate (Metric 3) — using better proxy: for POOL_MISS
       cases, GT is "admittable" if its best-source rank < 100 AND GT satisfies
       constraints. For UNREACHABLE, admittable if constrained_pool_size < 1000
       AND GT satisfies (much weaker proxy, flagged).
     - False positive rate (Metric 4)
  6. Save to exp/eval/expR59_c2_constraint_diagnostic.json + console summary.

Gates (per C2 design):
  PROCEED to Phase 2 implementation: Metric 3 >= 150 cases on POOL_MISS+UNREACHABLE.
  TUNE: Metric 2 < 40% (parser noisy).
  ARCHIVE: Metric 3 < 100 cases.
"""
from __future__ import annotations

import json
import os
import pickle
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "4"

import numpy as np  # type: ignore[reportMissingImports]

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.expF1_cfbpr_retrieval import weighted_rrf  # noqa: E402
from scripts.expS2_lambdarank import build_als  # noqa: E402
from scripts.expS2_lambdarank_grouped import als_session_vector  # noqa: E402

R12_CACHE = REPO / "exp" / "eval" / "_R12_all_turns_payload.pkl"
R21_OOF = REPO / "cache" / "r21_production" / "dev_r21_oof_lists.json"
R54_PHASE2_OOF = REPO / "cache" / "r54" / "phase2_full" / "oof_r54_lists.json"
OUT = REPO / "exp" / "eval" / "expR59_c2_constraint_diagnostic.json"

POOL_K = 300
RRF_K = 20
SW = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "ALS": 1.0, "R21": 1.0, "R54": 1.0}

CONFIDENCE_THRESHOLD = 0.7


# -------------------- regex patterns --------------------

# Temporal — decades ("90s", "2010s", "nineties")
DECADE_NUMERIC = re.compile(r"\b(19[2-9]0|20[0-2]0)['']?s\b")  # 1920s..2020s
DECADE_SHORT = re.compile(r"\b(20|30|40|50|60|70|80|90|00|10|20)['']?s\b")
DECADE_WORDS = {
    "twenties": (1920, 1929), "thirties": (1930, 1939), "forties": (1940, 1949),
    "fifties": (1950, 1959), "sixties": (1960, 1969), "seventies": (1970, 1979),
    "eighties": (1980, 1989), "nineties": (1990, 1999), "two thousands": (2000, 2009),
    "twothousands": (2000, 2009), "noughties": (2000, 2009),
}

# Single year mentions
YEAR_FROM = re.compile(r"\bfrom\s+(19\d\d|20\d\d)\b", re.I)
YEAR_IN = re.compile(r"\b(?:in|of)\s+(19\d\d|20\d\d)\b", re.I)
YEAR_BARE = re.compile(r"\b(19\d\d|20\d\d)\b")

# Recency / classics
RECENT_TOKENS = re.compile(r"\b(recent|recent[ -]?(?:release|hit|song)s?|new(?!\s+artist)|newer|latest|just\s+(?:released|came\s+out|dropped)|this\s+year|2024|2025|2026)\b", re.I)
CLASSIC_TOKENS = re.compile(r"\b(classic|classics|oldies|old[\s-]?school|vintage|throwback|retro)\b", re.I)
MODERN_TOKENS = re.compile(r"\b(modern|contemporary|present\s+day)\b", re.I)

# Duration
DURATION_SHORT = re.compile(r"\b(?:short|quick|brief)\b", re.I)
DURATION_LONG = re.compile(r"\b(?:long|extended|lengthy|epic)\b", re.I)
DURATION_UNDER = re.compile(r"\b(?:under|less\s+than|below|<\s*)(\d+)\s*(?:min|minute|m\b|m,)", re.I)
DURATION_OVER = re.compile(r"\b(?:over|more\s+than|above|>\s*)(\d+)\s*(?:min|minute|m\b|m,)", re.I)

# Artist exclusion / something new
ARTIST_EXCL_PHRASES = [
    "different artist", "different artists", "different band", "different bands",
    "new artist", "new artists", "another artist", "another band",
    "not the same", "not by", "but not", "except", "other than",
    "switch artists", "switch it up", "change artists", "new band",
    "diverse artists", "variety", "mix it up",
]
ARTIST_EXCL = re.compile(r"\b(" + "|".join(re.escape(p) for p in ARTIST_EXCL_PHRASES) + r")\b", re.I)

# Tag vocabulary (lower-cased)
TAG_VOCAB = {
    # mood / energy
    "upbeat", "energetic", "uplifting", "feel-good", "feel good", "high energy",
    "chill", "chillout", "relaxed", "relaxing", "mellow", "calm", "soothing",
    "sad", "melancholy", "melancholic", "emotional", "moody", "somber",
    "happy", "cheerful", "joyful", "fun", "playful",
    "intense", "powerful", "epic", "dramatic", "aggressive",
    "dark", "edgy", "haunting",
    "romantic", "love song", "love songs", "sensual",
    # tempo / dynamics
    "fast", "slow", "mid-tempo", "uptempo", "downtempo", "danceable",
    # production / instrumentation
    "instrumental", "acoustic", "unplugged", "live", "studio",
    "synth", "synthwave", "acapella", "orchestral",
    # genre (top-level)
    "rock", "pop", "hip-hop", "hip hop", "rap", "trap", "country", "jazz",
    "blues", "folk", "electronic", "edm", "house", "techno", "indie", "metal",
    "rnb", "r&b", "soul", "reggae", "punk", "classical", "ambient", "lo-fi",
    "lofi", "alternative", "alt", "funk", "disco", "gospel", "latin", "kpop",
    "k-pop", "j-pop", "afrobeats", "afrobeat", "dancehall", "dub", "drill",
    "grime", "shoegaze", "post-rock", "post rock", "math rock", "emo",
    "hardcore", "screamo", "death metal", "black metal", "doom", "thrash",
    "progressive", "prog", "psychedelic", "garage", "surf", "ska", "swing",
    "bossa nova", "bossa", "samba", "cumbia", "merengue", "salsa", "tango",
    "bluegrass", "americana", "rockabilly", "doo-wop", "doowop",
    # occasion / context
    "party", "workout", "running", "study", "focus", "sleep", "dinner",
    "morning", "evening", "night", "summer", "winter", "fall", "autumn", "spring",
    "wedding", "road trip", "roadtrip", "driving", "commute",
}


def ts():
    return f"[{datetime.now():%H:%M:%S}]"


def parse_year_from_token(s):
    """Return (year_min, year_max, confidence) or None."""
    if not s:
        return None
    s_low = s.lower()
    # decade word
    for w, (lo, hi) in DECADE_WORDS.items():
        if w in s_low:
            return (lo, hi, 0.9)
    # numeric decade
    m = DECADE_NUMERIC.search(s)
    if m:
        d = int(m.group(1))
        return (d, d + 9, 0.95)
    # short decade like "90s" → assume 1990s OR 2090s? Disambiguate by surrounding tokens.
    m = DECADE_SHORT.search(s)
    if m:
        dd = int(m.group(1))
        # heuristic: 00, 10, 20 → 2000s, 2010s, 2020s. Others → 19XX.
        if dd <= 20:
            base = 2000 + dd
        else:
            base = 1900 + dd
        return (base, base + 9, 0.85)
    return None


def extract_constraints(query, played_track_ids, track_meta, track_artist):
    """Return dict of constraints + confidence + which patterns matched."""
    constraints = {
        "year_min": None, "year_max": None,
        "duration_min_ms": None, "duration_max_ms": None,
        "required_tags": [],
        "exclude_played_artists": False,
        "patterns_matched": [],
        "confidences": {},
    }

    q = query

    # ---- Temporal ----
    # decade words / numeric decade / short decade
    yr = parse_year_from_token(q)
    if yr:
        lo, hi, conf = yr
        constraints["year_min"] = lo
        constraints["year_max"] = hi
        constraints["patterns_matched"].append("decade")
        constraints["confidences"]["temporal"] = conf

    # specific year mentions
    if constraints["year_min"] is None:
        m = YEAR_FROM.search(q) or YEAR_IN.search(q)
        if m:
            y = int(m.group(1))
            constraints["year_min"] = y
            constraints["year_max"] = y
            constraints["patterns_matched"].append("specific_year")
            constraints["confidences"]["temporal"] = 0.9
        elif YEAR_BARE.search(q):
            # bare 4-digit year — only count if it's a plausible release year and
            # the query has "release"/"track from"/etc. context. Conservative: skip.
            pass

    # recent / classic / modern
    if constraints["year_min"] is None:
        if RECENT_TOKENS.search(q):
            constraints["year_min"] = 2022
            constraints["year_max"] = 2026
            constraints["patterns_matched"].append("recent")
            constraints["confidences"]["temporal"] = 0.75
        elif CLASSIC_TOKENS.search(q):
            constraints["year_min"] = None
            constraints["year_max"] = 1999
            constraints["patterns_matched"].append("classic")
            constraints["confidences"]["temporal"] = 0.7
        elif MODERN_TOKENS.search(q):
            constraints["year_min"] = 2010
            constraints["year_max"] = 2026
            constraints["patterns_matched"].append("modern")
            constraints["confidences"]["temporal"] = 0.65

    # ---- Duration ----
    m = DURATION_UNDER.search(q)
    if m:
        mins = int(m.group(1))
        constraints["duration_max_ms"] = mins * 60_000
        constraints["patterns_matched"].append("duration_under")
        constraints["confidences"]["duration"] = 0.95
    m = DURATION_OVER.search(q)
    if m:
        mins = int(m.group(1))
        constraints["duration_min_ms"] = mins * 60_000
        constraints["patterns_matched"].append("duration_over")
        constraints["confidences"]["duration"] = 0.95
    if constraints["duration_max_ms"] is None and constraints["duration_min_ms"] is None:
        if DURATION_SHORT.search(q):
            constraints["duration_max_ms"] = 180_000  # 3 min
            constraints["patterns_matched"].append("duration_short_qualitative")
            constraints["confidences"]["duration"] = 0.6
        elif DURATION_LONG.search(q):
            constraints["duration_min_ms"] = 300_000  # 5 min
            constraints["patterns_matched"].append("duration_long_qualitative")
            constraints["confidences"]["duration"] = 0.6

    # ---- Artist exclusion ----
    if ARTIST_EXCL.search(q):
        constraints["exclude_played_artists"] = True
        constraints["patterns_matched"].append("artist_exclusion")
        constraints["confidences"]["artist_exclusion"] = 0.8

    # ---- Tag vocabulary ----
    q_low = " " + q.lower() + " "
    matched_tags = []
    for tag in TAG_VOCAB:
        # word-boundary-ish match
        if " " + tag + " " in q_low or q_low.startswith(" " + tag) or q_low.endswith(tag + " "):
            matched_tags.append(tag)
    # also match plurals like "rocks", "blues" handled by vocabulary already
    if matched_tags:
        constraints["required_tags"] = matched_tags
        constraints["patterns_matched"].append("tag")
        # Confidence: higher if multiple tags match. Tags are noisier than year/duration.
        constraints["confidences"]["tag"] = min(0.85, 0.55 + 0.10 * len(matched_tags))

    # ---- Overall confidence (max across categories) ----
    if constraints["confidences"]:
        constraints["overall_confidence"] = max(constraints["confidences"].values())
    else:
        constraints["overall_confidence"] = 0.0

    constraints["any_constraint"] = bool(constraints["patterns_matched"])
    constraints["high_confidence"] = constraints["overall_confidence"] >= CONFIDENCE_THRESHOLD

    return constraints


def gt_satisfies_constraints(gt_meta, played_artist_set, constraints):
    """Return (satisfies: bool, per-category check dict)."""
    details = {}
    ok = True

    # Year
    gt_year = gt_meta.get("release_year")
    if constraints["year_min"] is not None or constraints["year_max"] is not None:
        if gt_year is None:
            details["year"] = "gt_missing"
            ok = False
        else:
            yr_ok = True
            if constraints["year_min"] is not None and gt_year < constraints["year_min"]:
                yr_ok = False
            if constraints["year_max"] is not None and gt_year > constraints["year_max"]:
                yr_ok = False
            details["year"] = "pass" if yr_ok else "fail"
            if not yr_ok:
                ok = False
    # Duration
    gt_dur = gt_meta.get("duration_ms")
    if constraints["duration_min_ms"] is not None or constraints["duration_max_ms"] is not None:
        if gt_dur is None:
            details["duration"] = "gt_missing"
            ok = False
        else:
            d_ok = True
            if constraints["duration_min_ms"] is not None and gt_dur < constraints["duration_min_ms"]:
                d_ok = False
            if constraints["duration_max_ms"] is not None and gt_dur > constraints["duration_max_ms"]:
                d_ok = False
            details["duration"] = "pass" if d_ok else "fail"
            if not d_ok:
                ok = False
    # Artist exclusion
    if constraints["exclude_played_artists"]:
        gt_artist = gt_meta.get("artist_lower")
        if gt_artist and gt_artist in played_artist_set:
            details["artist_exclusion"] = "fail"
            ok = False
        else:
            details["artist_exclusion"] = "pass"
    # Tags
    if constraints["required_tags"]:
        gt_tags = set(gt_meta.get("tags_lower", []))
        # Match if ANY required tag in gt_tags (OR semantics). This is generous
        # to keep ceiling high; the diagnostic measures ceiling, not lift.
        matched = [t for t in constraints["required_tags"] if t in gt_tags]
        if matched:
            details["tag"] = f"pass({len(matched)})"
        else:
            details["tag"] = "fail"
            ok = False
    return ok, details


def constrained_pool_size_estimator(constraints, track_meta_index, played_set):
    """Estimate how many catalog tracks satisfy the constraints (rough)."""
    if not constraints["high_confidence"]:
        return None
    cnt = 0
    for tid, m in track_meta_index.items():
        if tid in played_set:
            continue
        # year
        if constraints["year_min"] is not None or constraints["year_max"] is not None:
            y = m.get("release_year")
            if y is None:
                continue
            if constraints["year_min"] is not None and y < constraints["year_min"]:
                continue
            if constraints["year_max"] is not None and y > constraints["year_max"]:
                continue
        # duration
        if constraints["duration_min_ms"] is not None or constraints["duration_max_ms"] is not None:
            d = m.get("duration_ms")
            if d is None:
                continue
            if constraints["duration_min_ms"] is not None and d < constraints["duration_min_ms"]:
                continue
            if constraints["duration_max_ms"] is not None and d > constraints["duration_max_ms"]:
                continue
        # required tags (OR semantics)
        if constraints["required_tags"]:
            tags = m.get("tags_lower", set())
            if not any(t in tags for t in constraints["required_tags"]):
                continue
        cnt += 1
    return cnt


def load_track_metadata():
    """Load track_id -> {release_year, duration_ms, artist_lower, tags_lower}."""
    from datasets import DownloadConfig, load_dataset  # type: ignore[reportMissingImports]
    ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
                      download_config=DownloadConfig(local_files_only=True))["all_tracks"]
    meta = {}
    for item in ds:
        tid = str(item["track_id"])
        # Year
        release_date = item.get("release_date")
        year = None
        if release_date:
            s = str(release_date)
            if len(s) >= 4 and s[:4].isdigit():
                year = int(s[:4])
        # Duration ms
        dur = item.get("duration")
        if isinstance(dur, list):
            dur = dur[0] if dur else None
        dur_ms = float(dur) if dur is not None else None
        # Artist (lowercased first artist name)
        artists = item.get("artist_name", [])
        if isinstance(artists, list):
            artist = artists[0] if artists else ""
        else:
            artist = artists or ""
        artist_lower = str(artist).strip().lower() if artist else ""
        # Tags
        tags = item.get("tag_list", [])
        if not isinstance(tags, list):
            tags = [tags] if tags else []
        tags_lower = {str(t).strip().lower() for t in tags if t}
        meta[tid] = {
            "release_year": year,
            "duration_ms": dur_ms,
            "artist_lower": artist_lower,
            "tags_lower": tags_lower,
        }
    return meta


def main():
    t0 = time.time()
    print("R59 Phase A — C2 constraint parser diagnostic (heuristic regex)")
    print("=" * 70)

    print(f"\n{ts()} Loading R12 payload, R21 OOF, R54 Phase 2 OOF...")
    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)
    cases = payload["cases"]
    n = len(cases)
    ta = payload["track_artist"]
    with open(R21_OOF) as f:
        r21_source = json.load(f)
    with open(R54_PHASE2_OOF) as f:
        r54_data = json.load(f)
    r54_source = [[t for t, _ in cl] for cl in r54_data["lists"]]
    print(f"  {n} dev cases")

    print(f"{ts()} Building ALS...")
    als_factors, als_track_ids, als_track_to_idx = build_als()
    als_source = []
    for c in cases:
        played = c["music_turns"]
        sv = als_session_vector(played, als_track_to_idx, als_factors)
        if sv is not None:
            sc = als_factors @ sv
            for t in played:
                if t in als_track_to_idx:
                    sc[als_track_to_idx[t]] = -np.inf
            top = np.argpartition(-sc, 200)[:200]
            top = top[np.argsort(-sc[top])]
            als_source.append([als_track_ids[j] for j in top])
        else:
            als_source.append([])

    print(f"{ts()} Building pool@300 per case (no LR retrain)...")
    pool_per_case = []
    gt_in_pool = np.zeros(n, dtype=bool)
    src_union_has_gt = np.zeros(n, dtype=bool)
    gt_best_source_rank = np.full(n, -1, dtype=np.int64)
    for i, c in enumerate(cases):
        src_lists = {
            "A": payload["src_a"][i], "B": payload["src_b"][i],
            "C": payload["src_c"][i], "D": payload["src_d"][i],
            "F": payload["src_f"][i], "ALS": als_source[i],
            "R21": r21_source[i], "R54": r54_source[i],
        }
        pool = weighted_rrf(src_lists, SW, topk=POOL_K, k=RRF_K)
        pool_per_case.append(pool)
        gt = c["gt"]
        if gt in pool:
            gt_in_pool[i] = True
        # source union check + best source rank
        best_rank = 10**9
        for sname, sl in src_lists.items():
            if gt in sl:
                src_union_has_gt[i] = True
                rk = sl.index(gt) + 1
                if rk < best_rank:
                    best_rank = rk
        if best_rank < 10**9:
            gt_best_source_rank[i] = best_rank

    pool_hit = float(gt_in_pool.mean())
    src_union_hit = float(src_union_has_gt.mean())
    print(f"  pool_hit@300 = {pool_hit:.4f}  src_union_hit = {src_union_hit:.4f}")

    # Bucket assignment (approximate: HIT = in pool@300; DEMOTED is folded into HIT here
    # because we are not running LR for this diagnostic. POOL_MISS and UNREACHABLE are exact.)
    bucket = []
    for i in range(n):
        if gt_in_pool[i]:
            bucket.append("IN_POOL")  # = HIT + DEMOTED for diagnostic purposes
        elif src_union_has_gt[i]:
            bucket.append("POOL_MISS")
        else:
            bucket.append("UNREACHABLE")
    bucket_counts = Counter(bucket)
    print(f"  Bucket counts: {dict(bucket_counts)}")

    print(f"\n{ts()} Loading track metadata (release_year, duration, artist, tags)...")
    track_meta = load_track_metadata()
    print(f"  {len(track_meta)} tracks indexed")

    print(f"\n{ts()} Extracting constraints from {n} user queries...")
    constraint_results = []
    extraction_categories = Counter()
    any_constraint_count = 0
    high_conf_count = 0
    for i, c in enumerate(cases):
        q = c["user_query"]
        played_artists = {ta.get(t, "").strip().lower() for t in c["music_turns"]} - {""}
        # also use the metadata artist as fallback (richer)
        played_artists |= {track_meta.get(t, {}).get("artist_lower", "") for t in c["music_turns"]} - {""}
        constraints = extract_constraints(q, c["music_turns"], track_meta, ta)
        constraint_results.append(constraints)
        if constraints["any_constraint"]:
            any_constraint_count += 1
            for p in constraints["patterns_matched"]:
                extraction_categories[p] += 1
        if constraints["high_confidence"]:
            high_conf_count += 1
    print(f"  ANY constraint extracted: {any_constraint_count}/{n} ({any_constraint_count / n:.1%})")
    print(f"  HIGH-confidence (>= {CONFIDENCE_THRESHOLD}): {high_conf_count}/{n} ({high_conf_count / n:.1%})")
    print("  Per-category extraction counts:")
    for k, v in extraction_categories.most_common():
        print(f"    {k:>30s}: {v}")

    print(f"\n{ts()} Checking GT compliance with extracted constraints...")
    compliance_by_bucket = defaultdict(Counter)
    compliance_details = []
    for i, c in enumerate(cases):
        constraints = constraint_results[i]
        if not constraints["high_confidence"]:
            compliance_details.append(None)
            continue
        gt = c["gt"]
        gt_meta = track_meta.get(gt, {})
        played_artists = {ta.get(t, "").strip().lower() for t in c["music_turns"]} - {""}
        played_artists |= {track_meta.get(t, {}).get("artist_lower", "") for t in c["music_turns"]} - {""}
        ok, details = gt_satisfies_constraints(gt_meta, played_artists, constraints)
        compliance_by_bucket[bucket[i]]["total_with_constraint"] += 1
        if ok:
            compliance_by_bucket[bucket[i]]["gt_complies"] += 1
        compliance_details.append({"satisfies": ok, "details": details})

    print("  GT compliance rates (high-confidence constraints only):")
    for b in ("IN_POOL", "POOL_MISS", "UNREACHABLE"):
        cnt = compliance_by_bucket[b]
        tot = cnt.get("total_with_constraint", 0)
        gt_ok = cnt.get("gt_complies", 0)
        rate = gt_ok / tot if tot else 0
        print(f"    {b:>12s}: {gt_ok}/{tot} = {rate:.1%}")

    # ---- Hypothetical admission (Metric 3, the headline) ----
    print(f"\n{ts()} Computing hypothetical admission rate on POOL_MISS + UNREACHABLE...")
    # For POOL_MISS: admittable if GT satisfies constraints AND GT's best-source rank < 100
    # For UNREACHABLE: admittable if constrained_pool_size < 1000 AND GT satisfies
    pool_miss_admittable = 0
    unreachable_admittable = 0
    pool_miss_constrained_w_compliance = 0
    pool_miss_constrained_no_compliance = 0
    unreachable_constrained_w_compliance = 0
    unreachable_constrained_no_compliance = 0

    # Cache constrained-pool-size lookups (one per unique constraint key)
    # We rebuild per case because constraints are per-case; cache only across exact duplicates.
    constrained_size_cache = {}

    for i, c in enumerate(cases):
        if bucket[i] not in ("POOL_MISS", "UNREACHABLE"):
            continue
        constraints = constraint_results[i]
        if not constraints["high_confidence"]:
            continue
        cdet = compliance_details[i]
        if cdet is None:
            continue
        if not cdet["satisfies"]:
            if bucket[i] == "POOL_MISS":
                pool_miss_constrained_no_compliance += 1
            else:
                unreachable_constrained_no_compliance += 1
            continue
        # GT satisfies
        if bucket[i] == "POOL_MISS":
            pool_miss_constrained_w_compliance += 1
            # Proxy: best-source rank < 100 → assume constrained RRF would promote
            if gt_best_source_rank[i] > 0 and gt_best_source_rank[i] < 100:
                pool_miss_admittable += 1
        else:  # UNREACHABLE
            unreachable_constrained_w_compliance += 1
            # Proxy: constrained_pool_size < 1000 AND GT satisfies
            ck = (constraints["year_min"], constraints["year_max"],
                  constraints["duration_min_ms"], constraints["duration_max_ms"],
                  tuple(constraints["required_tags"]),
                  constraints["exclude_played_artists"])
            played_set = set(c["music_turns"])
            if ck not in constrained_size_cache:
                sz = constrained_pool_size_estimator(constraints, track_meta, played_set)
                constrained_size_cache[ck] = sz
            else:
                sz = constrained_size_cache[ck]
            if sz is not None and sz < 1000:
                unreachable_admittable += 1

    total_admittable = pool_miss_admittable + unreachable_admittable
    print("  POOL_MISS:")
    print(f"    high-conf constraint + GT satisfies: {pool_miss_constrained_w_compliance}")
    print(f"    of those, best-source-rank < 100:    {pool_miss_admittable}")
    print("  UNREACHABLE:")
    print(f"    high-conf constraint + GT satisfies: {unreachable_constrained_w_compliance}")
    print(f"    of those, constrained pool < 1000:   {unreachable_admittable}")
    print(f"  TOTAL hypothetically admittable: {total_admittable}")

    # False positive rate proxy (Metric 4):
    # Among cases where parser extracted high-conf constraint but GT does NOT satisfy,
    # the parser is "wrong" or "too strict" — would harm if filter applied.
    # Breakdown across all buckets:
    fp_by_bucket = defaultdict(Counter)
    for i, c in enumerate(cases):
        constraints = constraint_results[i]
        if not constraints["high_confidence"]:
            continue
        if compliance_details[i] is None:
            continue
        b = bucket[i]
        fp_by_bucket[b]["total"] += 1
        if not compliance_details[i]["satisfies"]:
            fp_by_bucket[b]["fp"] += 1

    print(f"\n{ts()} False-positive (filter-would-have-dropped-GT) rates:")
    for b in ("IN_POOL", "POOL_MISS", "UNREACHABLE"):
        cnt = fp_by_bucket[b]
        tot = cnt.get("total", 0)
        fp = cnt.get("fp", 0)
        rate = fp / tot if tot else 0
        print(f"  {b:>12s}: {fp}/{tot} = {rate:.1%}")

    # Gate verdict (per C2 design doc §5)
    GATE_PROCEED = 150
    GATE_ARCHIVE = 100
    GATE_TUNE_COMPLIANCE = 0.40
    overall_compliance = (compliance_by_bucket["IN_POOL"].get("gt_complies", 0)
                           + compliance_by_bucket["POOL_MISS"].get("gt_complies", 0)
                           + compliance_by_bucket["UNREACHABLE"].get("gt_complies", 0))
    overall_constrained = sum(c.get("total_with_constraint", 0) for c in compliance_by_bucket.values())
    overall_compliance_rate = overall_compliance / overall_constrained if overall_constrained else 0

    print(f"\n{'=' * 70}")
    print("GATES (per docs/r59_candidates/c2_entity_constraint_parser.md §5):")
    print(f"  Metric 3 (hypothetical admittable POOL_MISS+UNREACHABLE): {total_admittable}")
    print(f"    PROCEED gate (>= {GATE_PROCEED}): {'PASS' if total_admittable >= GATE_PROCEED else 'FAIL'}")
    print(f"    ARCHIVE gate (< {GATE_ARCHIVE}):  {'TRIGGERED — ARCHIVE' if total_admittable < GATE_ARCHIVE else 'not triggered'}")
    print(f"  Metric 2 (overall GT compliance): {overall_compliance_rate:.1%}")
    print(f"    TUNE threshold (< {GATE_TUNE_COMPLIANCE:.0%}): {'TRIGGERED — parser too strict/noisy' if overall_compliance_rate < GATE_TUNE_COMPLIANCE else 'not triggered'}")

    if total_admittable >= GATE_PROCEED and overall_compliance_rate >= GATE_TUNE_COMPLIANCE:
        verdict = "PROCEED"
    elif total_admittable < GATE_ARCHIVE:
        verdict = "ARCHIVE"
    elif overall_compliance_rate < GATE_TUNE_COMPLIANCE:
        verdict = "TUNE_REQUIRED"
    else:
        verdict = "BORDERLINE"
    print(f"\n  OVERALL VERDICT: {verdict}")

    # Save
    out_data = {
        "n_cases": n,
        "buckets": dict(bucket_counts),
        "pool_hit_at_300": pool_hit,
        "src_union_hit": src_union_hit,
        "extraction": {
            "any_constraint": any_constraint_count,
            "high_confidence": high_conf_count,
            "per_category_counts": dict(extraction_categories),
        },
        "compliance_by_bucket": {b: dict(c) for b, c in compliance_by_bucket.items()},
        "overall_compliance_rate": overall_compliance_rate,
        "admission": {
            "pool_miss_constrained_w_compliance": pool_miss_constrained_w_compliance,
            "pool_miss_admittable": pool_miss_admittable,
            "unreachable_constrained_w_compliance": unreachable_constrained_w_compliance,
            "unreachable_admittable": unreachable_admittable,
            "total_admittable": total_admittable,
        },
        "false_positive_rates": {
            b: {"total": c.get("total", 0), "fp": c.get("fp", 0)} for b, c in fp_by_bucket.items()
        },
        "gates": {
            "proceed_threshold": GATE_PROCEED,
            "archive_threshold": GATE_ARCHIVE,
            "tune_compliance_threshold": GATE_TUNE_COMPLIANCE,
            "verdict": verdict,
        },
        "elapsed_s": time.time() - t0,
        "created_at": datetime.now().isoformat(),
        "notes": (
            "Heuristic regex extractor only. No LLM, no training, no API. "
            "Bucket IN_POOL = HIT + DEMOTED (LR not run). POOL_MISS and UNREACHABLE "
            "are exact. Admission proxies: POOL_MISS uses best-source-rank < 100; "
            "UNREACHABLE uses constrained_pool_size < 1000 (much weaker)."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out_data, f, indent=2, default=str)
    print(f"\n{ts()} Saved: {OUT}  elapsed={time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
