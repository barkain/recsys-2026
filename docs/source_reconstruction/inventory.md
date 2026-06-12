# Source-Session Reconstruction — Research Track (Inventory + Verdict)

**Date:** 2026-06-12 · **Status:** research only (no Blind-A variants, no model training)
**Pivot rationale:** arXiv v5 (TalkPlayData 2, 2509.09685) — the challenge is built on **LFM-2b**;
each conversation draws a real listening session (≥21 tracks), samples profile tracks, and samples
a **16–32 track recommendation pool**; the Recsys-LLM (and hence the hidden GT) is constrained to
that pool (Appendix B). So nDCG is bounded by *pool identity*, not semantics — consistent with our
reranker runs (R480/R481): a strong instruction reranker hits `rec@1≈0`, i.e. it cannot identify which
of the hidden 16–32 set is the GT. The reconstruction question: can we recover that pool per Blind-A row
from lawfully accessible data?

## 0. CORRECTION to the premise (measured locally, decisive)

- **Blind-A `session_date` range = 2009-06-16 .. 2018-12-28** (years: 2009×2, 2010×1, 2011×3, 2012×8,
  2013×3, 2014×10, 2015×8, 2016×6, 2017×18, 2018×21). **NOT post-2019.** The paper's "sessions after
  2019" does not describe this challenge's blind split.
- Implication: the relevant external window is **2009–2018**. Post-2019 sources (ListenBrainz, likely
  Music4All-Onion) are the *wrong era*. Pre-2014 sources (**MLHD**, non-commercial license — not CC0)
  become era-relevant for the ~27 rows dated 2009–2014.
- **No candidate-pool field is leaked in the provided data.** Challenge dataset columns:
  `session_id, user_id, session_date, user_profile, conversation_goal, conversations,
  goal_progress_assessments`. No `available_tracks`/`pool`/`candidates`.
- **No 5 profile tracks in `user_profile`** — only demographics: `age, age_group, country_code,
  country_name, gender, preferred_language, preferred_musical_culture`. Per-row fingerprint =
  demographics + `session_date` + played tracks in the conversation (strong for multi-turn rows,
  weak/absent for turn-1 cold rows).

## 1. Rules / integrity gate — VERDICT: HIGH-RISK, clear with organizers first

Official terms (https://nlp4musa.github.io/music-crs-challenge/terms.html), via web research:
- External data: **UNSPECIFIED** (not explicitly allowed or forbidden; no "provided-data-only" clause).
- BUT: **"Cheating … or gaming the evaluation in bad faith will result in disqualification"** (§3d);
  winners **submit code for verification** (§6); "academic challenge run in good faith / same spirit" (§9).
- Recovering the hidden GT by matching an anonymized session to its source LFM-2b session is
  **de-anonymization of the withheld split** — mechanically a leak, not a modeling win — and is the
  archetype of "gaming the evaluation." It is **discoverable** (winner code audit).
- Source license: **LFM-2b is withdrawn "due to license issues"**; challenge data is **CC-BY-NC-ND**
  ("do not redistribute outside the challenge"). Submission attests "does not violate IP" (§6).
- **Read:** no explicit rule broken, but against the spirit, exposed under the bad-faith clause, and
  legally encumbered. Not submission-safe without explicit organizer clearance. (Confirms prior
  `external reconstruction NOT_VIABLE`.) **This gate applies even with a cleanly-licensed external
  dataset, because the potentially-disallowed act is recovering the hidden answer, not the data license itself.**

## 2. Dataset inventory (corrected for the 2009–2018 window)

| Dataset | Window | user+ts+track? | titles/Spotify? | demo? | License/availability | Verdict for THIS task |
|---|---|---|---|---|---|---|
| **LFM-2b** | 2005–**2020** (covers all) | yes | titles + Spotify-URIs (4.6M) | yes (country/age/gender) | **WITHDRAWN** ("license issues"); no lawful mirror found (HF/Zenodo/Archive/Torrents) | The only full-coverage source — **unavailable + license-encumbered + rules-exposed** |
| **MLHD / MLHD+** | 2005–**2014** | yes (MBID only) | MBID only (need MB→title join); no Spotify | yes (age/country/gender CSVs) | **non-commercial / no clear open license** (MetaBrainz MLHD+ lists "Licenses: None") — NOT CC0; downloadable | Era-relevant for the ~27 rows in 2009–2014 but **not cleanly-licensed**; different crawl (583k users) → session presence uncertain; MBID-only |
| LFM-1b (+UGP/BeyMS/UserGroups) | 2005–2014 | events: yes (titles); UGP/derivs: aggregate | titles, no Spotify | partial | LFM-1b withdrawn; Zenodo derivatives CC-BY but aggregate/3k-user | LOW (withdrawn base; derivatives too small/aggregate) |
| lastfm-1K (Celma) | ≤2009-05-05 | yes (titles + MBID) | titles, MBID | no | non-commercial, downloadable | **covers 0 Blind-A rows** (ends before earliest Blind-A date 2009-06-16); schema reference only |
| lastfm-360K (Celma) | ~2009 | **no timestamps** | artist only | yes | non-commercial | NONE (aggregate) |
| **Music4All-Onion** | likely ~2019–2021 (unconfirmed; **wrong era**) | yes (internal IDs) | via catalog join; ~109k tracks (filtered) | partial | CC-BY-4.0, downloadable | Likely wrong era for 2009–2018; deprioritized |
| ListenBrainz dumps | 2005–present (**post-2019 strong**) | yes (MBID+Spotify) | yes | partial | **CC0** | Wrong **population** (different users) + skewed to wrong era |
| #nowplaying-RS | pre-2019, Twitter | yes (Spotify) | yes | no | CC-BY-4.0 | wrong source population + era |

**Net:** the only full-coverage source (LFM-2b) is withdrawn/license-encumbered/rules-exposed. There is
**no cleanly-licensed, era-relevant source.** The closest era-relevant lead is **MLHD/MLHD+ (2005–2014)**
— covering ~1/3 of Blind-A rows — but it is **non-commercial / no clear open license** (not CC0),
MBID-only, from a *different* Last.fm crawl (so the specific session may not be present), and still
**inside the integrity gate** (de-anonymizing to recover the GT).

## 3. Session-fingerprint plan (if the gate is cleared)

Per Blind-A row: fingerprint = `{age, country, gender, preferred_language/culture}` + `session_date`
+ played-track {title, artist} from the conversation history (and MBID via MusicBrainz). Match against
an external listening corpus to find the user-session window whose plays on/near `session_date` contain
those exact tracks → the session's other tracks approximate the 16–32 pool → GT ∈ pool.

- Multi-turn rows: several played tracks → strong fingerprint.
- Turn-1 cold rows: demographics + date only → likely non-unique → not reconstructable.
- Auditor: `scripts/lfm_source_auditor.py` — header-only field check for any discovered file.

## 4. Success metric (no hidden GT used)

For each Blind-A row, can we narrow to **≤100 plausible candidate tracks** purely from
fingerprint→session matching (no GT lookup in the loop)? Report per-row candidate-set size; a row is
"reconstructed" if size ≤100 and the played history tracks are recovered as a sanity check.
Only AFTER pools exist do we rerank (Qwen / R453 shell).

## 5. Honest verdict

- The **clean** win (pool leaked in provided data) does **not** exist.
- The **full-coverage** external source (LFM-2b) is **withdrawn, license-encumbered, rules-exposed**.
- The closest era-relevant source (**MLHD, 2009–2014, ~27 rows**) is **non-commercial / not cleanly
  licensed**, technically hard (MBID-only, different crawl, session-presence uncertain) **and still
  inside the bad-faith/integrity gate**.
- Recommendation: **do not invest in reconstruction as a submission path without explicit organizer
  clearance.** As pure research, the only candidate experiment is an **MLHD feasibility probe**
  on the 2009–2014 rows (can fingerprints even find a unique session?) — under MLHD's non-commercial
  terms, research-only/not-redistributable — clearly labeled
  not-submission-safe. Otherwise R453 (composite 0.7536) stands and nDCG is treated as capped on
  legitimate data.
