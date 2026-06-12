# Draft — Organizer Clarification Question (external data & session matching)

**Purpose:** before investing in any source-session reconstruction, obtain an explicit ruling from the
Music-CRS Challenge organizers on whether external-data session matching is permitted. Post to the
challenge forum (or email organizers). Send the *short version*; the *context* is for our records.

---

## Short version (to send)

> **Subject: Rules clarification — use of external public listening datasets to match challenge sessions**
>
> Hi organizers,
>
> Could you clarify what the challenge rules permit regarding **external data**? Specifically:
>
> 1. **Is the use of external, publicly available datasets allowed at all** for building submissions
>    (the Terms describe the provided data and its CC-BY-NC license but don't explicitly address
>    supplementing it)?
>
> 2. If external data is allowed in general, is the following **specific approach** permitted, or would
>    it be considered **outside the intended evaluation protocol** (including under the bad-faith
>    "gaming the evaluation" clause)? The TalkPlayData 2 paper states the data is derived from real
>    LFM-2b listening sessions and that the recommendation candidate pool (and hence the ground truth)
>    is sampled from each user's session. One could take a **lawfully accessible external listening
>    dataset**, and **match an anonymized challenge conversation to its underlying real listening
>    session** — using the provided demographics, `session_date`, and the tracks played in the
>    conversation as a fingerprint — in order to **infer that conversation's candidate pool / likely
>    ground-truth track** rather than predict it from the conversation alone.
>
> To be clear, **we have not used any hidden labels and are asking before implementing or submitting
> anything of this kind.** We want to stay within the spirit of the challenge. Is session matching /
> source-session reconstruction of this kind **allowed, disallowed, or out of scope**? And are there
> any constraints on which external datasets may be used?
>
> Thank you!

---

## Context / our records (do not send)

- Why we ask: the hidden GT is a single track sampled from a per-conversation 16–32 pool that is itself
  sampled from a real session (arXiv 2509.09685v5, App. B). Conversation semantics cannot identify
  which sampled track is the GT (we measured `rec@1≈0` even with a strong instruction reranker, R480),
  so the only mechanism that closes the leaderboard gap is recovering the pool/session — which requires
  matching to the underlying listening data.
- Integrity exposure: the official Terms (nlp4musa.github.io/music-crs-challenge/terms.html) don't name
  external data but include "gaming the evaluation in bad faith → disqualification" and require winners
  to submit code for verification. Session matching is de-anonymization of the withheld split, so a
  ruling is needed before any investment.
- Data reality (for our own go/no-go even if allowed): LFM-2b is withdrawn ("license issues"); the only
  legally-clean, era-relevant source is MLHD/CC0 (2005–2014), which covers only the ~27 Blind-A rows
  dated 2009–2014, is MBID-only, and is a *different* Last.fm crawl (the exact session may not be present).
- If organizers say **disallowed/out-of-scope** → reconstruction is closed; R453 (composite 0.7536) is
  the legitimate-data anchor and nDCG is treated as capped.
- If organizers say **allowed** → proceed to the MLHD feasibility probe (can a fingerprint uniquely
  resolve a session?), still labeled and audited carefully.
