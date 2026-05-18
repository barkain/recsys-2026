# Email to RecSys 2026 Music-CRS Organizers: External Metadata Clarification

**DELIVERY METHOD:**  
**Recommended:** Post on Codabench forum at https://www.codabench.org/competitions/15786/ (organizers monitor participant questions there)  
**Alternative:** Email directly if contact address becomes available

---

## Email Header

```
To: TBD — Post on Codabench forum (https://www.codabench.org/competitions/15786/) or email organizers if address provided
From: Nadav Barkai (team: dirac), barkai.nadav@gmail.com
Subject: RecSys Challenge 2026 Music-CRS: Clarification on external metadata sources
```

---

## Email Body

Dear RecSys 2026 Music-CRS organizers,

I am a participant in the Music-CRS challenge (team: dirac, registered on Blind-A) and would like to request clarification regarding the use of external metadata sources.

**Question:** Are participants permitted to augment the provided TalkPlayData track metadata with external metadata sources (e.g., MusicBrainz tags/genres, Spotify audio features, Last.fm tags) for use as additional features in the recommendation model?

**Scenario:** I would use ISRC codes (provided in TalkPlayData at 97.17% coverage) to join with MusicBrainz recordings and retrieve tags, genres, and artist relations. These external metadata fields would be used internally to compute ranking features (e.g., tag overlap between candidate tracks and user listening history). I would not redistribute the augmented dataset. All track recommendations would remain exclusively within the provided TalkPlayData catalog, and track IDs would match TalkPlayData-Challenge-Track-Metadata exactly.

**My understanding of current rules:**
- Track IDs must match those in TalkPlayData-Challenge-Track-Metadata (no external track injection)
- Pre-extracted embeddings are provided to ensure focus on model architecture
- The dataset license is CC-BY-NC-ND-4.0 (NoDerivatives clause)

However, I have not found explicit guidance on whether enriching catalog metadata with external sources for feature engineering is permitted.

**Specific sub-questions:**

(a) Is using external metadata (e.g., MusicBrainz tags, Spotify audio features) as ranking features permitted, provided that recommendations remain within the supplied catalog and we do not redistribute an enriched dataset?

(b) If yes, are there specific disclosure requirements for the submission paper (e.g., documenting external sources, API versions, snapshot dates, or derived features)?

Thank you for your clarification on this matter.

Best regards,  
Nadav Barkai  
Team: dirac  
Email: barkai.nadav@gmail.com

---

## Post-Send Actions

Once response is received:
1. Record ruling in `docs/r59_candidates/c4_rules_verdict.md` (update VERDICT section)
2. If ALLOWED: Proceed to bounded diagnostic (§ "If ALLOWED" in c4_rules_verdict.md)
3. If FORBIDDEN: Archive R59 C4 immediately, document reason
4. If AMBIGUOUS response: Request explicit yes/no confirmation before proceeding

---

**Document created:** 2026-05-16  
**Source:** Polished from draft in `docs/r59_candidates/c4_rules_verdict.md` § "Organizer Question"  
**Instance:** r59-c4-rules-feasibility (af2f7129-41bd-4fdc-bb66-7b6d7d11ba60)
