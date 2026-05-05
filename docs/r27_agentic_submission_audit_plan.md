# R27 Agentic Submission Audit Plan

## Goal

Use a small agent team to audit and conservatively improve the current best Blind-A submission.

This is a final-polish process, not a replacement for modeling. The default action for every row is **no change**.

Current anchor submissions:

- Best nDCG track baseline: R21/R25 track IDs, nDCG@20 = 0.4734
- Best composite response baseline: R25, LexDiv = 0.8133, LLM = 4.65, composite = 0.5948
- Best LLM response baseline: R21, LLM = 4.70, LexDiv = 0.7730, composite = 0.5945

Expected value:

- Response polish: small positive composite gain possible.
- Track edits: high variance; only allow high-confidence corrections.

## Non-Negotiable Rules

1. Start from the current best artifact.
2. Do not invent track IDs from external music knowledge.
3. Do not make broad track-list edits.
4. Do not change a row unless there is explicit evidence.
5. Every proposed track change must cite:
   - the user constraint,
   - the current track metadata issue,
   - the replacement candidate metadata,
   - the source candidate list where the replacement appears.
6. Every response change must preserve alignment with the submitted top-1 track.
7. Final artifact must pass full validation before upload.

## Inputs

Use:

```text
exp/inference/blind_a/r25_lexdiv_v2_submission.zip
exp/inference/blind_a/lr_r21_v1_hybrid_submission.zip
```

Recommended additional context per row:

- Blind-A conversation history
- current top-20 tracks and metadata
- candidate pools from R21/V3/R24/R26 if available
- previous R21 and R25 responses

If candidate pools are unavailable, limit track edits to reranking within the submitted top-20 only.

## Output Artifacts

Create:

```text
exp/inference/blind_a/r27_agent_audit_report.json
exp/inference/blind_a/r27_agent_audit_submission.json
exp/inference/blind_a/r27_agent_audit_submission.zip
```

The report should include every reviewed row, whether changed or unchanged.

## Team Roles

### Coordinator

Responsibilities:

- assign rows to reviewers
- enforce no-change default
- merge only high-confidence edits
- run validation
- produce final report

### Track Reviewer Agents

Each reviewer gets 20 rows.

Responsibilities:

- inspect conversation and current top-20
- identify obvious top-1/top-20 contradictions
- propose only conservative track changes
- provide evidence for every proposed change

### Response Reviewer Agents

Each reviewer gets 20 rows.

Responsibilities:

- compare R21 and R25 responses
- flag weak or risky responses
- propose response edits only when quality improves
- preserve LexDiv without sacrificing direct user fit

### Skeptic Agent

Responsibilities:

- review all proposed track edits
- reject speculative changes
- reject edits based on taste rather than evidence
- check for response hallucinations and unsupported claims

## Row Assignment

Assign rows by index:

```text
Agent 1: rows 0-19
Agent 2: rows 20-39
Agent 3: rows 40-59
Agent 4: rows 60-79
Skeptic: review all proposed changes
Coordinator: final merge
```

## Track Edit Policy

Track edits are allowed only in the following cases.

### Allowed Track Changes

1. **Explicit negative constraint violation**

Example:

```text
User: not Noisia / not DmC
Top-1: Noisia from DmC
```

Allowed action:

- rerank a better existing candidate above the violating track
- preferably from current top-20/top-50

2. **Clear genre mismatch**

Example:

```text
User asks for acoustic folk
Top-1 is electronic dance
Candidate rank 3 is acoustic folk by metadata
```

3. **Explicit artist request missed**

Example:

```text
User asks for Kendrick-like or Kendrick track
Current top candidates ignore all relevant artist/genre metadata
```

4. **Response had to apologize or hedge**

If a response says the recommendation is wrong or not aligned, audit the row.

### Disallowed Track Changes

Reject changes based on:

- general music taste
- external knowledge not reflected in catalog metadata
- "this feels better" without constraint evidence
- moving a candidate only because it is popular
- replacing many tracks in a row

### Track Change Threshold

Use three confidence labels:

```text
HIGH: explicit constraint violation or metadata contradiction
MEDIUM: plausible but not certain
LOW: taste-based or weak evidence
```

Only HIGH edits may be merged.

Expected number of accepted track edits:

```text
0-5 rows
```

If reviewers propose more than 10 track edits, the process is too aggressive.

## Response Edit Policy

Response edits are lower risk than track edits but still must be controlled.

### Allowed Response Changes

- remove apology/refusal language
- remove trailing questions
- improve direct user fit
- preserve or improve lexical diversity
- fix track/artist mismatch
- reduce hallucinated details
- replace vague generic prose with grounded metadata-based explanation

### Disallowed Response Changes

- unsupported album/era/production claims
- overly long review-style responses
- responses that do not mention or clearly support the top-1 track
- generic "great choice" filler
- asking follow-up questions instead of recommending

### Response Quality Heuristic

Each response should:

- address the user's request in the first sentence
- name the top-1 track and artist
- cite 1-3 grounded reasons from metadata/context
- avoid trailing questions
- stay between 45 and 130 words unless there is a strong reason

## Review JSON Schema

Each agent should return records like:

```json
{
  "row_index": 29,
  "session_id": "...",
  "turn_number": 7,
  "decision": "change_response|change_tracks|change_both|no_change",
  "confidence": "HIGH|MEDIUM|LOW",
  "issue": "explicit negative constraint violation",
  "evidence": {
    "user_constraint": "user asked to move away from Noisia/DmC",
    "current_top1": {
      "track_id": "...",
      "track_name": "...",
      "artist": "Noisia"
    },
    "replacement": {
      "track_id": "...",
      "track_name": "...",
      "artist": "...",
      "source": "current_top20|R21_pool|BM25|R24|R26"
    }
  },
  "proposed_track_ids": ["optional full 20-list if changed"],
  "proposed_response": "optional response text if changed",
  "rationale": "short explanation"
}
```

## Coordinator Merge Rules

For each row:

1. If no proposed change: keep R25 row.
2. If response-only HIGH/MEDIUM and no hallucination risk: apply.
3. If track-change HIGH and skeptic approves: apply.
4. If track-change MEDIUM/LOW: reject.
5. If disagreement: keep baseline.

Coordinator must record rejected changes in the report.

## Validation Checklist

Run after merging:

```text
zip contains exactly prediction.json
80 rows
80 unique session IDs
same session_id/user_id/turn_number set as baseline
20 predicted_track_ids per row
20 unique IDs per row
all track IDs valid catalog IDs
no empty predicted_response
no comma prefix
no refusal/apology boilerplate
no trailing questions
response length min/max sane
LexDiv proxy compared to R21 and R25
track overlap with R25 reported
top-1 changed count reported
```

If response-only:

```text
track IDs must be bitwise identical to R25
```

If track edits exist:

```text
list changed rows
top-1 changed count <= 5 preferred
mean top-20 overlap with R25 >= 0.95 preferred
```

## Submission Gate

Submit only if one of these holds.

### Response-Only Gate

```text
LexDiv >= R25 - 0.005
LLM-risk checks clean
no track changes
```

### Track+Response Gate

```text
accepted track edits <= 5
all track edits HIGH confidence
skeptic approved all track edits
no response validation failures
```

If no clear improvements are found, do not submit.

## Recommended Workflow For Claude

1. Build a row-inspection bundle:

```text
scripts/build_r27_review_bundle.py
```

Bundle should include row index, conversation, top-20 metadata, R21 response, R25 response, and optional candidate pools.

2. Spawn four reviewer agents and one skeptic.

3. Collect JSON review outputs.

4. Merge conservatively.

5. Validate.

6. Report:

```text
accepted changes
rejected changes
track IDs changed count
top-1 changed count
LexDiv proxy
LLM-risk checks
artifact path
```

7. Do not upload automatically.

## Expected Outcome

Most likely outcome:

- 0-2 response patches
- 0-3 high-confidence track reranks
- small or no measurable gain

This process is worthwhile only as final polish. It is not expected to close the nDCG gap to the leader.
