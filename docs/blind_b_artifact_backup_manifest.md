# Blind-B Critical-Artifact Backup Manifest

**Created:** 2026-06-04
**Why:** The Mac disk is tight and these production / Blind-B-critical caches are **single-copy**
(not in git, not otherwise duplicated). Losing them would force expensive retrains. This
packages them for off-machine backup to Google Drive.

- **Local staging:** `.scratch/backup_critical_artifacts_2026_06/`
- **Drive target:** `MyDrive/recsys_critical_artifacts_2026_06/`
- **Checksums:** `.scratch/backup_critical_artifacts_2026_06/SHA256SUMS.txt`
- **Machine-readable:** `exp/eval/blind_b_artifact_backup_manifest.json`

## Tarballs

| tarball | size | entries | sha256 (first 20) | sources |
|---|---|---|---|---|
| `r54_phase3_fold0_model_and_embs.tar.gz` | 538.6 MB | 16 | `6c599f8d4f02ea57f2fc` | `cache/r54/phase3_smoke/fold_0/` (model + track_embs.npy + oof_lists) |
| `r54_phase3_lr_support.tar.gz` | 36.4 MB | 4 | `3c48207b2ae64fbd3286` | R54c LR (`r54_phase3_lr_model.txt`) + `als.npz` + `payload_maps.pkl` + `track_pop.json` |
| `production_retrieval_lists.tar.gz` | 2.7 MB | 5 | `d610fb9eebc7d1b08fb1` | `cache/r54_production/` + `cache/r84_production/` + `cache/blind_a/source_cache.pkl` |
| `r21_production.tar.gz` | 1.81 GB | 32 | `b117dd289e0fb740ee77` | `cache/r21_production/` (R21/R39 source, 1 of 8) |

Total local: ~2.4 GB. Full sha256 in `SHA256SUMS.txt`.

## Already on Drive (NOT re-duplicated)

- **`r54_phase3_full_folds1_4.zip`** — folds 1–4 of the R54 ensemble (model.safetensors +
  track_embs.npy + oof_lists.json each). Restored + validated 2026-06-04: reproduces the
  Blind-A R54 lists **80/80 sessions identical**. This is why folds 1–4 are not re-tarred here.
- **`r54_phase3_artifacts.zip`** — earlier R54 phase-3 bundle.

Together with `r54_phase3_fold0_model_and_embs.tar.gz` (this backup), all 5 ensemble folds
are now covered off-machine.

## Restore

Extract any tarball at the **repo root** (paths are repo-relative):
```
tar -xzf r54_phase3_fold0_model_and_embs.tar.gz     # -> cache/r54/phase3_smoke/fold_0/
tar -xzf r54_phase3_lr_support.tar.gz               # -> cache/r54_phase3_*
tar -xzf production_retrieval_lists.tar.gz          # -> cache/r54_production/, r84_production/, blind_a/
tar -xzf r21_production.tar.gz                      # -> cache/r21_production/
```
Verify against `SHA256SUMS.txt`: `shasum -a 256 -c SHA256SUMS.txt`.

## Manual upload (rclone here is read-only → upload is manual)

Upload these 5 files from `.scratch/backup_critical_artifacts_2026_06/` to
`MyDrive/recsys_critical_artifacts_2026_06/`:
1. `r54_phase3_fold0_model_and_embs.tar.gz`
2. `r54_phase3_lr_support.tar.gz`
3. `production_retrieval_lists.tar.gz`
4. `r21_production.tar.gz`
5. `SHA256SUMS.txt`

Local tarballs are retained until upload is confirmed.
