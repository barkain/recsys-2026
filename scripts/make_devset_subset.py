"""Create deterministic devset session subsets for cheap paired experiments."""
import argparse
import os
import random
from pathlib import Path

from datasets import Dataset, DownloadConfig, load_dataset


DATASET_NAME = "talkpl-ai/TalkPlayData-Challenge-Dataset"
DEFAULT_HF_CACHE = Path.home() / ".cache" / "huggingface" / "datasets"


def cached_test_arrow_path() -> str | None:
    matches = sorted(
        DEFAULT_HF_CACHE.glob(
            "talkpl-ai___talk_play_data-challenge-dataset/default/*/*/*-test.arrow"
        )
    )
    return str(matches[-1]) if matches else None


def load_devset(allow_network: bool, gt_arrow: str | None):
    if gt_arrow:
        return Dataset.from_file(gt_arrow)

    arrow_path = cached_test_arrow_path()
    if arrow_path and not allow_network:
        return Dataset.from_file(arrow_path)

    return load_dataset(
        DATASET_NAME,
        split="test",
        download_config=DownloadConfig(local_files_only=not allow_network),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True)
    parser.add_argument("--allow_network", action="store_true")
    parser.add_argument("--gt_arrow", default=None)
    args = parser.parse_args()

    ds = load_devset(args.allow_network, args.gt_arrow)
    n = min(args.size, len(ds))
    indices = sorted(random.Random(args.seed).sample(range(len(ds)), n))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for i in indices:
            f.write(f"{ds[i]['session_id']}\t{ds[i].get('user_id', '')}\n")
    print(f"Wrote {n} case IDs to {args.out} (seed={args.seed})")


if __name__ == "__main__":
    main()
