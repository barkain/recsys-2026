"""Fail-fast connectivity check for paid LLM experiment runs."""
import argparse
import sys

from mcrs.utils import call_llm_api


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="claude-haiku-4-5-20251001")
    args = parser.parse_args()

    is_reasoning = args.model.startswith(("o1", "o3", "o4"))
    max_tokens = 256 if is_reasoning else 8
    text = call_llm_api(
        "Return exactly OK.",
        "Return exactly OK.",
        model=args.model,
        max_tokens=max_tokens,
    )
    if text is None:
        print(f"FAIL: LLM connectivity check failed for {args.model}: {text!r}")
        sys.exit(1)
    if not is_reasoning and "OK" not in text.upper():
        print(f"FAIL: LLM connectivity check failed for {args.model}: {text!r}")
        sys.exit(1)
    print(f"OK: LLM connectivity check passed for {args.model}")


if __name__ == "__main__":
    main()
