"""Resume, validate, and report the completed BIOF3001 pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from complete_pipeline import (
    PipelineContext,
    build_stages,
    run_stages,
    validate_completion,
    write_completion_report,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workshop", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--allow-reference-fallback", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    context = PipelineContext(
        workshop=args.workshop.resolve(),
        reference=args.reference.resolve(),
        allow_reference_fallback=args.allow_reference_fallback,
    )
    state_path = context.workshop / "outputs" / "pipeline_state.json"
    try:
        state = run_stages(build_stages(), context, state_path, resume=args.resume)
        validate_completion(context.workshop)
        report = write_completion_report(context, state)
    except Exception as exc:
        print(f"Pipeline incomplete: {exc}", file=sys.stderr)
        return 1
    print(f"Pipeline complete: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
