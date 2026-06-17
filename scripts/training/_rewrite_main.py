#!/usr/bin/env python3
"""Rewrite main() in train_helix_ids_full.py for Phase 13A-4 extraction."""
import sys
from pathlib import Path

filepath = Path("/Users/kdhiraj/Downloads/RP-2/scripts/training/train_helix_ids_full.py")
content = filepath.read_text("utf-8")
lines = content.splitlines()

# Find def main() and end before if __name__
main_start = None
main_end = None
for i, line in enumerate(lines):
    if line.strip().startswith("def main()"):
        main_start = i
    if main_start is not None and line.strip().startswith('if __name__ == "__main__"'):
        main_end = i
        break

if main_start is None or main_end is None:
    print("ERROR: Could not find main() boundaries")
    sys.exit(1)

print(f"Original main: lines {main_start+1} - {main_end - 1} ({main_end - main_start - 1} lines)")

NEW_MAIN = '''def main():  # NOSONAR   # Phase 13A-4 orchestration extraction
    """Main training entry point. parse -> orchestrate -> exit pattern."""
    from scripts.training.orchestration import parse_config

    parsed = parse_config()
    args = parsed.args

    # Early exit for multi-seed governance mode
    if parsed.governance_only_mode:
        import json
        from scripts.training.orchestration.governance_pipeline import run_multiseed_governance
        gov_result = run_multiseed_governance(parsed)
        print(json.dumps(gov_result.return_payload, indent=2, default=str))
        return gov_result.return_payload

    from scripts.training.train_helix_ids_full import (
        HELIX_FULL_RESULTS_DIR, PROJECT_ROOT,
        _assert_real_dataset_required, _load_precomputed_splits,
        _assert_validated_unsw_artifact,
        setup_logging, _validate_per_dataset_splits,
    )
    from helix_ids.data.learnability_contract import (
        freeze_snapshot_if_valid,
    )
    from helix_ids.governance.determinism import set_global_determinism
    from pathlib import Path
    import os, time
    import numpy as np

    os.environ["HELIX_STRICT_MISSING"] = "1"
    os.environ["STRICT_MISSING"] = "1"
    os.environ["HELIX_SEED"] = str(args.seed)
    set_global_determinism(args.seed)

    split_start = time.perf_counter()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = HELIX_FULL_RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(results_dir)

    if args.dataset is not None:
        _assert_real_dataset_required(
            project_root=PROJECT_ROOT,
            dataset_name=args.dataset,
        )

    logger.info("Decoupled training mode enabled.")
    precomputed_splits_dir = Path(args.precomputed_splits_dir)
    precomputed_splits = None
    if not args.force_recompute_splits:
        precomputed_splits = _load_precomputed_splits(
            splits_dir=precomputed_splits_dir,
            logger=logger,
            expected_feature_dim=None,
        )

    if precomputed_splits is not None:
        logger.info(
            f"Using precomputed per-dataset splits from {precomputed_splits_dir} for isolated training."
        )
        require_frozen_snapshot = str(args.snapshot_mode).strip().lower() == "strict"
        if require_frozen_snapshot:
            freeze_meta = freeze_snapshot_if_valid(artifact_dir=precomputed_splits_dir)
            logger.info(
                "Strict snapshot mode: freeze attempted for learnability contract snapshot_id=%s frozen=%s",
                str(freeze_meta.get("snapshot_id", "")),
                bool(freeze_meta.get("frozen", False)),
            )
        else:
            logger.warning(
                "Running with research_override snapshot mode (allow_unfrozen_snapshot=%s); "
                "reproducibility promotion gates remain disabled for this run.",
                bool(args.allow_unfrozen_snapshot),
            )
        _assert_validated_unsw_artifact(
            splits_dir=precomputed_splits_dir,
            logger=logger,
            require_frozen=require_frozen_snapshot,
        )
        splits = precomputed_splits
    else:
        raise RuntimeError(
            "Training requires validated processed artifacts. "
            "Run preprocessing and scripts/validation/validate_unsw_learnability.py first."
        )

    _validate_per_dataset_splits(
        splits,
        logger=logger,
        seed=args.seed,
        enforce_cross_dataset_scale=False,
    )

    split_end = time.perf_counter()
    split_elapsed = split_end - split_start

    # ----------------------------------------------------------------
    # Orchestration
    # ----------------------------------------------------------------
    from scripts.training.orchestration import run_orchestration
    from scripts.training.orchestration.governance_pipeline import run_governance_pipeline

    orchestration_result = run_orchestration(
        parsed=parsed,
        splits=splits,
        results_dir=results_dir,
        output_dir=output_dir,
        logger=logger,
    )

    gov_result = run_governance_pipeline(
        parsed=parsed,
        orchestration_result=orchestration_result,
        results_dir=results_dir,
        output_dir=output_dir,
        logger=logger,
        split_elapsed=split_elapsed,
        splits=splits,
    )

    return gov_result.return_payload
'''

new_lines = lines[:main_start] + NEW_MAIN.splitlines() + lines[main_end - 1:]
new_content = "\n".join(new_lines)

new_main_lines_count = len(NEW_MAIN.splitlines())
print(f"New main: {new_main_lines_count} lines")
print(f"Total lines: {len(lines)} -> {len(new_lines)}")
print(f"LOC reduction: {main_end - main_start - new_main_lines_count}")

filepath.write_text(new_content + "\n", "utf-8")
print(f"Written to {filepath}")
print("Done")
