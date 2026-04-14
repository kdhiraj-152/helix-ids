#!/usr/bin/env python3
"""
DEMONSTRATION: Root-Cause Reducer in Action

Shows the complete flow from diagnostics → root-cause extraction → action directives.

Run: python3 scripts/demo_root_cause_reducer.py
"""

from __future__ import annotations

from pathlib import Path
import sys
import json
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from helix_ids.data.learnability_contract import (
    load_meta,
    derive_root_cause,
    get_action_directive,
    create_summary,
    format_failure_message,
)


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")

def print_diagnostics(meta: dict) -> None:
    """Print loaded diagnostics section."""
    print_section("1. LOADED DIAGNOSTICS")
    print(f"Dataset: {meta.get('dataset', 'unknown')}")
    print(f"Validated: {meta.get('validated', False)}")
    print(f"Linear Probe Macro F1: {meta.get('linear_probe_macro_f1', 0.0):.4f}")
    print(f"Unique Pred Coverage: {meta.get('unique_pred_coverage', 0.0):.4f}")
    print(f"Centroid Min Distance: {meta.get('centroid_min_distance', 0.0):.6f}")
    print(f"Random Macro F1: {meta.get('random_macro_f1', 0.0):.4f}")
    
    violations = meta.get("violations", {})
    if violations.get("BLOCKER"):
        print(f"\n🚫 BLOCKER violations ({len(violations['BLOCKER'])}):")
        for v in violations["BLOCKER"][:5]:
            print(f"   - {v}")
        if len(violations["BLOCKER"]) > 5:
            print(f"   ... and {len(violations['BLOCKER']) - 5} more")


def print_stage_diagnostics(meta: dict) -> None:
    """Print stage diagnostics section."""
    print_section("2. STAGE DIAGNOSTICS")
    stage_diags = meta.get("stage_diagnostics", {})
    for stage_name, stage_data in stage_diags.items():
        print(f"Stage: {stage_name}")
        print(f"  Macro F1: {stage_data.get('macro_f1', 0.0):.4f}")
        print(f"  Feature Variance: min={min(stage_data.get('feature_variance', [0.0])):.6f}, "
              f"max={max(stage_data.get('feature_variance', [1.0])):.6f}")
        
        dropped = stage_data.get("dropped_features", [])
        if dropped:
            print(f"  Dropped Features: {dropped}")
        
        zero_var = stage_data.get("zero_variance_features", [])
        if zero_var:
            print(f"  Zero Variance: {len(zero_var)} features")


def print_root_cause(meta: dict) -> Any:
    """Print root-cause extraction and return root_cause dict."""
    print_section("3. DETERMINISTIC ROOT-CAUSE EXTRACTION")
    if isinstance(meta.get("diagnosis"), dict):
        root_cause = meta["diagnosis"]
    else:
        root_cause = derive_root_cause(meta)
    print(f"Primary Cause: {root_cause['primary']}")
    print(f"Confidence: {root_cause['confidence']:.2f} (0-1 scale)")
    print(f"Failure Stage: {root_cause['stage']}")
    
    if root_cause.get("secondary"):
        print("\nSecondary Causes:")
        for cause in root_cause["secondary"]:
            print(f"  • {cause}")
    
    if root_cause.get("kill_list"):
        print("\nFeature Kill List (responsible for collapse):")
        for feat in root_cause["kill_list"]:
            print(f"  - {feat}")
    return root_cause


def print_action_directive(root_cause: dict) -> Any:
    """Print action directive section and return action."""
    print_section("4. AUTO-GENERATED ACTION DIRECTIVE")
    action = get_action_directive(root_cause, context={})
    print(f"Action Type: {action['type']}")
    print(f"Target Stage: {action['target_stage']}")
    if action.get("target_features"):
        print(f"Target Features: {', '.join(action['target_features'])}")
    print(f"Rationale: {action['rationale']}")
    
    print("\n💡 Translation:")
    print(f"   The system detected {root_cause['primary']}")
    print(f"   Recommended fix: {action['type']}")
    if action.get("target_features"):
        print(f"   Focus on: {', '.join(action['target_features'])}")
    return action


def print_summary(meta: dict) -> Any:
    """Print summary section and return summary."""
    print_section("5. COMPRESSED SUMMARY FOR CI")
    summary = create_summary(meta)
    print(f"Status: {summary['status']}")
    print(f"Primary Issue: {summary['primary_issue']}")
    print(f"Stage: {summary['stage']}")
    print(f"Action: {summary['action']}")
    print(f"Confidence: {summary['confidence']:.2f}")
    
    if summary.get("kill_list"):
        print(f"Kill List: {', '.join(summary['kill_list']) or 'None'}")
    
    print_section("6. ERROR MESSAGE FOR LOGS")
    msg = format_failure_message(summary)
    print(msg)
    return summary


def main() -> None:
    artifact_dir = PROJECT_ROOT / "data" / "processed" / "multi_dataset_v1"
    
    if not artifact_dir.exists():
        print(f"❌ Artifact directory not found: {artifact_dir}")
        return
    
    # Load the meta.json with all diagnostics
    try:
        meta = load_meta(artifact_dir=artifact_dir)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return
    
    print_diagnostics(meta)
    print_stage_diagnostics(meta)
    root_cause = print_root_cause(meta)
    action = print_action_directive(root_cause)
    summary = print_summary(meta)
    
    error_msg = format_failure_message(summary)
    print("RuntimeError raised during training:\n")
    print(error_msg)
    
    print_section("7. COMPLETE SUMMARY (JSON)")
    
    print(json.dumps(summary, indent=2, sort_keys=True))
    
    print_section("8. COMPARISON: BEFORE vs AFTER")
    
    print("BEFORE (Manual Interpretation Needed):")
    print("  RuntimeError: Dataset learnability contract invalid: validated=false")
    print("  (violations=['linear_probe_macro_f1_below_min', ...])")
    print("\nDeveloper must:")
    print("  1. Read all violation codes")
    print("  2. Look up what each means")
    print("  3. Trace through meta.json")
    print("  4. Identify root cause manually")
    print("  5. Decide what to fix")
    
    print("\nAFTER (Deterministic Diagnosis):")
    print(f"  RuntimeError: {error_msg.split(chr(10))[0]}")
    print(f"  Primary: {root_cause['primary']}")
    print(f"  Action: {action['type']}")
    print("\nDeveloper immediately knows:")
    print(f"  1. Root cause: {root_cause['primary']}")
    print(f"  2. What to do: {action['type']}")
    print(f"  3. Confidence: {root_cause['confidence']:.0%}")
    print("  4. (Optional) Which features to look at")
    
    print_section("9. DECISION FLOW VISUALIZED")
    
    print("""
Diagnostic Chain (Deterministic, No Ambiguity):
┌─ centroid_min_distance < 0.01? 
│  └─ YES → feature_space_collapse ✓
│          (This is what happened)
├─ unique_pred_coverage < 0.80? → class_prediction_collapse
├─ zero_variance_fraction > 0.10? → feature_degeneracy
├─ centroid_shrinkage_ratio < 0.30? → scaling_destruction
├─ label_entropy low? → label_distribution_issue
├─ macro_f1 < random*2.5? → weak_signal
└─ else? → unknown
    """)
    
    print_section("10. KEY FEATURES")
    
    features = [
        ("🔍 Root Cause Detection", "Deterministic hard rules, no ML"),
        ("🎯 Action Synthesis", "Maps root cause → specific action"),
        ("📊 Feature Attribution", "Kill list identifies culprit features"),
        ("🔐 Confidence Scoring", "0.85-0.95 depending on signal strength"),
        ("📋 Summary Layer", "Compressed output for CI integration"),
        ("🚨 Enhanced Errors", "Messages include diagnosis + action"),
        ("✅ Zero Ambiguity", "Single primary cause, no guessing"),
        ("⚡ No Manual Work", "System tells you exactly what to fix"),
    ]
    
    for feature, description in features:
        print(f"{feature}: {description}")
    
    print_section("11. NEXT STEPS (FOR DEVELOPERS)")
    
    print("To fix the issue, the engineer should:")
    print(f"1. Understand root cause: {root_cause['primary']}")
    print(f"2. Execute action: {action['type']}")
    if action.get("target_features"):
        print(f"3. Focus on features: {', '.join(action['target_features'])}")
    print("4. Re-run validation: python3 scripts/validation/validate_unsw_learnability.py")
    print("5. Verify: Check summary status changes to PASS")
    
    print_section("12. DOCUMENTATION")
    
    print("For more details, see:")
    print("  • docs/ROOT_CAUSE_REDUCER.md - Full system documentation")
    print("  • tests/test_data/test_root_cause_reducer.py - Unit tests")
    print("  • src/helix_ids/data/learnability_contract.py - Implementation")


if __name__ == "__main__":
    main()
