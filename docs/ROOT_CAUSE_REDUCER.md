"""
ROOT-CAUSE REDUCER: LEARNABILITY CONTRACT DETERMINISTIC DIAGNOSIS ENGINE

PROBLEM STATEMENT
=================

System had:
- Full diagnostics collection ✓
- Violation detection ✓
- Missing: Automatic root-cause extraction + action synthesis

Resulted in:
- Large multi-field meta.json output
- Manual interpretation required
- No actionable failure messages
- CI outputs required expertise to decode

SOLUTION: DETERMINISTIC ROOT-CAUSE EXTRACTION
==============================================

Added 6 new functions to src/helix_ids/data/learnability_contract.py:

1. rank_failure_stages(stage_diagnostics, stage_transitions)
   - Ranks stages by F1 drop magnitude
   - Identifies primary failure stage
   - Returns: dict with 'primary_failure_stage', 'f1_drop', 'stages_ranked'

2. extract_feature_kill_list(stage_diagnostics, top_n=5)
   - Extracts top-N features with highest negative mutual_info_delta
   - These features are responsible for collapse
   - Returns: list of feature names (e.g., ["f_12", "f_87"])

3. derive_root_cause(meta) → PRIMARY FUNCTION
   - Applies hard logical rules (no ML, deterministic)
   - Priority order (no ambiguity):
     1. centroid_min_distance < threshold → feature_space_collapse
     2. unique_pred_coverage < threshold → class_prediction_collapse
     3. zero_variance_fraction high → feature_degeneracy
     4. scaling shrinkage ratio < threshold → scaling_destruction
     5. label_entropy low → label_distribution_issue
     6. macro_f1 low but above random*2.5 → weak_signal
   
   Returns:
   {
     "primary": "root_cause_name",
     "secondary": ["secondary_issue_1", ...],
     "stage": "failure_stage",
     "confidence": 0.0-1.0,
     "offending_stage": "stage_name",
     "kill_list": ["f_0", "f_5", ...]
   }

4. get_action_directive(root_cause)
   - Maps root-cause → deterministic action type
   
   Mapping:
   - scaling_destruction → REMOVE_SCALING
   - feature_degeneracy → DROP_FEATURES
   - class_prediction_collapse → REBUILD_LABELS
   - feature_space_collapse → FIX_ENCODING
   - label_distribution_issue → REBALANCE_CLASSES
   - weak_signal → FEATURE_ENGINEERING
   - unknown → INVESTIGATE
   
   Returns:
   {
     "type": "ACTION_TYPE",
     "target_stage": "stage_name",
     "target_features": ["f_0", "f_1"],
     "rationale": "Root cause explanation"
   }

5. create_summary(meta)
   - Compresses diagnosis into quick decision layer
   - Extracts only essential fields for CI
   
   Returns:
   {
     "status": "PASS" | "FAIL",
     "primary_issue": "root_cause",
     "stage": "failure_stage",
     "action": "ACTION_TYPE",
     "confidence": 0.0-1.0,
     "kill_list": ["f_0", ...]
   }

6. format_failure_message(summary)
   - Formats deterministic error message for training logs
   
   Output:
   UNSW CONTRACT FAILURE
   Primary: feature_space_collapse
   Stage: standard_scaling
   Action: FIX_ENCODING
   Confidence: 0.95


INTEGRATION POINTS
==================

1. Updated build_meta() to include:
   - "root_cause": derive_root_cause(meta)
   - "action": get_action_directive(root_cause)
   - "summary": create_summary(meta)

2. Updated assert_contract() to use format_failure_message(summary)
   - Error messages now include root cause + action
   - Zero ambiguity in failure diagnosis

3. Updated validate_unsw_learnability.py:
   - Added --ci-output flag for condensed summary
   - CI output shows only: status, primary_issue, stage, action, confidence
   - Replaces large JSON with actionable text


USAGE EXAMPLES
==============

Example 1: CI Output (Condensed)
$ python3 scripts/validation/validate_unsw_learnability.py --ci-output

Output:
LEARNABILITY: FAIL
  Primary Issue: feature_space_collapse
  Stage: None
  Action: FIX_ENCODING
  Confidence: 0.95
  Secondary Issues: high_feature_overlap, insufficient_class_separation


Example 2: Full Diagnostics (Default)
$ python3 scripts/validation/validate_unsw_learnability.py

Output: Full meta.json with root_cause, action, summary sections


Example 3: Training Script Error
$ python3 scripts/training/train_helix_ids_full.py

RuntimeError:
UNSW CONTRACT FAILURE
Primary: scaling_destruction
Stage: standard_scaling
Action: REMOVE_SCALING
Confidence: 0.91


Example 4: Programmatic Access
from helix_ids.data.learnability_contract import (
    load_meta,
    format_failure_message,
)

meta = load_meta(artifact_dir=Path("data/processed/multi_dataset_v1"))
summary = meta["summary"]  # Already computed
root_cause = meta["root_cause"]  # Already computed
action = meta["action"]  # Already computed

# Get failure message
if not meta["validated"]:
    msg = format_failure_message(summary)
    print(msg)  # UNSW CONTRACT FAILURE...


DECISION FLOW (DETERMINISTIC)
==============================

Input: Complete meta.json with all diagnostics

1. CHECK centroid_min_distance < 0.01?
   Yes → primary = "feature_space_collapse"
   No → Continue to 2

2. CHECK unique_pred_coverage < 0.80?
   Yes → primary = "class_prediction_collapse"
   No → Continue to 3

3. CHECK zero_variance_fraction > 0.10?
   Yes → primary = "feature_degeneracy"
   No → Continue to 4

4. CHECK any stage with centroid_shrinkage_ratio < 0.30?
   Yes → primary = "scaling_destruction"
   No → Continue to 5

5. CHECK label_entropy < 0.60 * log(num_classes)?
   Yes → primary = "label_distribution_issue"
   No → Continue to 6

6. CHECK macro_f1 < random_f1 * 2.5?
   Yes → primary = "weak_signal"
   No → primary = "unknown"

Output: Single, unambiguous root cause + action


TEST COVERAGE
=============

All new functions have comprehensive unit tests:
✓ test_derive_root_cause_feature_space_collapse
✓ test_derive_root_cause_class_prediction_collapse
✓ test_derive_root_cause_feature_degeneracy
✓ test_rank_failure_stages
✓ test_extract_feature_kill_list
✓ test_get_action_directive_scaling_destruction
✓ test_get_action_directive_feature_degeneracy
✓ test_get_action_directive_class_prediction_collapse
✓ test_create_summary
✓ test_format_failure_message
✓ test_actual_artifact_contract_diagnosis

Run tests:
$ pytest tests/test_data/test_root_cause_reducer.py -v


SYSTEM COMPLETENESS
====================

✓ Detection: Multiple violation checks (BLOCKERs)
✓ Prevention: Hard-coded rules prevent false positives
✓ Localization: Stage ranking + feature kill list
✓ Decision: Deterministic root-cause extraction (NEW)
✓ Action: Auto-generated action directives (NEW)
✓ Diagnosis: Compressed summary for CI (NEW)
✓ Explanation: Format failure messages (NEW)

Before: Observability-heavy, cognitively manual
After: Self-diagnosing, action-driven, zero-ambiguity


EXAMPLE DIAGNOSIS SESSION
==========================

Scenario: UNSW learnability validation fails

1. System runs full contract validation
2. compute_contract_metrics() generates full diagnostics
3. build_meta() creates meta.json with:
   - All raw metrics ✓
   - Violations list ✓
   - root_cause analysis (NEW)
   - action directive (NEW)
   - summary layer (NEW)

4. assert_contract() checks validation
5. If failed, raises RuntimeError with:
   "UNSW CONTRACT FAILURE
   Primary: feature_space_collapse
   Stage: unknown
   Action: FIX_ENCODING
   Confidence: 0.95"

6. Engineer reads error, knows exactly:
   - What failed (feature_space_collapse)
   - Why it failed (centroid_min_distance too low)
   - How to fix it (FIX_ENCODING - adjust encoding scheme)
   - Confidence (0.95 - very likely correct diagnosis)

No expert interpretation required. No guessing. Deterministic action.


FUTURE EXTENSIONS
=================

1. Auto-fix suggestions
   - Given action directive, suggest code changes

2. Failure trajectory tracking
   - Store historical root causes
   - Detect recurring failure patterns

3. Feature attribution refinement
   - Currently: top-N by MI delta
   - Future: SHAP/permutation importance

4. A/B testing framework
   - Compare fix effectiveness
   - Measure action success rate

5. Feedback loop
   - Engineer verifies diagnosis
   - Train classifier on labeled failures
   - Refine confidence scoring


FILE CHANGES
============

Modified:
- src/helix_ids/data/learnability_contract.py (added 6 functions + updated build_meta + updated assert_contract)
- scripts/validation/validate_unsw_learnability.py (added --ci-output flag + print_ci_summary)

Added:
- tests/test_data/test_root_cause_reducer.py (11 comprehensive unit tests)

Unchanged (inherit new diagnostics automatically):
- scripts/training/train_helix_ids_full.py (already uses assert_contract)
- data/processed/multi_dataset_v1/meta.json (updated with root_cause, action, summary)
"""
