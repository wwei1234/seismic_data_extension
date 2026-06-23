# Context-Masked Local Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and evaluate experiment 22 using 256-trace F3 narrow-band context with wide-band supervision limited to a 32-trace well window.

**Architecture:** Reuse experiment 21's four-fold leakage protocol and phase-consistent model. Replace its 32-trace local samples with 256-trace context samples carrying an explicit 32-trace binary supervision mask, then apply masked fold losses while preserving the common pretraining and locked held-out evaluation flow.

**Tech Stack:** Python, NumPy, PyTorch, SciPy, Matplotlib, pytest, SEG-Y reader.

---

### Task 1: Isolated Experiment Structure

**Files:**
- Create: `22_context_masked_calibration/README.md`
- Create: `22_context_masked_calibration/code/`
- Create: `22_context_masked_calibration/tests/`
- Create: `docs/24_22号实验改进思路.txt`

- [ ] Create branch `feature/experiment22-context-mask` in `.worktrees/experiment22`.
- [ ] Copy experiment 21 tracked code and tests as the starting implementation.
- [ ] Rename configuration paths and experiment identifiers to experiment 22.
- [ ] Run the copied tests and record the clean baseline.

### Task 2: Context And Mask Geometry

**Files:**
- Create: `22_context_masked_calibration/code/context_masked_samples.py`
- Modify: `22_context_masked_calibration/code/config.py`
- Test: `22_context_masked_calibration/tests/test_context_masked_samples.py`

- [ ] Write failing tests asserting a centered 32-trace mask inside a 256-trace context.
- [ ] Write a failing test asserting context extraction uses consecutive geometry coordinates.
- [ ] Implement context placement, mask construction, and edge-safe context planning.
- [ ] Run the focused tests until they pass.

### Task 3: Leakage-Safe Sample Generation

**Files:**
- Modify: `22_context_masked_calibration/code/03_generate_fold_samples.py`
- Test: `22_context_masked_calibration/tests/test_context_masked_samples.py`
- Test: `22_context_masked_calibration/tests/test_leakage_guard.py`

- [ ] Write failing tests for zero targets outside the mask and exact closure inside it.
- [ ] Generate full low-pass context from the 256-trace wide extraction.
- [ ] Copy residual values only from the approved 32-trace calibration window.
- [ ] Save `train_masks.npy`, `val_masks.npy`, and coordinate metadata.
- [ ] Reject incomplete or non-consecutive contexts.
- [ ] Generate all four folds and audit mask counts, closure, coordinate overlap, and adjacent-trace continuity.

### Task 4: Masked Dataset And Loss

**Files:**
- Modify: `22_context_masked_calibration/code/datasets.py`
- Modify: `22_context_masked_calibration/code/phase_loss.py`
- Modify: `22_context_masked_calibration/code/04_finetune_fold.py`
- Test: `22_context_masked_calibration/tests/test_datasets.py`
- Test: `22_context_masked_calibration/tests/test_losses.py`

- [ ] Write failing tests proving dataset augmentation flips mask and data together.
- [ ] Write failing tests proving errors outside the mask do not affect supervised loss.
- [ ] Return masks from the local dataset.
- [ ] Apply masked residual, reconstruction, correlation, and phase losses with valid-count normalization.
- [ ] Keep full-context low-frequency leakage regularization.
- [ ] Run all experiment 22 tests.

### Task 5: Training

**Files:**
- Reuse: `22_context_masked_calibration/code/02_pretrain_common.py`
- Modify: `22_context_masked_calibration/code/04_finetune_fold.py`

- [ ] Reuse or verify the experiment 21 common checkpoint by SHA because common pretraining data and model are unchanged.
- [ ] Train all four folds with the experiment 21 epoch and lock policy.
- [ ] Verify every fold lock references the same common checkpoint and forbids held-out wide targets.

### Task 6: Held-Out Inference And Evaluation

**Files:**
- Modify: `22_context_masked_calibration/code/05_predict_heldout.py`
- Modify: `22_context_masked_calibration/code/06_evaluate_folds.py`

- [ ] Predict four held-out inline and four held-out crossline sections.
- [ ] Save direct and high-pass recombinations.
- [ ] Re-run the experiment 21 aggregate metrics and success gate.
- [ ] Add experiment 21 versus experiment 22 metric comparison.
- [ ] Add sample continuity diagnostics and figures.
- [ ] Write the final Chinese evaluation report.

### Task 7: Verification And Integration

**Files:**
- Update: `22_context_masked_calibration/README.md`
- Update: `docs/24_22号实验改进思路.txt`

- [ ] Run the full pytest suite.
- [ ] Compile every experiment 22 Python file.
- [ ] Audit eight prediction sections, four fold locks, masks, report, and common checkpoint SHA.
- [ ] Commit only tracked experiment code, tests, README, and documentation.
- [ ] Merge locally only after verification and copy ignored experiment artifacts back to the main workspace.
