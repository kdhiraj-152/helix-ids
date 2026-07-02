# TON-IoT Deduplication Report

**Generated:** 2026-06-21 IST

## Summary

| Metric | Value |
|--------|-------|
| Rows before cleaning | 211,043 |
| Rows after cleaning (label rename) | 211,043 |
| Rows after dedup | 190,474 |
| Rows removed | 20,569 |
| Removal percentage | 9.75% |

## Per-Class Dedup Breakdown

| Label | Before | After | Removed | % Removed |
|-------|--------|-------|---------|-----------|
| backdoor     |   20,000 |   18,711 |    1,289 |      6.44% |
| ddos         |   20,000 |   19,993 |        7 |      0.03% |
| dos          |   20,000 |   18,992 |    1,008 |      5.04% |
| injection    |   20,000 |   19,964 |       36 |      0.18% |
| mitm         |    1,043 |    1,041 |        2 |      0.19% |
| normal       |   50,000 |   42,040 |    7,960 |     15.92% |
| password     |   20,000 |   19,861 |      139 |      0.69% |
| ransomware   |   20,000 |   14,735 |    5,265 |     26.32% |
| scanning     |   20,000 |   20,000 |        0 |       0.0% |
| xss          |   20,000 |   15,137 |    4,863 |     24.32% |

## Integrity Checks

| Check | Status |
|-------|--------|
| No label corruption | PASS |
| No class disappeared | PASS |
| Removal rate within acceptable bound (< 50%) | PASS |

## Verdict

**PASS** — Deduplication removes only exact duplicate rows without label corruption or class disappearance.
