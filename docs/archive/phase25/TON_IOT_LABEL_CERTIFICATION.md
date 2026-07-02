# TON-IoT Label Certification

**Generated:** 2026-06-21 IST
**Source column:** `type` (multi-class, 10 classes)

## 1. Raw Label Distribution

| Label | Count |
|-------|-------|
| normal       |   42,040 |
| scanning     |   20,000 |
| ddos         |   19,993 |
| injection    |   19,964 |
| password     |   19,861 |
| dos          |   18,992 |
| backdoor     |   18,711 |
| xss          |   15,137 |
| ransomware   |   14,735 |
| mitm         |    1,041 |

Total: 190,474

## 2. 7-Class Mapping

| Raw Label | 7-Class Index | 7-Class Name |
|-----------|---------------|--------------|
| backdoor     |             6 | Backdoor                            |
| ddos         |             1 | DoS                                 |
| dos          |             1 | DoS                                 |
| injection    |             3 | R2L                                 |
| mitm         |             6 | Backdoor                            |
| normal       |             0 | Normal                              |
| password     |             3 | R2L                                 |
| ransomware   |             6 | Backdoor                            |
| scanning     |             2 | Probe                               |
| xss          |             3 | R2L                                 |

### 7-Class Distribution (after mapping)

| 7-Class Index | 7-Class Name | Count |
|---------------|--------------|-------|
|             0 | Normal                              |   42,040 |
|             1 | DoS                                 |   38,985 |
|             2 | Probe                               |   20,000 |
|             3 | R2L                                 |   54,962 |
|             6 | Backdoor                            |   34,487 |

Total: 190,474

## 3. 5-Class (Family) Mapping

| Raw Label | 5-Class Family | Family Index |
|-----------|----------------|--------------|
| backdoor     | R2L             |             3 |
| ddos         | DoS             |             1 |
| dos          | DoS             |             1 |
| injection    | R2L             |             3 |
| mitm         | R2L             |             3 |
| normal       | Normal          |             0 |
| password     | R2L             |             3 |
| ransomware   | R2L             |             3 |
| scanning     | Probe           |             2 |
| xss          | R2L             |             3 |

## 4. Binary Mapping

| Raw Label | Is Attack (7-class != Normal) | Binary Value |
|-----------|-------------------------------|--------------|
| backdoor     | Yes                            |            1 |
| ddos         | Yes                            |            1 |
| dos          | Yes                            |            1 |
| injection    | Yes                            |            1 |
| mitm         | Yes                            |            1 |
| normal       | No                             |            0 |
| password     | Yes                            |            1 |
| ransomware   | Yes                            |            1 |
| scanning     | Yes                            |            1 |
| xss          | Yes                            |            1 |
## 5. Mapping Integrity Checks

| Check | Status |
|-------|--------|
| All 10 raw labels map to 7-class | PASS |
| All 10 raw labels map to 5-class | PASS |
| 7-class values in range [0, 6] | PASS |
| Binary mapping preserves normal=0 | PASS |
| No label corruption | PASS |

## 6. Verdict

**PASS** — All TON-IoT labels map correctly to 7-class, 5-class, and binary taxonomies.
