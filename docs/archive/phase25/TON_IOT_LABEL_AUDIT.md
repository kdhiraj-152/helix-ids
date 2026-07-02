# TON-IoT Label Audit

Generated: 2026-06-21 20:21:05

## Binary Labels

Values: [0, 1]
- `0` = Normal
- `1` = Attack

## Raw Type Labels

**Total unique types: 10**

### Type Distribution (by count)

| Type | Count | % of Dataset | Family Mapping | Status |
|------|-------|-------------|---------------|--------|
| normal | 100,000 | 23.69% | Normal | known |
| backdoor | 40,000 | 9.48% | R2L | known |
| ddos | 40,000 | 9.48% | DoS | known |
| dos | 40,000 | 9.48% | DoS | known |
| injection | 40,000 | 9.48% | R2L | needs expansion |
| password | 40,000 | 9.48% | R2L | needs expansion |
| ransomware | 40,000 | 9.48% | DoS | needs expansion |
| scanning | 40,000 | 9.48% | Probe | needs expansion |
| xss | 40,000 | 9.48% | R2L | needs expansion |
| mitm | 2,086 | 0.49% | R2L | impossible |

## Label Mapping Summary

- **Known to Helix (no changes needed):** 4
- **Requires label expansion (add mapping):** 5
- **Impossible to map:** 1

### Labels Already Known to Helix

- `backdoor`
- `ddos`
- `dos`
- `normal`

### Labels Requiring Expansion

- `injection`
- `password`
- `ransomware`
- `scanning`
- `xss`

### Labels Impossible to Map

- `mitm`

## 5-Class Family Distribution (proposed)

| Family | Count | % of Dataset |
|--------|-------|-------------|
| R2L | 162,086 | 38.40% |
| DoS | 120,000 | 28.43% |
| Normal | 100,000 | 23.69% |
| Probe | 40,000 | 9.48% |