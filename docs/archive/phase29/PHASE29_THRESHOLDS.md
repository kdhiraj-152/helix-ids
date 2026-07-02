# Phase 29 — Threshold Optimization

**Generated**: 2026-06-23 23:31:45 IST
**Seeds**: [42, 1337, 2026]
**Method**: Per-class F1-optimal threshold search

### Seed 42

| Class | Optimal Threshold | F1 at Threshold | Youden's J | Default F1 |
|------|-----------------:|---------------:|----------:|----------:|
| Normal | 0.677 | 0.9435 | 0.8647 | — |
| DoS | 0.424 | 0.8115 | 0.8032 | — |
| Probe | 0.505 | 0.7288 | 0.7230 | — |
| R2L | 0.333 | 0.6840 | 0.8134 | — |
| U2R | 0.051 | 0.1044 | 0.5216 | — |
| Generic | 0.626 | 0.8872 | 0.8098 | — |
| Backdoor | 0.020 | 0.0850 | 0.7641 | — |

**Default Macro F1 (argmax)**: 0.5760

### Seed 1337

| Class | Optimal Threshold | F1 at Threshold | Youden's J | Default F1 |
|------|-----------------:|---------------:|----------:|----------:|
| Normal | 0.414 | 0.9396 | 0.8431 | — |
| DoS | 0.495 | 0.8052 | 0.8015 | — |
| Probe | 0.404 | 0.7215 | 0.7267 | — |
| R2L | 0.364 | 0.6695 | 0.8431 | — |
| U2R | 0.051 | 0.1094 | 0.1002 | — |
| Generic | 0.182 | 0.8805 | 0.7998 | — |
| Backdoor | 0.030 | 0.0674 | 0.7578 | — |

**Default Macro F1 (argmax)**: 0.5730

### Seed 2026

| Class | Optimal Threshold | F1 at Threshold | Youden's J | Default F1 |
|------|-----------------:|---------------:|----------:|----------:|
| Normal | 0.343 | 0.9420 | 0.7975 | — |
| DoS | 0.455 | 0.8025 | 0.8005 | — |
| Probe | 0.303 | 0.7105 | 0.7196 | — |
| R2L | 0.414 | 0.6737 | 0.8289 | — |
| U2R | 0.061 | 0.1163 | 0.0722 | — |
| Generic | 0.960 | 0.8935 | 0.8117 | — |
| Backdoor | 0.030 | 0.0691 | 0.7541 | — |

**Default Macro F1 (argmax)**: 0.5778

