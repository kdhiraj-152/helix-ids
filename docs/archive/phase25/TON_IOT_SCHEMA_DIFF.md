# TON-IoT Schema Diff vs Reference Datasets

Generated: 2026-06-21 20:21:05

TON-IoT has **44** columns.
NSL-KDD reference: 42 columns.
UNSW-NB15 reference: 49 columns.
CICIDS-2018 reference: 8 columns.
Helix canonical: 17 features.

## Comparison Summary

| Reference | Exact Matches | Partial Matches | Missing | Extra (in TON-IoT) |
|-----------|--------------|----------------|---------|--------------------|
| NSL-KDD | 5 | 3 | 37 | 36 |
| UNSW-NB15 | 3 | 3 | 46 | 38 |
| CICIDS-2018 | 1 | 3 | 7 | 40 |
| Helix Canonical | 3 | 3 | 14 | 38 |

## Detailed Canonical Feature Mapping

| Canonical Feature | Match Type | TON-IoT Source |
|-----------------|-----------|----------------|
| `protocol_type` | exact (via proto) | proto |
| `connection_state` | derivable | conn_state |
| `traffic_direction` | derivable | src_bytes, dst_bytes |
| `has_rst` | derivable | conn_state |
| `log_src_bytes` | derivable | src_bytes, dst_bytes |
| `log_dst_bytes` | derivable | dst_bytes |
| `src_dst_bytes_ratio` | derivable | src_bytes, dst_bytes |
| `dst_src_bytes_ratio` | derivable | src_bytes, dst_bytes |
| `same_host_rate_x_service` | derivable | count, srv_count, protocol_type, service, flag |
| `diff_srv_rate_x_flag` | derivable | count, srv_count, protocol_type, service, flag |
| `count_x_srv_count` | derivable | count, srv_count, protocol_type, service, flag |
| `protocol_service_flag` | derivable | count, srv_count, protocol_type, service, flag |
| `src_bytes` | exact | src_bytes |
| `dst_bytes` | exact | dst_bytes |
| `service_tier` | derivable | service |
| `duration` | exact | duration |
| `flag` | derivable | conn_state |

## Summary

- **Automatic (exact match):** 4 / 17
- **Derivable (requires transformation):** 13 / 17
- **Missing:** 0 / 17