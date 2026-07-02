# TON-IoT Feature Harmonization Report

Generated: 2026-06-21 20:21:07

## Summary

- **Total canonical features:** 17
- **Automatic mappings:** 17 / 17
- **Manual mappings required:** 0 / 17
- **Unresolved fields:** 0 / 17
- **Can fully convert:** YES

## Automatic Mappings

| Canonical Feature | TON-IoT Source |
|-----------------|---------------|
| `connection_state` | `derived from raw features` |
| `count_x_srv_count` | `geometry expansion (derived)` |
| `diff_srv_rate_x_flag` | `geometry expansion (derived)` |
| `dst_bytes` | `dst_bytes` |
| `dst_src_bytes_ratio` | `geometry expansion (derived)` |
| `duration` | `duration` |
| `flag` | `conn_state (encode via FLAG_MAP)` |
| `has_rst` | `derived from raw features` |
| `log_dst_bytes` | `geometry expansion (derived)` |
| `log_src_bytes` | `geometry expansion (derived)` |
| `protocol_service_flag` | `geometry expansion (derived)` |
| `protocol_type` | `proto (rename)` |
| `same_host_rate_x_service` | `geometry expansion (derived)` |
| `service_tier` | `derived from raw features` |
| `src_bytes` | `src_bytes` |
| `src_dst_bytes_ratio` | `geometry expansion (derived)` |
| `traffic_direction` | `derived from raw features` |

## Extra TON-IoT Features (not in canonical schema)

**28 extra features**

- `dns_query`
- `dns_qclass`
- `dns_qtype`
- `dns_rcode`
- `dns_rejected`
- `ssl_version`
- `ssl_cipher`
- `ssl_resumed`
- `ssl_established`
- `ssl_subject`
- `ssl_issuer`
- `http_trans_depth`
- `http_method`
- `http_uri`
- `http_version`
- `http_request_body_len`
- `http_response_body_len`
- `http_status_code`
- `http_user_agent`
- `http_orig_mime_types`
- `http_resp_mime_types`
- `weird_name`
- `weird_addl`
- `weird_notice`
- `src_ip`
- `src_port`
- `dst_ip`
- `dst_port`