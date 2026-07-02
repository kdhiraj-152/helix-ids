# TON-IoT Dataset Inventory

Generated: 2026-06-21 20:21:05

## Files

| File | Size (MB) | Rows | Columns |
|------|-----------|------|---------|
| test.csv | 28.39 | 211,043 | 44 |
| train.csv | 28.39 | 211,043 | 44 |

**Total rows across all files:** 422,086
**Total size:** 56.78 MB
**Total columns:** 44

## Columns

0: `src_ip`
1: `src_port`
2: `dst_ip`
3: `dst_port`
4: `proto`
5: `service`
6: `duration`
7: `src_bytes`
8: `dst_bytes`
9: `conn_state`
10: `missed_bytes`
11: `src_pkts`
12: `src_ip_bytes`
13: `dst_pkts`
14: `dst_ip_bytes`
15: `dns_query`
16: `dns_qclass`
17: `dns_qtype`
18: `dns_rcode`
19: `dns_AA`
20: `dns_RD`
21: `dns_RA`
22: `dns_rejected`
23: `ssl_version`
24: `ssl_cipher`
25: `ssl_resumed`
26: `ssl_established`
27: `ssl_subject`
28: `ssl_issuer`
29: `http_trans_depth`
30: `http_method`
31: `http_uri`
32: `http_version`
33: `http_request_body_len`
34: `http_response_body_len`
35: `http_status_code`
36: `http_user_agent`
37: `http_orig_mime_types`
38: `http_resp_mime_types`
39: `weird_name`
40: `weird_addl`
41: `weird_notice`
42: `label`
43: `type`

## Attack Categories

**Count: 9**

- `backdoor`: 40,000 rows
- `ddos`: 40,000 rows
- `dos`: 40,000 rows
- `injection`: 40,000 rows
- `mitm`: 2,086 rows
- `password`: 40,000 rows
- `ransomware`: 40,000 rows
- `scanning`: 40,000 rows
- `xss`: 40,000 rows

## Normal Categories

**Count: 1**

- `normal`: 100,000 rows