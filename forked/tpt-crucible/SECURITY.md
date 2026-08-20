# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |

## Reporting a Vulnerability

Please **do not** open a public GitHub issue for security vulnerabilities.

Email **security@tpt.solutions** with:
- A description of the vulnerability and affected component
- Steps to reproduce or a proof-of-concept (no live exploit code required)
- Your assessment of impact and severity

You will receive an acknowledgement within **48 hours** and a status update within **7 days**.

## Disclosure Timeline

TPT Solutions follows coordinated disclosure:

1. Vulnerability reported privately
2. TPT Solutions confirms and assesses within 7 days
3. Fix developed and tested (target: within 30 days for critical, 90 days for lower severity)
4. Fix released and advisory published simultaneously
5. Reporter credited in advisory (unless anonymity requested)

Maximum embargo: **90 days** from initial report.

## Scope

The following are in scope:
- Rust crates: `tpt-catalyst`, `tpt-alloy`, `tpt-catalyst-wasm`
- Python packages: `tpt_catalyst`, `tpt_alloy`, `tpt_fusion`, `tpt_element`, `tpt_mosaic`, `tpt_drivers`, `tpt_emulator`
- Go services: `tpt-observer`, synthesis workers, crucible-cloud
- Frontend: `frontend/`

Out of scope:
- Upstream tools (MLIR, Yosys, Nextpnr, Xyce, PlatformIO) — report these to their respective projects
- Issues in user-managed hardware driver packages
- Denial-of-service via resource exhaustion on deliberately oversized inputs

## Known Mitigations

- **WebSocket**: CORS origin allowlist + per-IP connection limit (max 5)
- **Package extraction**: `.tptpkg` ZIP path-traversal guard (rejects `..` and absolute paths)
- **WASM**: IR input size capped at 50 MB; structured error JSON on invalid input
- **File uploads**: type allowlist + 10 GB size cap
- **localStorage**: all reads validated with Zod schemas; corrupt entries cleared automatically
