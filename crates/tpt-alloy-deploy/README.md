# tpt-alloy-deploy

The Alloy (MCU swarm) deploy layer of TPT Cobalt, written natively in Rust.
The forked `tpt-alloy` crate partitions models and generates per-node
firmware sources; upstream had *"no deployment mechanism"* — this crate is
the missing deploy half. Protocol knowledge only; no port of
tpt-basestation's Python (see `docs/phase5-exotic-backends.md`).

## Features

- **RP2040 UF2 generation** (`uf2`) — raw flash images to UF2 blocks
  (USB mass-storage drop format): spec byte layout
  (magics `0x0A324655`/`0x9E5D5157`/`0x0AB16F30`, family `0xE48BFF56`,
  flags `0x2000`, 256-byte chunks zero-padded, seq/total counters),
  known-vector tested.
- **ESP32 ROM-UART protocol** (`esp`) — SLIP encode/decode with escape
  handling, 9-byte-header command packets (Sync / FlashBegin / FlashData /
  FlashEnd, XOR-0xEF data checksums), and `EspFlasher`, a flash driver
  over the `Transport` trait: a real serial backend is one impl away, and
  tests drive an in-memory mock device (success, error-status, and
  multi-packet sequences).
- **Fleet OTA** (`ota`) — sha256-digested `NodeArtifact`s keyed by node id
  + `tpt_alloy::FirmwareTarget`; manifest validation rejects duplicate
  node ids, foreign release ids, and digest tampering; two-phase staged
  rollout (stage every node in id order, then a commit gate) with
  abort-on-failure and skip modes; JSON export strips image bytes but
  keeps digests.

## Usage

```rust,ignore
use tpt_alloy_deploy::{to_uf2, UF2_FAMILY_RP2040, Rollout};

let uf2 = to_uf2(&image, 0x1000_0000, UF2_FAMILY_RP2040);

let mut rollout = Rollout::begin(manifest)?;
while let Some(node) = rollout.next_node() {
    /* stage via your transport */
    rollout.mark_staged();
}
rollout.commit()?;
```

## Testing

```sh
cargo test -p tpt-alloy-deploy
```

17 tests cover the UF2 block layout against the spec, SLIP round-trips and
escape edge cases, protocol packet layout field-by-field, the flash
sequence over a mock transport (chunking, sequence numbers, header
checksums, reboot flag), device-error propagation, and the rollout state
machine.

## Status

0.1.x, first slice. Remaining (optional): a real `serialport` transport
implementation and signed release bundles.
