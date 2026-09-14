# tpt-alloy-deploy

The Alloy (MCU swarm) deploy layer of TPT Cobalt, written natively in Rust.
The forked `tpt-alloy` crate partitions models and generates per-node
firmware sources; upstream had no deployment mechanism — this crate is the
missing deploy half (protocol knowledge only; no port of tpt-basestation's
Python).

- **RP2040**: raw flash images → UF2 blocks (USB mass-storage drop format),
  family ID `0xE48BFF56`, spec-verified block layout.
- **ESP32**: ROM-UART SLIP framing, command packets (Sync / FlashBegin /
  FlashData / FlashEnd) with XOR-0xEF checksums, and a flash driver over the
  [`Transport`] trait — the real serial backend is one impl away; tests run
  against an in-memory device.
- **Fleet OTA**: sha256-digested `NodeArtifact`s keyed by node id +
  `FirmwareTarget`, two-phase staging with abort-on-failure (or skip mode),
  and a commit gate.

```rust,ignore
let uf2 = tpt_alloy_deploy::to_uf2(&image, 0x1000_0000, UF2_FAMILY_RP2040);
let mut rollout = Rollout::begin(manifest)?;
while let Some(node) = rollout.next_node() { /* stage via transport */ }
rollout.commit()?;
```
