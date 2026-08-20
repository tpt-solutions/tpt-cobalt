use crate::partition::{CrossEdge, Partition};
use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum FirmwareTarget {
    Esp32,
    Rp2040,
    RiscV,
}

impl fmt::Display for FirmwareTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FirmwareTarget::Esp32 => write!(f, "esp32"),
            FirmwareTarget::Rp2040 => write!(f, "rp2040"),
            FirmwareTarget::RiscV => write!(f, "riscv"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FirmwareBundle {
    pub node_id: usize,
    pub target: FirmwareTarget,
    pub source_code: String,
    pub config_json: String,
}

pub fn generate_firmware(partition: &Partition, target: &FirmwareTarget) -> FirmwareBundle {
    let source = match target {
        FirmwareTarget::Esp32 => generate_esp32_firmware(partition),
        FirmwareTarget::Rp2040 => generate_rp2040_firmware(partition),
        FirmwareTarget::RiscV => generate_riscv_firmware(partition),
    };

    let config = serde_json::json!({
        "node_id": partition.node_id,
        "layers": partition.assigned_layers,
        "assigned_heads": partition.assigned_heads,
        "is_aggregator": partition.is_aggregator,
        "cross_edges": partition.cross_node_edges.len(),
        "sum_reduce_edges": partition.cross_node_edges.iter().filter(|e| e.protocol == "sum_reduce").count(),
    });

    FirmwareBundle {
        node_id: partition.node_id,
        target: target.clone(),
        source_code: source,
        config_json: serde_json::to_string_pretty(&config).unwrap(),
    }
}

fn generate_sum_reduce_handshake(partition: &Partition) -> String {
    let sum_reduce_edges: Vec<&CrossEdge> = partition
        .cross_node_edges
        .iter()
        .filter(|e| e.protocol == "sum_reduce")
        .collect();

    if sum_reduce_edges.is_empty() {
        return String::new();
    }

    if partition.is_aggregator {
        // Aggregator node: receive partial head results and sum-reduce
        String::from(
            "// === Sum-Reduce Aggregator ===\n\
             // Receives partial attention head results from peer nodes\n\
             void sum_reduce_aggregate() {\n\
             #ifdef TPT_USE_WIFI\n\
                 for (int i = 0; i < TPT_NUM_PEERS; i++) {\n\
                     float partial[TPT_HEAD_DIM];\n\
                     if (tpt_recv(i, partial, sizeof(partial)) == 0) {\n\
                         // Accumulate partial head results\n\
                         for (int j = 0; j < TPT_HEAD_DIM; j++) {\n\
                             tpt_accum_buffer[j] += partial[j];\n\
                         }\n\
                     }\n\
                 }\n\
             #endif\n\
             }\n\n",
        )
    } else {
        // Worker node: send partial head results to aggregator
        String::from(
            "// === Sum-Reduce Worker ===\n\
             // Sends partial attention head results to aggregator node\n\
             void sum_reduce_send() {\n\
             #ifdef TPT_USE_WIFI\n\
                 float partial[TPT_HEAD_DIM];\n\
                 // Fill partial with this node's head computation results\n\
                 tpt_send(0, partial, sizeof(partial));\n\
             #endif\n\
             }\n\n",
        )
    }
}

fn generate_esp32_firmware(partition: &Partition) -> String {
    let sum_reduce_code = generate_sum_reduce_handshake(partition);
    let head_info = if !partition.assigned_heads.is_empty() {
        format!(
            "// Assigned heads: {:?}\n// Is aggregator: {}",
            partition.assigned_heads, partition.is_aggregator
        )
    } else {
        String::new()
    };

    let layer_count = partition.assigned_layers.len();
    let layer_array: String = partition
        .assigned_layers
        .iter()
        .map(|l| l.to_string())
        .collect::<Vec<_>>()
        .join(", ");

    format!(
        r#"// Auto-generated ESP32 firmware for node {}
// Layers: {:?}
{}
#include <Arduino.h>

static const int tpt_assigned_layers[] = {{{}}};
static const int TPT_LAYER_COUNT = {};

void setup() {{
    Serial.begin(115200);
    Serial.println("TPT Alloy node {} starting");
}}

{}
void loop() {{
    for (int i = 0; i < TPT_LAYER_COUNT; i++) {{
        tpt_run_layer(tpt_assigned_layers[i]);
    }}
    tpt_sync_neighbors();
}}
"#,
        partition.node_id,
        partition.assigned_layers,
        head_info,
        layer_array,
        layer_count,
        partition.node_id,
        sum_reduce_code
    )
}

fn generate_rp2040_firmware(partition: &Partition) -> String {
    let sum_reduce_code = generate_sum_reduce_handshake(partition);
    let head_info = if !partition.assigned_heads.is_empty() {
        format!(
            "// Assigned heads: {:?}\n// Is aggregator: {}",
            partition.assigned_heads, partition.is_aggregator
        )
    } else {
        String::new()
    };

    let layer_count = partition.assigned_layers.len();
    let layer_array: String = partition
        .assigned_layers
        .iter()
        .map(|l| l.to_string())
        .collect::<Vec<_>>()
        .join(", ");

    format!(
        r#"// Auto-generated RP2040 firmware for node {}
// Layers: {:?}
{}
#include "pico/stdlib.h"
#include "hardware/flash.h"

static const int tpt_assigned_layers[] = {{{}}};
static const int TPT_LAYER_COUNT = {};

/* TPT hardware-lock verify — reads unique_id from RP2040 flash */
static void tpt_verify_hw_lock(void) {{
    uint8_t uid[8];
    flash_get_unique_id(uid);
    uint64_t id64 = 0;
    for (int i = 0; i < 8; i++) id64 = (id64 << 8) | uid[i];
    (void)id64; /* compare against embedded fingerprint if --lock-to-hardware was used */
}}

int main() {{
    stdio_init_all();
    tpt_verify_hw_lock();
    printf("TPT Alloy node {} starting\n");
    {}
    while (true) {{
        for (int i = 0; i < TPT_LAYER_COUNT; i++) {{
            tpt_run_layer(tpt_assigned_layers[i]);
        }}
        tpt_sync_neighbors();
    }}
    return 0;
}}
"#,
        partition.node_id,
        partition.assigned_layers,
        head_info,
        layer_array,
        layer_count,
        partition.node_id,
        sum_reduce_code
    )
}

fn generate_riscv_firmware(partition: &Partition) -> String {
    let sum_reduce_code = generate_sum_reduce_handshake(partition);
    let head_info = if !partition.assigned_heads.is_empty() {
        format!(
            "// Assigned heads: {:?}\n// Is aggregator: {}",
            partition.assigned_heads, partition.is_aggregator
        )
    } else {
        String::new()
    };

    let layer_count = partition.assigned_layers.len();
    let layer_array: String = partition
        .assigned_layers
        .iter()
        .map(|l| l.to_string())
        .collect::<Vec<_>>()
        .join(", ");

    format!(
        r#"// Auto-generated RISC-V firmware for node {}
// Layers: {:?}
{}
#include <stdio.h>
#include <stdint.h>

static const int tpt_assigned_layers[] = {{{}}};
static const int TPT_LAYER_COUNT = {};

/* TPT hardware-lock verify — reads OTP fuses via SiFive OTP MMIO */
static void tpt_verify_hw_lock(void) {{
#ifdef SIFIVE_OTP_BASE
    volatile uint32_t *otp = (volatile uint32_t *)SIFIVE_OTP_BASE;
    uint64_t id64 = ((uint64_t)otp[1] << 32) | otp[0];
    (void)id64; /* compare against embedded fingerprint if --lock-to-hardware was used */
#endif
}}

int main() {{
    tpt_verify_hw_lock();
    printf("TPT Alloy node {} starting\n");
    {}
    while (1) {{
        for (int i = 0; i < TPT_LAYER_COUNT; i++) {{
            tpt_run_layer(tpt_assigned_layers[i]);
        }}
        tpt_sync_neighbors();
    }}
    return 0;
}}
"#,
        partition.node_id,
        partition.assigned_layers,
        head_info,
        layer_array,
        layer_count,
        partition.node_id,
        sum_reduce_code
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::partition::Partition;

    #[test]
    fn test_esp32_firmware_gen() {
        let partition = Partition {
            node_id: 0,
            assigned_layers: vec![0, 1],
            cross_node_edges: vec![],
            assigned_heads: vec![],
            is_aggregator: false,
        };
        let bundle = generate_firmware(&partition, &FirmwareTarget::Esp32);
        assert_eq!(bundle.node_id, 0);
        assert!(bundle.source_code.contains("ESP32"));
    }

    #[test]
    fn test_firmware_with_sum_reduce() {
        let partition = Partition {
            node_id: 1,
            assigned_layers: vec![0, 2],
            cross_node_edges: vec![
                crate::partition::CrossEdge {
                    from_node: 1,
                    to_node: 0,
                    tensor_name: "sum_reduce_attn_0".into(),
                    protocol: "sum_reduce".into(),
                },
            ],
            assigned_heads: vec![2, 3],
            is_aggregator: false,
        };
        let bundle = generate_firmware(&partition, &FirmwareTarget::Esp32);
        assert!(bundle.source_code.contains("sum_reduce_send"));
        assert!(bundle.source_code.contains("Assigned heads"));
        assert!(bundle.config_json.contains("sum_reduce_edges"));
    }

    #[test]
    fn test_firmware_aggregator() {
        let partition = Partition {
            node_id: 0,
            assigned_layers: vec![0, 2],
            cross_node_edges: vec![CrossEdge {
                from_node: 1,
                to_node: 0,
                tensor_name: "head".into(),
                protocol: "sum_reduce".into(),
            }],
            assigned_heads: vec![0, 1],
            is_aggregator: true,
        };
        let bundle = generate_firmware(&partition, &FirmwareTarget::Esp32);
        assert!(bundle.source_code.contains("sum_reduce_aggregate"));
        assert!(bundle.config_json.contains(r#""is_aggregator": true"#));
    }
}
