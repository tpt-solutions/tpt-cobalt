extern crate alloc;

use crate::types::ScalarType;

/// Description of a quantized tensor layout.
///
/// Quantized formats group `block_size` elements into a fixed-width block of
/// `bytes_per_block` bytes. Some layouts store per-block scale values; the
/// optional `scales` field records how many scale values follow each block
/// (e.g. `1` for `Q4_0`/`Q8_0`, `2` for `Q4_1`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct QuantizationParams {
    pub block_size: u32,
    pub bytes_per_block: u32,
    pub num_blocks: u32,
    pub scales: Option<u32>,
}

impl QuantizationParams {
    /// Total serialized byte size of all `num_blocks` blocks.
    ///
    /// Uses checked arithmetic and returns `None` on overflow.
    pub fn byte_size(&self) -> Option<usize> {
        (self.num_blocks as usize).checked_mul(self.bytes_per_block as usize)
    }
}

/// GGUF-correct block parameters for the packed 4-bit / 8-bit formats.
pub const Q4_0_BLOCK: (u32, u32) = (32, 18);
pub const Q4_1_BLOCK: (u32, u32) = (32, 20);
pub const Q8_0_BLOCK: (u32, u32) = (32, 34);

/// Compute the byte size of a tensor of `num_elements` elements of type `dtype`.
///
/// Non-quantized scalars use [`ScalarType::size_bytes`]. Quantized formats use
/// the GGUF-correct block sizes (`Q4_0`/`Q4_1`/`Q8_0`) and integer arithmetic.
/// Returns `None` on overflow.
pub fn tensor_byte_size(dtype: ScalarType, num_elements: usize) -> Option<usize> {
    match dtype {
        ScalarType::Q4_0 => {
            quantized_blocks(num_elements, Q4_0_BLOCK.0 as usize, Q4_0_BLOCK.1 as usize)
        }
        ScalarType::Q4_1 => {
            quantized_blocks(num_elements, Q4_1_BLOCK.0 as usize, Q4_1_BLOCK.1 as usize)
        }
        ScalarType::Q8_0 => {
            quantized_blocks(num_elements, Q8_0_BLOCK.0 as usize, Q8_0_BLOCK.1 as usize)
        }
        other => num_elements.checked_mul(other.size_bytes()),
    }
}

/// Byte size for a quantized layout: `ceil(num_elements / block_size) * bytes_per_block`.
fn quantized_blocks(
    num_elements: usize,
    block_size: usize,
    bytes_per_block: usize,
) -> Option<usize> {
    let blocks = num_elements.div_ceil(block_size);
    blocks.checked_mul(bytes_per_block)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_sizes() {
        assert_eq!(tensor_byte_size(ScalarType::F32, 100).unwrap(), 400);
        assert_eq!(tensor_byte_size(ScalarType::I8, 100).unwrap(), 100);
    }

    #[test]
    fn q4_0_sizes() {
        // 32 elements -> 18 bytes (one block)
        assert_eq!(tensor_byte_size(ScalarType::Q4_0, 32).unwrap(), 18);
        // 64 elements -> 36 bytes (two blocks)
        assert_eq!(tensor_byte_size(ScalarType::Q4_0, 64).unwrap(), 36);
        // 1 element -> still one block (rounds up)
        assert_eq!(tensor_byte_size(ScalarType::Q4_0, 1).unwrap(), 18);
    }

    #[test]
    fn q4_1_q8_0_sizes() {
        assert_eq!(tensor_byte_size(ScalarType::Q4_1, 32).unwrap(), 20);
        assert_eq!(tensor_byte_size(ScalarType::Q8_0, 32).unwrap(), 34);
    }

    #[test]
    fn overflow_is_none() {
        let huge = tensor_byte_size(ScalarType::F32, usize::MAX);
        assert!(huge.is_none());
    }

    #[test]
    fn quantized_params_byte_size() {
        let q = QuantizationParams {
            block_size: 32,
            bytes_per_block: 18,
            num_blocks: 4,
            scales: Some(1),
        };
        assert_eq!(q.byte_size().unwrap(), 72);
    }
}
