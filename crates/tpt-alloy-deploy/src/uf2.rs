//! RP2040 UF2 generation: raw flash-image bytes → USB mass-storage blocks.
//!
//! Block layout per the UF2 spec (all fields little-endian):
//! 512-byte blocks; 0x0A324655 / 0x9E5D5157 start magics, flags
//! `0x00002000` (family ID present), target address, 256-byte data chunks
//! (zero-padded in the final block), sequence/total counters, the RP2040
//! family ID `0xE48BFF56`, and the `0x0AB16F30` end magic.

/// UF2 family ID for RP2040.
pub const UF2_FAMILY_RP2040: u32 = 0xE48_BF56;
const MAGIC_START0: u32 = 0x0A32_4655;
const MAGIC_START1: u32 = 0x9E5D_5157;
const MAGIC_END: u32 = 0x0AB1_6F30;
const FLAG_FAMILY_ID: u32 = 0x0000_2000;
const DATA_CHUNK: usize = 256;
const BLOCK_SIZE: usize = 512;

/// Convert a raw flash image (loaded at `base_addr`) into UF2 blocks.
/// Returns one 512-byte block per 256 payload bytes (`total = ceil(len/256)`).
pub fn to_uf2(image: &[u8], base_addr: u32, family_id: u32) -> Vec<u8> {
    let total = image.len().div_ceil(DATA_CHUNK);
    let mut out = Vec::with_capacity(total * BLOCK_SIZE);
    for (seq, chunk) in image.chunks(DATA_CHUNK).enumerate() {
        let mut block = vec![0u8; BLOCK_SIZE];
        put_u32(&mut block[0x00..], MAGIC_START0);
        put_u32(&mut block[0x04..], MAGIC_START1);
        put_u32(&mut block[0x08..], FLAG_FAMILY_ID);
        put_u32(&mut block[0x0C..], base_addr + (seq * DATA_CHUNK) as u32);
        put_u32(&mut block[0x10..], chunk.len() as u32);
        put_u32(&mut block[0x14..], seq as u32);
        put_u32(&mut block[0x18..], total as u32);
        put_u32(&mut block[0x1C..], family_id);
        block[0x20..0x20 + chunk.len()].copy_from_slice(chunk);
        // bytes up to 0x1FC stay zero (padding)
        put_u32(&mut block[0x1FC..], MAGIC_END);
        out.extend_from_slice(&block);
    }
    out
}

fn put_u32(buf: &mut [u8], v: u32) {
    buf[..4].copy_from_slice(&v.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn u32_at(block: &[u8], off: usize) -> u32 {
        u32::from_le_bytes(block[off..off + 4].try_into().unwrap())
    }

    #[test]
    fn uf2_block_layout_matches_the_spec() {
        // 300 bytes → 2 blocks (256 + 44 padded)
        let mut image = vec![0u8; 300];
        for (i, b) in image.iter_mut().enumerate() {
            *b = (i % 251) as u8;
        }
        let uf2 = to_uf2(&image, 0x1000_0000, UF2_FAMILY_RP2040);
        assert_eq!(uf2.len(), 2 * 512);

        for (seq, block) in uf2.chunks(512).enumerate() {
            assert_eq!(u32_at(block, 0x00), MAGIC_START0);
            assert_eq!(u32_at(block, 0x04), MAGIC_START1);
            assert_eq!(u32_at(block, 0x08), FLAG_FAMILY_ID);
            assert_eq!(u32_at(block, 0x0C), 0x1000_0000 + (seq * DATA_CHUNK) as u32);
            assert_eq!(u32_at(block, 0x14), seq as u32);
            assert_eq!(u32_at(block, 0x18), 2);
            assert_eq!(u32_at(block, 0x1C), UF2_FAMILY_RP2040);
            assert_eq!(u32_at(block, 0x1FC), MAGIC_END);
        }
        // first block carries a full 256-byte chunk at its data offset
        assert_eq!(&uf2[0x20..0x20 + 256], &image[..256]);
        // second block carries the 44-byte tail, zero-padded to 256
        assert_eq!(u32_at(&uf2[512..], 0x10), 44);
        assert_eq!(&uf2[512 + 0x20..512 + 0x20 + 44], &image[256..]);
        assert!(
            uf2[512 + 0x20 + 44..512 + 0x20 + 256]
                .iter()
                .all(|&b| b == 0)
        );
    }

    #[test]
    fn empty_image_yields_no_blocks_and_exact_boundary_yields_n() {
        assert!(to_uf2(&[], 0x1000_0000, UF2_FAMILY_RP2040).is_empty());
        assert_eq!(
            to_uf2(&vec![0u8; 512], 0x1000_0000, UF2_FAMILY_RP2040).len(),
            2 * 512
        );
    }
}
