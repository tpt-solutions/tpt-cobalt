//! Sliding-window KV cache for autoregressive decoding.
//!
//! Stores K and V tensors for every transformer layer on the host. When the
//! sequence length reaches `context_len`, the oldest token's K/V vectors are
//! dropped (shift-left) so decoding can continue indefinitely.

/// Host-side KV cache with one K and one V buffer per transformer layer.
///
/// Layout per layer: contiguous `f32` slice of length
/// `context_len × num_kv_heads × head_dim`.
/// Valid data occupies `[0, seq_len × row_size)`.
#[derive(Debug)]
pub struct KvCache {
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
    pub seq_len: usize,
    context_len: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl KvCache {
    /// Allocate a zeroed cache for `num_layers` layers.
    pub fn new(num_layers: u32, context_len: u32, num_kv_heads: u32, head_dim: u32) -> Self {
        let nl = num_layers as usize;
        let cl = context_len as usize;
        let h = num_kv_heads.max(1) as usize;
        let d = head_dim.max(1) as usize;
        let cap = cl * h * d;
        Self {
            keys: vec![vec![0.0f32; cap]; nl],
            values: vec![vec![0.0f32; cap]; nl],
            seq_len: 0,
            context_len: cl,
            num_kv_heads: h,
            head_dim: d,
        }
    }

    /// Append one token's K and V vectors for a given layer.
    ///
    /// When the cache is full, the oldest token is evicted (ring-buffer
    /// shift) before the new vectors are written.
    pub fn append(&mut self, layer: usize, k_row: &[f32], v_row: &[f32]) {
        let row_size = self.num_kv_heads * self.head_dim;

        if self.seq_len == self.context_len {
            // Evict the oldest token: shift all rows left by one.
            let keys = &mut self.keys[layer];
            keys.copy_within(row_size.., 0);
            let values = &mut self.values[layer];
            values.copy_within(row_size.., 0);
            // Only decrement once — seq_len is shared across layers, so
            // only the first layer's append should update it.
            if layer == 0 {
                self.seq_len -= 1;
            }
        }

        let pos = self.seq_len * row_size;
        let copy_len = k_row
            .len()
            .min(row_size)
            .min(self.keys[layer].len().saturating_sub(pos));
        if copy_len > 0 {
            self.keys[layer][pos..pos + copy_len].copy_from_slice(&k_row[..copy_len]);
            self.values[layer][pos..pos + copy_len].copy_from_slice(&v_row[..copy_len]);
        }

        // Advance seq_len only after the last layer has appended.
        if layer + 1 == self.keys.len() {
            self.seq_len += 1;
        }
    }

    /// Return the valid slice of K vectors for `layer` as `[seq_len, row_size]` data.
    pub fn get_k(&self, layer: usize) -> &[f32] {
        let row_size = self.num_kv_heads * self.head_dim;
        &self.keys[layer][..self.seq_len * row_size]
    }

    /// Return the valid slice of V vectors for `layer` as `[seq_len, row_size]` data.
    pub fn get_v(&self, layer: usize) -> &[f32] {
        let row_size = self.num_kv_heads * self.head_dim;
        &self.values[layer][..self.seq_len * row_size]
    }

    /// Return the K slice for `layer` at an explicit token count.
    ///
    /// Use this after `append` to include the just-written token whose position
    /// has been filled but `seq_len` has not yet been incremented (because
    /// `seq_len` only advances after the final layer has appended).
    pub fn get_k_at(&self, layer: usize, count: usize) -> &[f32] {
        let row_size = self.num_kv_heads * self.head_dim;
        let end = (count * row_size).min(self.keys[layer].len());
        &self.keys[layer][..end]
    }

    /// Return the V slice for `layer` at an explicit token count.
    pub fn get_v_at(&self, layer: usize, count: usize) -> &[f32] {
        let row_size = self.num_kv_heads * self.head_dim;
        let end = (count * row_size).min(self.values[layer].len());
        &self.values[layer][..end]
    }

    /// Zero out all K/V data and reset the sequence counter.
    pub fn reset(&mut self) {
        self.seq_len = 0;
        for (k, v) in self.keys.iter_mut().zip(self.values.iter_mut()) {
            k.fill(0.0);
            v.fill(0.0);
        }
    }

    /// Total number of layers.
    pub fn num_layers(&self) -> usize {
        self.keys.len()
    }

    /// Maximum sequence length before eviction occurs.
    pub fn context_len(&self) -> usize {
        self.context_len
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_row(val: f32, size: usize) -> Vec<f32> {
        vec![val; size]
    }

    #[test]
    fn append_and_retrieve_single_layer() {
        let mut cache = KvCache::new(1, 4, 1, 8);
        let row_size = 8; // 1 head × 8 head_dim

        let k0 = make_row(1.0, row_size);
        let v0 = make_row(2.0, row_size);
        cache.append(0, &k0, &v0);

        assert_eq!(cache.seq_len, 1);
        assert!(cache.get_k(0).iter().all(|&x| x == 1.0));
        assert!(cache.get_v(0).iter().all(|&x| x == 2.0));
    }

    #[test]
    fn seq_len_advances_after_all_layers() {
        let mut cache = KvCache::new(3, 8, 2, 4);
        let row_size = 8; // 2 × 4
        let k = make_row(0.5, row_size);
        let v = make_row(0.5, row_size);
        // Append for layers 0, 1, 2 → seq_len should advance to 1.
        cache.append(0, &k, &v);
        cache.append(1, &k, &v);
        cache.append(2, &k, &v);
        assert_eq!(cache.seq_len, 1);
    }

    #[test]
    fn eviction_at_context_len() {
        // context_len = 2, 1 layer
        let mut cache = KvCache::new(1, 2, 1, 4);
        let row = 4usize;

        // token 0: K = 1.0
        cache.append(0, &make_row(1.0, row), &make_row(10.0, row));
        assert_eq!(cache.seq_len, 1);

        // token 1: K = 2.0
        cache.append(0, &make_row(2.0, row), &make_row(20.0, row));
        assert_eq!(cache.seq_len, 2);

        // token 2: K = 3.0 — should evict token 0
        cache.append(0, &make_row(3.0, row), &make_row(30.0, row));
        assert_eq!(cache.seq_len, 2);

        // After eviction: get_k should be [2.0×row, 3.0×row]
        let k = cache.get_k(0);
        assert!(
            k[..row].iter().all(|&x| (x - 2.0).abs() < 1e-6),
            "first row should be 2.0 after eviction"
        );
        assert!(
            k[row..].iter().all(|&x| (x - 3.0).abs() < 1e-6),
            "second row should be 3.0"
        );
    }

    #[test]
    fn reset_clears_all_data() {
        let mut cache = KvCache::new(2, 4, 1, 4);
        let row = 4usize;
        cache.append(0, &make_row(5.0, row), &make_row(5.0, row));
        cache.append(1, &make_row(5.0, row), &make_row(5.0, row));
        cache.reset();
        assert_eq!(cache.seq_len, 0);
        assert!(cache.get_k(0).is_empty());
    }

    #[test]
    fn context_len_accessor() {
        let cache = KvCache::new(4, 2048, 8, 128);
        assert_eq!(cache.context_len(), 2048);
        assert_eq!(cache.num_layers(), 4);
    }
}
