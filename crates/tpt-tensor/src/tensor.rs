use std::sync::{Arc, Mutex};

use crate::device::Device;
use crate::dtype::{DType, DTypeError, Num};
use crate::meta::{Layout, TensorMeta, contiguous_strides};
use crate::storage::{CpuStorage, Storage};

/// Reverse-mode autograd node (spec §5.1 / §5.2).
///
/// `tpt-tensor` owns this *slot* on every [`Tensor`]; `tpt-autograd` fills it
/// in with the actual tape: the node records its parent nodes, a `backward`
/// closure that scatters this node's gradient into its parents, and the
/// accumulated gradient itself.
/// The VJP closure recorded per node: receives the upstream gradient.
pub type BackwardFn = Box<dyn Fn(&Tensor) + Send + Sync>;

pub struct AutogradNode {
    pub parents: Vec<Arc<AutogradNode>>,
    pub backward: Option<BackwardFn>,
    grad: Mutex<Option<Tensor>>,
}

impl AutogradNode {
    /// An interior (non-leaf) node with parents and a backward closure.
    pub fn new(
        parents: Vec<Arc<AutogradNode>>,
        backward: Box<dyn Fn(&Tensor) + Send + Sync>,
    ) -> Self {
        AutogradNode {
            parents,
            backward: Some(backward),
            grad: Mutex::new(None),
        }
    }

    /// A leaf node (a parameter) — no parents, no backward of its own.
    pub fn leaf() -> Self {
        AutogradNode {
            parents: Vec::new(),
            backward: None,
            grad: Mutex::new(None),
        }
    }

    /// Add `g` into this node's accumulated gradient (gradient accumulation).
    ///
    /// When both the existing gradient and `g` carry autograd nodes, the sum is
    /// itself recorded on the tape (a linear add node), so the accumulated
    /// gradient remains differentiable — this is what makes double-backward
    /// (second-order derivatives) work: `tensor.grad()` returns a tensor whose
    /// value is the gradient and whose graph knows how it was computed.
    pub fn accumulate_grad(&self, g: &Tensor) {
        let mut lock = self.grad.lock().unwrap();
        match lock.take() {
            Some(existing) => *lock = Some(add_grad_tensors(&existing, g)),
            None => *lock = Some(g.clone()),
        }
    }

    pub fn set_grad(&self, g: Tensor) {
        *self.grad.lock().unwrap() = Some(g);
    }

    /// Clear this node's accumulated gradient (used between backward passes,
    /// e.g. before a double-backward / second-order pass).
    pub fn zero_grad(&self) {
        *self.grad.lock().unwrap() = None;
    }

    pub fn grad(&self) -> Option<Tensor> {
        self.grad.lock().unwrap().clone()
    }

    /// Invoke this node's backward closure with its accumulated gradient.
    pub fn run_backward(&self, grad: &Tensor) {
        if let Some(bw) = &self.backward {
            bw(grad);
        }
    }

    pub fn parents(&self) -> &[Arc<AutogradNode>] {
        &self.parents
    }
}

/// Sum two gradient tensors, keeping the result on the tape when either side
/// carries an autograd node (gradient flows equally to both parents — the same
/// VJP as element-wise addition).
fn add_grad_tensors(a: &Tensor, b: &Tensor) -> Tensor {
    let result = a.add(b);
    let a_node = a.node();
    let b_node = b.node();
    if a_node.is_none() && b_node.is_none() {
        return result;
    }
    let parents: Vec<Arc<AutogradNode>> = a_node.iter().chain(b_node.iter()).cloned().collect();
    let closure_parents = parents.clone();
    let mut node_result = result;
    node_result.set_node(Arc::new(AutogradNode::new(
        parents,
        Box::new(move |grad: &Tensor| {
            for p in &closure_parents {
                p.accumulate_grad(grad);
            }
        }),
    )));
    node_result
}

/// The universal tensor: every crate consumes this handle.
///
/// Constructed over an [`Arc<dyn Storage>`] (copy-on-write) plus a
/// [`TensorMeta`] describing its logical view. Zero-copy operations rewrite only
/// the metadata; in-place mutation bumps `meta.version` so the autograd tape can
/// reject illegal mutation.
#[derive(Clone)]
pub struct Tensor {
    pub(crate) meta: TensorMeta,
    pub(crate) storage: Arc<dyn Storage>,
    pub(crate) autograd: Option<Arc<AutogradNode>>,
}

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tensor")
            .field("shape", &self.meta.shape)
            .field("dtype", &self.meta.dtype)
            .field("device", &self.meta.device)
            .field("layout", &self.meta.layout)
            .field("version", &self.meta.version)
            .finish()
    }
}

impl Tensor {
    /// Wrap any `Storage` implementation as a 1-D tensor of its flat length.
    pub fn new(storage: impl Storage + 'static) -> Self {
        let dtype = storage.dtype();
        let device = storage.device();
        let numel = storage.byte_len() / dtype.size_of();
        let shape = vec![numel];
        let strides = contiguous_strides(&shape);
        Tensor {
            meta: TensorMeta {
                shape,
                strides,
                dtype,
                device,
                layout: Layout::C,
                version: 0,
            },
            storage: Arc::new(storage),
            autograd: None,
        }
    }

    /// Convenience constructor for a CPU-backed buffer.
    pub fn from_cpu(storage: CpuStorage) -> Self {
        Self::new(storage)
    }

    /// Build a tensor from typed elements, inferred to a 1-D `shape`.
    pub fn from_typed<T: Num>(values: impl IntoIterator<Item = T>) -> Self {
        Self::from_cpu(CpuStorage::from_typed(values))
    }

    /// Allocate a zeroed tensor of `shape` on `device` (CPU only for now).
    pub fn zeros(shape: &[usize], dtype: DType, device: Device) -> Self {
        assert!(
            device.is_cpu(),
            "only CPU storage is implemented; got {device}"
        );
        let numel: usize = shape.iter().product();
        let storage = CpuStorage::zeros(numel, dtype);
        let strides = contiguous_strides(shape);
        Tensor {
            meta: TensorMeta {
                shape: shape.to_vec(),
                strides,
                dtype,
                device,
                layout: Layout::C,
                version: 0,
            },
            storage: Arc::new(storage),
            autograd: None,
        }
    }

    /// Allocate a ones tensor (f64) of `shape` on `device`, used as a gradient seed.
    pub fn ones(shape: &[usize], device: Device) -> Self {
        Self::ones_typed(shape, device, DType::F64)
    }

    /// Allocate a ones tensor of an explicit dtype — gradient seeds must
    /// match the dtype of the tensors they flow into (f32 GPU tapes etc.).
    pub fn ones_typed(shape: &[usize], device: Device, dtype: DType) -> Self {
        assert!(
            device.is_cpu(),
            "only CPU storage is implemented; got {device}"
        );
        let numel: usize = shape.iter().product();
        match dtype {
            DType::F64 => Self::from_typed(vec![1.0f64; numel])
                .reshape(shape)
                .unwrap(),
            DType::F32 => Self::from_typed(vec![1.0f32; numel])
                .reshape(shape)
                .unwrap(),
            other => panic!("ones_typed: unsupported dtype {other:?}"),
        }
    }

    /// Build a CPU tensor directly from little-endian raw bytes (e.g. a
    /// serialized buffer). Length must equal `numel(shape) * dtype.size_of()`.
    pub fn from_le_bytes(bytes: Vec<u8>, dtype: DType, shape: &[usize]) -> Tensor {
        Self::from_cpu(CpuStorage::from_bytes(bytes, dtype))
            .reshape(shape)
            .unwrap()
    }

    pub fn shape(&self) -> &[usize] {
        &self.meta.shape
    }
    pub fn strides(&self) -> &[usize] {
        &self.meta.strides
    }
    pub fn dtype(&self) -> DType {
        self.meta.dtype
    }
    pub fn device(&self) -> Device {
        self.meta.device
    }
    pub fn layout(&self) -> Layout {
        self.meta.layout
    }
    pub fn version(&self) -> u64 {
        self.meta.version
    }
    pub fn ndim(&self) -> usize {
        self.meta.ndim()
    }
    pub fn numel(&self) -> usize {
        self.meta.numel()
    }

    /// Raw little-endian bytes of the backing storage (zero-copy).
    pub fn as_bytes(&self) -> &[u8] {
        self.storage.as_bytes()
    }

    pub fn storage(&self) -> &Arc<dyn Storage> {
        &self.storage
    }

    /// Attach a leaf autograd node (marks this tensor as a parameter).
    pub fn with_autograd(mut self) -> Self {
        self.autograd = Some(Arc::new(AutogradNode::leaf()));
        self
    }

    pub fn requires_grad(&self) -> bool {
        self.autograd.is_some()
    }

    /// The autograd node attached to this tensor, if any.
    pub fn node(&self) -> Option<Arc<AutogradNode>> {
        self.autograd.clone()
    }

    /// Replace the attached autograd node (used by `tpt-autograd` when building the tape).
    pub fn set_node(&mut self, node: Arc<AutogradNode>) {
        self.autograd = Some(node);
    }

    /// Accumulated gradient for this tensor, if it has been back-propagated.
    pub fn grad(&self) -> Option<Tensor> {
        self.autograd.as_ref().and_then(|n| n.grad())
    }

    /// Reconstruct the typed element vector from storage bytes, honoring
    /// strides (so zero-copy views like `transpose` read correctly).
    ///
    /// Works for any `Storage` (CPU, mmap-shared, GPU-host-staged, ...) since
    /// it only reads the backend-agnostic byte view.
    pub fn to_vec<T: Num>(&self) -> Result<Vec<T>, DTypeError> {
        if self.storage.dtype() != T::DTYPE {
            return Err(DTypeError::Mismatch {
                expected: T::DTYPE.name(),
                found: self.storage.dtype().name(),
            });
        }
        let w = T::DTYPE.size_of();
        let bytes = self.storage.as_bytes();
        let mut out = Vec::with_capacity(self.numel());
        for off in self.logical_byte_offsets() {
            out.push(T::from_le(&bytes[off..off + w]));
        }
        Ok(out)
    }

    /// Byte offset of every logical element in row-major logical order.
    fn logical_byte_offsets(&self) -> Vec<usize> {
        let shape = &self.meta.shape;
        let strides = &self.meta.strides;
        let w = self.meta.dtype.size_of();
        let ndim = shape.len();
        let mut offsets = Vec::with_capacity(self.numel());
        if ndim == 0 {
            return offsets;
        }
        let mut idx = vec![0usize; ndim];
        loop {
            let off: usize = idx.iter().zip(strides).map(|(i, s)| i * s).sum::<usize>() * w;
            offsets.push(off);
            let mut d = ndim;
            loop {
                d -= 1;
                idx[d] += 1;
                if idx[d] < shape[d] {
                    break;
                } else if d == 0 {
                    return offsets;
                } else {
                    idx[d] = 0;
                }
            }
        }
    }

    /// Zero-copy, metadata-only reshape. Valid only when the buffer is
    /// contiguous and the new shape holds the same number of elements.
    pub fn reshape(&self, shape: &[usize]) -> Result<Tensor, DTypeError> {
        let new_numel: usize = shape.iter().product();
        if new_numel != self.numel() {
            return Err(DTypeError::ShapeMismatch {
                given: self.numel(),
                need: new_numel,
            });
        }
        let mut meta = self.meta.clone();
        meta.shape = shape.to_vec();
        meta.strides = contiguous_strides(shape);
        meta.layout = Layout::C;
        Ok(Tensor {
            meta,
            storage: self.storage.clone(),
            autograd: self.autograd.clone(),
        })
    }

    /// Bump the in-place mutation version, invalidating any attached tape view.
    pub fn mark_mutated(&mut self) {
        self.meta = self.meta.clone().bumped_version();
        self.autograd = None;
    }

    /// Overwrite this tensor's element values in place (used by optimizers).
    /// `values` must match `numel()` and the dtype must be `f64`. Bumps the
    /// mutation version and detaches the autograd tape.
    pub fn set_values(&mut self, values: Vec<f64>) {
        assert_eq!(
            values.len(),
            self.numel(),
            "set_values: length must equal numel"
        );
        assert_eq!(self.dtype(), DType::F64, "set_values: only f64 supported");
        self.storage = Arc::new(CpuStorage::from_typed(values));
        self.meta = self.meta.clone().bumped_version();
        self.autograd = None;
    }

    // --- CPU math (scaffold: f64, row-major). Differentiable wrappers live in
    //     `tpt-autograd`; these are the plain primitive ops. ---

    /// Element-wise add with broadcasting.
    pub fn add(&self, other: &Tensor) -> Tensor {
        binary(self, other, |x, y| x + y)
    }

    /// Element-wise mul with broadcasting.
    pub fn mul(&self, other: &Tensor) -> Tensor {
        binary(self, other, |x, y| x * y)
    }

    /// Multiply every element by a scalar.
    pub fn scale(&self, s: f64) -> Tensor {
        let a = self.to_vec::<f64>().unwrap();
        Tensor::from_typed(a.iter().map(|x| x * s))
            .reshape(self.shape())
            .unwrap()
    }

    /// Sum this tensor's elements into `shape`, reducing over broadcast axes.
    /// Used to scatter gradients back to a (possibly broadcast) parameter shape.
    pub fn sum_to(&self, shape: &[usize]) -> Tensor {
        let gi = broadcast_flat_indices(shape, self.shape());
        let sv = self.to_vec::<f64>().unwrap();
        let mut acc = vec![0.0f64; shape.iter().product()];
        for (pos, &tidx) in gi.iter().enumerate() {
            acc[tidx] += sv[pos];
        }
        Tensor::from_typed(acc).reshape(shape).unwrap()
    }

    /// Element-wise subtraction with broadcasting.
    pub fn sub(&self, other: &Tensor) -> Tensor {
        binary(self, other, |x, y| x - y)
    }

    /// Element-wise division with broadcasting.
    pub fn div(&self, other: &Tensor) -> Tensor {
        binary(self, other, |x, y| x / y)
    }

    /// Element-wise negation.
    pub fn neg(&self) -> Tensor {
        let a = self.to_vec::<f64>().unwrap();
        Tensor::from_typed(a.iter().map(|x| -x))
            .reshape(self.shape())
            .unwrap()
    }

    /// Element-wise natural log.
    pub fn log(&self) -> Tensor {
        let a = self.to_vec::<f64>().unwrap();
        Tensor::from_typed(a.iter().map(|x| x.ln()))
            .reshape(self.shape())
            .unwrap()
    }

    /// Element-wise exponential.
    pub fn exp(&self) -> Tensor {
        let a = self.to_vec::<f64>().unwrap();
        Tensor::from_typed(a.iter().map(|x| x.exp()))
            .reshape(self.shape())
            .unwrap()
    }

    /// Element-wise absolute value.
    pub fn abs(&self) -> Tensor {
        let a = self.to_vec::<f64>().unwrap();
        Tensor::from_typed(a.iter().map(|x| x.abs()))
            .reshape(self.shape())
            .unwrap()
    }

    /// Sigmoid, element-wise.
    pub fn sigmoid(&self) -> Tensor {
        let a = self.to_vec::<f64>().unwrap();
        Tensor::from_typed(a.iter().map(|x| 1.0 / (1.0 + (-x).exp())))
            .reshape(self.shape())
            .unwrap()
    }

    /// Ones tensor with this tensor's shape **and dtype** (seeds must match
    /// the dtype of the graph they flow into).
    pub fn ones_like(&self) -> Tensor {
        Tensor::ones_typed(self.shape(), self.device(), self.dtype())
    }

    /// Sum of all elements as a scalar (shape `[1]`).
    pub fn sum_all(&self) -> Tensor {
        let s = self.to_vec::<f64>().unwrap().iter().sum::<f64>();
        Tensor::from_typed(vec![s]).reshape(&[1]).unwrap()
    }

    /// Mean of all elements as a scalar (shape `[1]`).
    pub fn mean_all(&self) -> Tensor {
        let v = self.to_vec::<f64>().unwrap();
        let m = v.iter().sum::<f64>() / (v.len() as f64);
        Tensor::from_typed(vec![m]).reshape(&[1]).unwrap()
    }

    /// Softmax over the last axis (any rank).
    pub fn softmax(&self) -> Tensor {
        let ndim = self.ndim();
        assert!(ndim >= 1, "softmax requires rank >= 1");
        let c = self.shape()[ndim - 1];
        let rows = self.numel() / c;
        let v = self.to_vec::<f64>().unwrap();
        let mut out = vec![0.0f64; rows * c];
        for i in 0..rows {
            let row = &v[i * c..(i + 1) * c];
            let m = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let e: Vec<f64> = row.iter().map(|x| (x - m).exp()).collect();
            let s = e.iter().sum::<f64>();
            for j in 0..c {
                out[i * c + j] = e[j] / s;
            }
        }
        Tensor::from_typed(out).reshape(self.shape()).unwrap()
    }

    /// Batched 3-D matrix multiply: `[B, M, K] @ [B, K, N] -> [B, M, N]` (CPU, f64).
    pub fn bmm(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.ndim(), 3, "bmm requires 3-D inputs");
        assert_eq!(other.ndim(), 3, "bmm requires 3-D inputs");
        let b = self.shape()[0];
        assert_eq!(b, other.shape()[0], "bmm batch mismatch");
        let (m, k) = (self.shape()[1], self.shape()[2]);
        let (k2, n) = (other.shape()[1], other.shape()[2]);
        assert_eq!(k, k2, "bmm inner-dim mismatch: {k} vs {k2}");
        let a = self.to_vec::<f64>().unwrap();
        let bm = other.to_vec::<f64>().unwrap();
        let mut out = vec![0.0f64; b * m * n];
        for bb in 0..b {
            let a_off = bb * m * k;
            let b_off = bb * k * n;
            let o_off = bb * m * n;
            for i in 0..m {
                for j in 0..n {
                    let mut s = 0.0;
                    for kk in 0..k {
                        s += a[a_off + i * k + kk] * bm[b_off + kk * n + j];
                    }
                    out[o_off + i * n + j] = s;
                }
            }
        }
        Tensor::from_typed(out).reshape(&[b, m, n]).unwrap()
    }

    /// 2-D matrix multiply (CPU, f64).
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.ndim(), 2, "matmul requires 2-D inputs");
        assert_eq!(other.ndim(), 2, "matmul requires 2-D inputs");
        let (m, k) = (self.shape()[0], self.shape()[1]);
        let (k2, n) = (other.shape()[0], other.shape()[1]);
        assert_eq!(k, k2, "matmul inner-dim mismatch: {k} vs {k2}");
        let a = self.to_vec::<f64>().unwrap();
        let b = other.to_vec::<f64>().unwrap();
        let mut out = vec![0.0f64; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0;
                for kk in 0..k {
                    s += a[i * k + kk] * b[kk * n + j];
                }
                out[i * n + j] = s;
            }
        }
        Tensor::from_typed(out).reshape(&[m, n]).unwrap()
    }

    /// Transpose the last two axes (metadata-only; 2-D here).
    pub fn transpose(&self) -> Tensor {
        assert_eq!(self.ndim(), 2, "transpose scaffold supports 2-D only");
        let mut meta = self.meta.clone();
        meta.shape = vec![self.shape()[1], self.shape()[0]];
        meta.strides = vec![self.strides()[1], self.strides()[0]];
        meta.layout = Layout::Strided;
        Tensor {
            meta,
            storage: self.storage.clone(),
            autograd: self.autograd.clone(),
        }
    }

    /// Swap the last two axes (metadata-only strided view; any rank >= 2).
    /// The result is non-contiguous — call [`Tensor::contiguous`] before a
    /// zero-copy `reshape` of permuted data.
    pub fn transpose_last_two(&self) -> Tensor {
        let ndim = self.ndim();
        assert!(ndim >= 2, "transpose_last_two requires rank >= 2");
        let mut meta = self.meta.clone();
        let last = ndim - 1;
        meta.shape.swap(last - 1, last);
        meta.strides.swap(last - 1, last);
        meta.layout = Layout::Strided;
        Tensor {
            meta,
            storage: self.storage.clone(),
            autograd: self.autograd.clone(),
        }
    }

    /// Materialize a (possibly strided) view into a fresh contiguous buffer.
    /// Values are copied in row-major logical order; the autograd node is kept
    /// so gradients flow through the copy unchanged.
    pub fn contiguous(&self) -> Tensor {
        if self.meta.layout == Layout::C {
            return self.clone();
        }
        let data = self.to_vec::<f64>().unwrap();
        Tensor::from_typed(data).reshape(self.shape()).unwrap()
    }
}

/// Element-wise binary op with NumPy-style broadcasting (right-aligned shapes).
fn binary(lhs: &Tensor, rhs: &Tensor, f: impl Fn(f64, f64) -> f64) -> Tensor {
    let shape = broadcast_shapes(lhs.shape(), rhs.shape());
    let la = lhs.to_vec::<f64>().unwrap();
    let lb = rhs.to_vec::<f64>().unwrap();
    let l_idx = broadcast_flat_indices(lhs.shape(), &shape);
    let r_idx = broadcast_flat_indices(rhs.shape(), &shape);
    let out: Vec<f64> = (0..shape.iter().product())
        .map(|k| f(la[l_idx[k]], lb[r_idx[k]]))
        .collect();
    Tensor::from_typed(out).reshape(&shape).unwrap()
}

/// Broadcast two right-aligned shapes; panics on incompatible dims.
fn broadcast_shapes(a: &[usize], b: &[usize]) -> Vec<usize> {
    let n = a.len().max(b.len());
    let mut out = vec![1usize; n];
    for k in 0..n {
        let da = if k < n - a.len() {
            1
        } else {
            a[k - (n - a.len())]
        };
        let db = if k < n - b.len() {
            1
        } else {
            b[k - (n - b.len())]
        };
        assert!(
            da == db || da == 1 || db == 1,
            "incompatible broadcast: {a:?} vs {b:?}"
        );
        out[k] = da.max(db);
    }
    out
}

/// For every flat index of `shape`, the linear index into `os` (right-aligned,
/// broadcast dims map to 0). `os` must broadcast to `shape`.
fn broadcast_flat_indices(os: &[usize], shape: &[usize]) -> Vec<usize> {
    let off = shape.len() - os.len();
    let mut strides = vec![1usize; os.len()];
    for k in (0..os.len().saturating_sub(1)).rev() {
        strides[k] = strides[k + 1] * os[k + 1].max(1);
    }
    let numel: usize = shape.iter().product();
    let mut out = Vec::with_capacity(numel);
    let mut coord = vec![0usize; shape.len()];
    loop {
        let mut idx = 0usize;
        for k in 0..os.len() {
            let axis = if os[k] == 1 { 0 } else { coord[off + k] };
            idx += axis * strides[k];
        }
        out.push(idx);
        let mut d = shape.len();
        loop {
            d -= 1;
            coord[d] += 1;
            if coord[d] < shape[d] {
                break;
            } else if d == 0 {
                return out;
            } else {
                coord[d] = 0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeros_shape_and_numel() {
        let t = Tensor::zeros(&[2, 3, 4], DType::F64, Device::Cpu);
        assert_eq!(t.shape(), &[2, 3, 4]);
        assert_eq!(t.numel(), 24);
        assert_eq!(t.dtype(), DType::F64);
        assert_eq!(t.device(), Device::Cpu);
        assert!(t.as_bytes().iter().all(|&b| b == 0));
    }

    #[test]
    fn from_typed_roundtrip() {
        let t = Tensor::from_typed(vec![1.0_f64, 2.0, 3.0, 4.0]);
        assert_eq!(t.shape(), &[4]);
        assert_eq!(t.to_vec::<f64>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn reshape_keeps_storage() {
        let t = Tensor::from_typed(vec![1u8, 2, 3, 4, 5, 6]);
        let r = t.reshape(&[2, 3]).unwrap();
        assert_eq!(r.shape(), &[2, 3]);
        assert_eq!(r.to_vec::<u8>().unwrap(), vec![1, 2, 3, 4, 5, 6]);
        assert!(Arc::ptr_eq(&t.storage, &r.storage));
    }

    #[test]
    fn reshape_rejects_bad_numel() {
        let t = Tensor::from_typed(vec![1.0_f64, 2.0, 3.0]);
        assert!(t.reshape(&[2, 2]).is_err());
    }

    #[test]
    fn autograd_slot_and_versioning() {
        let mut t = Tensor::from_typed(vec![1.0_f64, 2.0]).with_autograd();
        assert!(t.requires_grad());
        let v0 = t.version();
        t.mark_mutated();
        assert!(!t.requires_grad());
        assert_eq!(t.version(), v0 + 1);
    }

    #[test]
    fn cpu_math_basics() {
        let a = Tensor::from_typed(vec![1.0_f64, 2.0, 3.0]);
        let b = Tensor::from_typed(vec![10.0_f64, 20.0, 30.0]);
        assert_eq!(a.add(&b).to_vec::<f64>().unwrap(), vec![11.0, 22.0, 33.0]);
        assert_eq!(a.mul(&b).to_vec::<f64>().unwrap(), vec![10.0, 40.0, 90.0]);
        assert_eq!(a.scale(2.0).to_vec::<f64>().unwrap(), vec![2.0, 4.0, 6.0]);
        let s = Tensor::from_typed(vec![2.0_f64]);
        assert_eq!(a.add(&s).to_vec::<f64>().unwrap(), vec![3.0, 4.0, 5.0]);
    }

    #[test]
    fn matmul_works() {
        let a = Tensor::from_typed(vec![1.0_f64, 2.0, 3.0, 4.0])
            .reshape(&[2, 2])
            .unwrap();
        let b = Tensor::from_typed(vec![0.0_f64, 1.0, 1.0, 0.0])
            .reshape(&[2, 2])
            .unwrap();
        let c = a.matmul(&b);
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.to_vec::<f64>().unwrap(), vec![2.0, 1.0, 4.0, 3.0]);
        let t = c.transpose();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn bmm_batched_matmul() {
        // batch 0: [[1,2],[3,4]] @ [[1,0],[0,1]] = [[1,2],[3,4]]
        // batch 1: [[5,6],[7,8]] @ [[0,1],[1,0]] = [[6,5],[8,7]]
        let a = Tensor::from_typed(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .reshape(&[2, 2, 2])
            .unwrap();
        let b = Tensor::from_typed(vec![1.0_f64, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0])
            .reshape(&[2, 2, 2])
            .unwrap();
        let c = a.bmm(&b);
        assert_eq!(c.shape(), &[2, 2, 2]);
        assert_eq!(
            c.to_vec::<f64>().unwrap(),
            vec![1.0, 2.0, 3.0, 4.0, 6.0, 5.0, 8.0, 7.0]
        );
    }

    #[test]
    fn transpose_last_two_and_contiguous() {
        // [B=2, T=2, D=2] permute last two -> [2, 2, 2] with T/D swapped.
        let x = Tensor::from_typed(vec![
            1.0_f64, 2.0, 3.0, 4.0, // batch 0: t0=(1,2) t1=(3,4)
            5.0_f64, 6.0, 7.0, 8.0, // batch 1
        ])
        .reshape(&[2, 2, 2])
        .unwrap();
        let p = x.transpose_last_two();
        assert_eq!(p.shape(), &[2, 2, 2]);
        // logical (b, d, t): batch0 -> [(1,3),(2,4)], batch1 -> [(5,7),(6,8)]
        assert_eq!(
            p.to_vec::<f64>().unwrap(),
            vec![1.0, 3.0, 2.0, 4.0, 5.0, 7.0, 6.0, 8.0]
        );
        // contiguous materializes the same logical order into fresh storage
        let c = p.contiguous();
        assert_eq!(c.to_vec::<f64>().unwrap(), p.to_vec::<f64>().unwrap());
        // reshape of the contiguous copy is now safe and correct
        let r = c.reshape(&[4, 2]).unwrap();
        assert_eq!(
            r.to_vec::<f64>().unwrap(),
            vec![1.0, 3.0, 2.0, 4.0, 5.0, 7.0, 6.0, 8.0]
        );
    }

    #[test]
    fn softmax_3d_last_axis() {
        let x = Tensor::from_typed(vec![0.0_f64, 0.0, 1.0, 3.0])
            .reshape(&[1, 2, 2])
            .unwrap();
        let s = x.softmax();
        assert_eq!(s.shape(), &[1, 2, 2]);
        let v = s.to_vec::<f64>().unwrap();
        assert!((v[0] - 0.5).abs() < 1e-9 && (v[1] - 0.5).abs() < 1e-9);
        assert!((v[2] - 0.11920292).abs() < 1e-6 && (v[3] - 0.88079708).abs() < 1e-6);
    }
}
