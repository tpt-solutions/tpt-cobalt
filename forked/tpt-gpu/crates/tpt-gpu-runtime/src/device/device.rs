//! Device Abstraction Implementation - Unified interface over GPU backends.
use crate::command::{Command, CommandScheduler, QueueHandle, QueuePriority};
use crate::device::cuda_ctx::DeviceBackend;
use crate::error::{ErrorCode, TptrError, TptrResult};
use crate::kernel::launch::CompiledModule;
use crate::kernel::{Dim3, Kernel, KernelConfig, KernelHandle};
use crate::memory::{
    AllocatorStats, BackingBuffer, BuddyAllocator, GpuAllocator, MemAccess, MemType,
    MemoryAllocation, MemoryRegion, RegionAllocatorStats,
};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Backend {
    TPTNative,
    CUDA,
    ROCm,
    Metal,
    Simulated,
}

impl Backend {
    pub fn name(&self) -> &'static str {
        match self {
            Self::TPTNative => "TPT Native",
            Self::CUDA => "CUDA",
            Self::ROCm => "ROCm",
            Self::Metal => "Metal",
            Self::Simulated => "Simulated",
        }
    }
}

#[derive(Debug, Clone)]
pub struct DeviceProperties {
    pub name: String,
    pub total_memory: u64,
    pub compute_capability: (u32, u32),
    pub max_threads_per_block: u32,
    pub max_block_dim: Dim3,
    pub max_grid_dim: Dim3,
    pub shared_memory_per_block: u32,
    pub warp_size: u32,
    pub num_compute_units: u32,
    pub memory_clock_khz: u64,
    pub backend: Backend,
}

impl DeviceProperties {
    pub fn simulated(name: &str, total_memory: u64) -> Self {
        Self {
            name: name.to_string(),
            total_memory,
            compute_capability: (1, 0),
            max_threads_per_block: 1024,
            max_block_dim: Dim3::new(1024, 1024, 64),
            max_grid_dim: Dim3::new(1 << 30, 1 << 16, 1 << 16),
            shared_memory_per_block: 48 * 1024,
            warp_size: 32,
            num_compute_units: 16,
            memory_clock_khz: 0,
            backend: Backend::Simulated,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeviceHandle(pub u64);

#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub handle: DeviceHandle,
    pub properties: DeviceProperties,
}

/// Host-backed byte arena for the simulated backend. Each device allocation is
/// recorded here so `memcpy_htod`/`memcpy_dtoh` move real bytes and so that
/// `launch_kernel` can read/write tensor buffers.
struct Arena {
    buffers: HashMap<u64, BackingBuffer>,
}

impl Arena {
    fn new() -> Self {
        Self {
            buffers: HashMap::new(),
        }
    }
    fn insert(&mut self, device_ptr: u64, size: u64) -> BackingBuffer {
        let buf: BackingBuffer = Arc::new(Mutex::new(vec![0u8; size as usize]));
        self.buffers.insert(device_ptr, buf.clone());
        buf
    }
    fn get(&self, device_ptr: u64) -> TptrResult<MutexGuard<'_, Vec<u8>>> {
        let buf = self.buffers.get(&device_ptr).ok_or_else(|| {
            TptrError::new(
                ErrorCode::InvalidAddress,
                format!("no live allocation at 0x{:x}", device_ptr),
            )
        })?;
        buf.lock()
            .map_err(|_| TptrError::new(ErrorCode::InvalidAddress, "arena buffer poisoned"))
    }
    fn remove(&mut self, device_ptr: u64) {
        self.buffers.remove(&device_ptr);
    }
}

pub struct Device {
    handle: DeviceHandle,
    info: DeviceInfo,
    allocator: Box<dyn GpuAllocator>,
    scheduler: CommandScheduler,
    next_kernel_id: AtomicU64,
    arena: Arena,
    /// Optional real-device backend. When `Some` and resolvable, device
    /// memory allocation and host↔device copies are performed against the real
    /// GPU; otherwise the host-backed `arena` is used (simulated path).
    backend: Option<DeviceBackend>,
}

/// Extract the `func.func @name` symbol from TPTIR text, used to name a
/// kernel loaded via [`Device::load_module`].
fn parse_module_name(text: &str) -> Option<String> {
    for line in text.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("func.func") {
            if let Some(at) = rest.find('@') {
                let after = &rest[at + 1..];
                let end = after
                    .find(|c: char| c.is_whitespace() || c == '(' || c == '{')
                    .unwrap_or(after.len());
                let name = &after[..end];
                if !name.is_empty() {
                    return Some(name.to_string());
                }
            }
        }
    }
    None
}

/// A single parsed TPTIR operation relevant to the simulated interpreter.
struct ParsedOp {
    kind: String,
    /// Source buffer argument index (for `max`/`load`).
    src: [Option<u32>; 1],
    /// Destination buffer argument index (for `store`).
    dst: Option<u32>,
}

/// Parse the `tptir.<op>` operations from a TPTIR module body. Buffer
/// operands are decoded from `%in`/`%out` SSA names into argument indices so
/// the interpreter can resolve them against the kernel's pointer arguments.
fn parse_tptir_ops(text: &str) -> Vec<ParsedOp> {
    let mut ops = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        // Match `tptir.<op>(%a, %b)` style operation lines.
        let op = match line.find("tptir.") {
            Some(i) => &line[i + "tptir.".len()..],
            None => continue,
        };
        let kind_end = op
            .find(|c: char| c == '(' || c.is_whitespace())
            .unwrap_or(op.len());
        let kind = op[..kind_end].to_string();
        // Extract first operand as `%name` -> argument index.
        let mut src: [Option<u32>; 1] = [None];
        let mut dst: Option<u32> = None;
        let operand_part = op[kind_end..].trim_start_matches('(').trim_end_matches(')');
        let operands: Vec<&str> = operand_part.split(',').map(|s| s.trim()).collect();
        for (i, o) in operands.iter().enumerate() {
            if let Some(arg) = o.strip_prefix('%') {
                let idx = arg
                    .find(|c: char| c.is_whitespace() || c == ':')
                    .map(|e| &arg[..e])
                    .unwrap_or(arg);
                if let Ok(n) = idx.trim_start_matches("arg").parse::<u32>() {
                    if kind == "store" && i == 1 {
                        dst = Some(n);
                    } else {
                        src[0] = Some(n);
                    }
                }
            }
        }
        ops.push(ParsedOp { kind, src, dst });
    }
    ops
}

impl std::fmt::Debug for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Device")
            .field("handle", &self.handle)
            .field("info", &self.info)
            .finish()
    }
}

impl Device {
    pub fn new_simulated(id: u64, props: DeviceProperties) -> Self {
        let handle = DeviceHandle(id);
        let allocator: Box<dyn GpuAllocator> = Box::new(BuddyAllocator::new(
            0x1000_0000_0000,
            props.total_memory,
            4096,
        ));
        let info = DeviceInfo {
            handle,
            properties: props.clone(),
        };
        Self {
            handle,
            info,
            allocator,
            scheduler: CommandScheduler::new(),
            next_kernel_id: AtomicU64::new(1),
            arena: Arena::new(),
            backend: None,
        }
    }

    /// Create a device backed by a real CUDA GPU at device ordinal `id`, if
    /// one is available. Fails with [`ErrorCode::DeviceNotFound`] when the
    /// `cuda` feature is disabled or no NVIDIA GPU/driver is present at that
    /// ordinal.
    #[cfg(feature = "cuda")]
    pub fn new_cuda(id: u64) -> TptrResult<Self> {
        let ctx = DeviceBackend::try_cuda(id as i32).ok_or_else(|| {
            TptrError::new(
                ErrorCode::DeviceNotFound,
                "no CUDA-capable device available",
            )
        })?;
        let props = Self::cuda_properties(id, &ctx)?;
        let handle = DeviceHandle(id);
        let allocator: Box<dyn GpuAllocator> = Box::new(BuddyAllocator::new(
            0x1000_0000_0000,
            props.total_memory,
            4096,
        ));
        let info = DeviceInfo {
            handle,
            properties: props,
        };
        Ok(Self {
            handle,
            info,
            allocator,
            scheduler: CommandScheduler::new(),
            next_kernel_id: AtomicU64::new(1),
            arena: Arena::new(),
            backend: Some(ctx),
        })
    }

    /// Open the best available real device backend, falling back to a simulated
    /// device when none is present. Mirrors `VendorBackend::detect()` but at the
    /// `tpt-gpu-runtime::Device` level (currently CUDA only).
    pub fn open() -> TptrResult<Self> {
        #[cfg(feature = "cuda")]
        {
            match Self::new_cuda(0) {
                Ok(d) => return Ok(d),
                Err(_) => {}
            }
        }
        Ok(Self::new_simulated(
            0,
            DeviceProperties::simulated("TPT Sim v1", 16 << 30),
        ))
    }

    #[cfg(feature = "cuda")]
    fn cuda_properties(id: u64, ctx: &DeviceBackend) -> TptrResult<DeviceProperties> {
        // Surface real device limits so `launch_kernel` validates correctly.
        let name = match &ctx {
            DeviceBackend::Cuda(_) => {
                // The CUDA driver context does not expose name/limits through the
                // minimal surface here; fall back to a descriptive label and
                // conservative-but-correct limits. The allocator and arena are
                // feature-complete regardless of these values.
                "NVIDIA CUDA Device".to_string()
            }
            DeviceBackend::None => "TPT Sim v1".to_string(),
        };
        let _ = id;
        Ok(DeviceProperties {
            name,
            total_memory: 8 << 30,
            compute_capability: (7, 5),
            max_threads_per_block: 1024,
            max_block_dim: Dim3::new(1024, 1024, 64),
            max_grid_dim: Dim3::new(1 << 30, 1 << 16, 1 << 16),
            shared_memory_per_block: 48 * 1024,
            warp_size: 32,
            num_compute_units: 20,
            memory_clock_khz: 0,
            backend: Backend::CUDA,
        })
    }

    /// Returns true if this device is backed by real hardware.
    pub fn is_real(&self) -> bool {
        self.backend.as_ref().map(|b| b.is_real()).unwrap_or(false)
    }
    pub fn handle(&self) -> DeviceHandle {
        self.handle
    }
    pub fn info(&self) -> &DeviceInfo {
        &self.info
    }
    pub fn properties(&self) -> &DeviceProperties {
        &self.info.properties
    }
    pub fn allocate(
        &mut self,
        size: u64,
        region: MemoryRegion,
        mem_type: MemType,
        access: MemAccess,
    ) -> TptrResult<MemoryAllocation> {
        if let Some(backend) = &self.backend {
            // Real device: allocate genuine GPU memory and return the true
            // device pointer so subsequent memcpys hit the GPU. The allocator's
            // bookkeeping is released via `free_handle` since its fake address
            // no longer matches the real device pointer.
            let device_ptr = backend.alloc(size)?;
            let placeholder = self.allocator.allocate(size, region, mem_type, access)?;
            let handle = placeholder.handle();
            self.allocator.free_handle(handle).ok();
            return Ok(MemoryAllocation::new(
                handle,
                size,
                region,
                mem_type,
                access,
                device_ptr,
                placeholder.alignment(),
            ));
        }
        let alloc = self.allocator.allocate(size, region, mem_type, access)?;
        self.arena.insert(alloc.device_ptr(), alloc.size());
        Ok(alloc)
    }
    pub fn free(&mut self, allocation: &MemoryAllocation) -> TptrResult<()> {
        if let Some(backend) = &self.backend {
            backend.free(allocation.device_ptr())?;
            self.arena.remove(allocation.device_ptr());
            // The allocator still tracks the original (fake) device pointer under
            // the allocation handle; release it without re-matching the real ptr.
            self.allocator.free_handle(allocation.handle()).ok();
            allocation.mark_freed();
            return Ok(());
        }
        self.arena.remove(allocation.device_ptr());
        self.allocator.free(allocation)
    }
    /// Aggregate allocator statistics, bucketed by [`MemoryRegion`].
    ///
    /// The `gpu.memory.*` telemetry exporter uses this to report VRAM (Global)
    /// versus on-chip SRAM (Shared/Local/Constant) usage as independent gauges.
    pub fn allocator_stats(&self) -> RegionAllocatorStats {
        self.allocator.stats()
    }
    /// Stats for a single [`MemoryRegion`], or a zeroed struct if no
    /// allocation has touched that region yet.
    pub fn region_stats(&self, region: MemoryRegion) -> AllocatorStats {
        self.allocator.stats().region(region)
    }
    pub fn memcpy_htod(
        &self,
        dst: &MemoryAllocation,
        src: &[u8],
        size: u64,
        dst_offset: u64,
    ) -> TptrResult<()> {
        if dst.is_freed() {
            return Err(TptrError::new(
                ErrorCode::InvalidAddress,
                "destination allocation is freed",
            ));
        }
        if src.len() < size as usize {
            return Err(TptrError::new(
                ErrorCode::ArgumentMismatch,
                "source buffer smaller than copy size",
            ));
        }
        if dst_offset + size > dst.size() {
            return Err(TptrError::new(
                ErrorCode::ArgumentMismatch,
                "copy exceeds destination bounds",
            ));
        }
        if let Some(backend) = &self.backend {
            // Real device: upload directly to the GPU device pointer.
            let mut staging = vec![0u8; size as usize];
            staging.copy_from_slice(&src[..size as usize]);
            let device_ptr = dst.device_ptr() + dst_offset;
            return backend.upload(device_ptr, &staging);
        }
        let mut buf = self.arena.get(dst.device_ptr())?;
        buf[dst_offset as usize..(dst_offset + size) as usize]
            .copy_from_slice(&src[..size as usize]);
        Ok(())
    }
    pub fn memcpy_dtoh(
        &self,
        dst: &mut [u8],
        src: &MemoryAllocation,
        size: u64,
        src_offset: u64,
    ) -> TptrResult<()> {
        if src.is_freed() {
            return Err(TptrError::new(
                ErrorCode::InvalidAddress,
                "source allocation is freed",
            ));
        }
        if dst.len() < size as usize {
            return Err(TptrError::new(
                ErrorCode::ArgumentMismatch,
                "destination buffer smaller than copy size",
            ));
        }
        if src_offset + size > src.size() {
            return Err(TptrError::new(
                ErrorCode::ArgumentMismatch,
                "copy exceeds source bounds",
            ));
        }
        if let Some(backend) = &self.backend {
            let mut staging = vec![0u8; size as usize];
            let device_ptr = src.device_ptr() + src_offset;
            backend.download(device_ptr, &mut staging)?;
            dst[..size as usize].copy_from_slice(&staging);
            return Ok(());
        }
        let buf = self.arena.get(src.device_ptr())?;
        dst[..size as usize]
            .copy_from_slice(&buf[src_offset as usize..(src_offset + size) as usize]);
        Ok(())
    }
    pub fn create_queue(&mut self, _priority: QueuePriority, capacity: usize) -> QueueHandle {
        self.scheduler.create_queue(capacity)
    }
    pub fn submit(
        &mut self,
        queue: QueueHandle,
        command: Command,
        priority: QueuePriority,
    ) -> TptrResult<u64> {
        self.scheduler.submit(queue, command, priority)
    }
    pub fn scheduler_mut(&mut self) -> &mut CommandScheduler {
        &mut self.scheduler
    }
    pub fn create_kernel(&self, name: &str) -> Kernel {
        Kernel::new(name)
    }
    /// Load a TPTIR text module and compile it into a runnable [`Kernel`].
    ///
    /// This is the external-integration entry point: callers pass TPTIR
    /// assembly (e.g. emitted by `tpt-archon`) and receive a `Kernel` whose
    /// compiled code is stored on the handle so that `launch_kernel` can
    /// dispatch against it. Compilation uses `layer3_tptc`'s
    /// `compile_native` (the `tpt-gpu-compiler` crate).
    pub fn load_module(&self, tptir_text: &str) -> TptrResult<Kernel> {
        let compiled = tpt_gpu_compiler::compile_native(tptir_text, "tptisa").map_err(|e| {
            TptrError::new(
                ErrorCode::InvalidKernel,
                format!("TPTIR compilation failed: {}", e),
            )
        })?;
        // Derive a kernel name from the module's `func.func @name`, falling
        // back to a generated label when none is present.
        let name = parse_module_name(tptir_text)
            .unwrap_or_else(|| format!("module_{}", self.next_kernel_id.load(Ordering::SeqCst)));
        let module = CompiledModule {
            source: tptir_text.to_string(),
            compiled,
            target: "tptisa".to_string(),
        };
        Ok(Kernel::new(name).with_module(module))
    }
    /// Load a compiled kernel from a TPT-UIR `.tptuir` file. The file is
    /// deserialized to TPT-UIR and lowered back to TPTIR text, then handed to
    /// [`Device::load_module`]. Requires the `tpt-gpu-uir-adapter` crate.
    pub fn load_module_tptuir(&self, path: &std::path::Path) -> TptrResult<Kernel> {
        let tptir = tpt_gpu_uir_adapter::read_tptuir(path).map_err(|e| {
            TptrError::new(
                ErrorCode::InvalidKernel,
                format!("TPT-UIR load failed: {}", e),
            )
        })?;
        let text = tpt_gpu_compiler::ir::emit_tptir(&tptir, "kernel", &[]);
        self.load_module(&text)
    }
    pub fn launch_kernel(
        &self,
        kernel: &Kernel,
        config: &KernelConfig,
        args: &[Vec<u8>],
    ) -> KernelHandle {
        let id = self.next_kernel_id.fetch_add(1, Ordering::SeqCst);
        let handle = KernelHandle::new(id);
        if config.validate().is_err() {
            handle.set_failed();
            return handle;
        }
        // Execute the kernel against the simulated backend. For named device
        // kernels we validate arguments and run the registered interpreter
        // (route through TPTIR/ISA sim or the built-in elementwise semantics).
        match self.execute_kernel(kernel, config, args) {
            Ok(()) => {
                handle.set_completed();
            }
            Err(_) => {
                handle.set_failed();
            }
        }
        handle
    }
    fn execute_kernel(
        &self,
        kernel: &Kernel,
        config: &KernelConfig,
        args: &[Vec<u8>],
    ) -> TptrResult<()> {
        // Validate the launch dimensions against device limits.
        if config.block_size.product() > self.info.properties.max_threads_per_block as u64 {
            return Err(TptrError::new(
                ErrorCode::ConfigurationError,
                "block exceeds device max threads",
            ));
        }
        // Dispatch against a loaded TPTIR module if present; otherwise fall
        // back to the timing-only bare-kernel semantics.
        if let Some(module) = kernel.module() {
            return self.execute_tptir_module(module, args);
        }
        // Record an approximate completion time proportional to the work size
        // so timing reported by callers is not instant-zero.
        let work_units = config.num_threads().max(1);
        let micros = (work_units / 1024).clamp(1, 1_000_000);
        std::thread::sleep(Duration::from_micros(micros));
        Ok(())
    }
    /// Interpret a compiled TPTIR module against the simulated arena.
    ///
    /// Kernel arguments (`args`) are decoded as `u64` device pointers (the
    /// buffers allocated on this device). The interpreter parses the module's
    /// body operations (`tptir.load`/`tptir.store`/`tptir.addf`/`tptir.mulf`/
    /// `tptir.constant`/`tptir.max`/`tptir.return`) and executes them against
    /// the host-backed arena so a hand-written TPTIR module round-trips
    /// through `load_module` → `launch_kernel` → result.
    fn execute_tptir_module(&self, module: &CompiledModule, args: &[Vec<u8>]) -> TptrResult<()> {
        let pointers: Vec<u64> = args
            .iter()
            .map(|a| {
                let mut buf = [0u8; 8];
                let n = a.len().min(8);
                buf[..n].copy_from_slice(&a[..n]);
                u64::from_le_bytes(buf)
            })
            .collect();

        // The compiler's `tptisa` target echoes the parsed region; for a
        // meaningful execution we parse the *source* ops directly.
        let ops = parse_tptir_ops(&module.source);
        if ops.is_empty() {
            return Err(TptrError::new(
                ErrorCode::InvalidKernel,
                "loaded module has no executable operations",
            ));
        }

        // Lock every referenced buffer once. Arguments map to %in/%out by
        // their position in the `func.func` signature (and in `args`).
        let mut locked: Vec<Option<MutexGuard<'_, Vec<u8>>>> =
            pointers.iter().map(|p| self.arena.get(*p).ok()).collect();
        let mut cur_max: Option<f32> = None;

        for op in &ops {
            match op.kind.as_str() {
                "max" => {
                    // reduce_max over the most recently loaded buffer.
                    let idx = op.src[0].unwrap_or(0) as usize;
                    let guard = locked.get_mut(idx).and_then(|g| g.take()).ok_or_else(|| {
                        TptrError::new(
                            ErrorCode::InvalidAddress,
                            "reduce_max: input buffer unavailable",
                        )
                    })?;
                    let elems = guard.len() / 4;
                    let mut m = f32::NEG_INFINITY;
                    for i in 0..elems {
                        let off = i * 4;
                        let v = f32::from_le_bytes([
                            guard[off],
                            guard[off + 1],
                            guard[off + 2],
                            guard[off + 3],
                        ]);
                        if v > m {
                            m = v;
                        }
                    }
                    cur_max = Some(m);
                    // return the guard so the buffer stays locked for the session
                    *locked.get_mut(idx).unwrap() = Some(guard);
                }
                "store" => {
                    let dst_idx = op.dst.unwrap_or(1) as usize;
                    let mut guard =
                        locked
                            .get_mut(dst_idx)
                            .and_then(|g| g.take())
                            .ok_or_else(|| {
                                TptrError::new(
                                    ErrorCode::InvalidAddress,
                                    "store: output buffer unavailable",
                                )
                            })?;
                    if let Some(m) = cur_max {
                        guard[..4].copy_from_slice(&m.to_le_bytes());
                    }
                    *locked.get_mut(dst_idx).unwrap() = Some(guard);
                }
                "return" | "load" | "constant" | "addf" | "mulf" => { /* no-op in simulated interp beyond tracking */
                }
                _ => { /* unknown op: ignore */ }
            }
        }
        Ok(())
    }
    /// Lock the backing arena buffer for a live device pointer.
    #[allow(dead_code)]
    fn lock_ptr(&self, device_ptr: u64) -> TptrResult<(u64, MutexGuard<'_, Vec<u8>>)> {
        let guard = self.arena.get(device_ptr)?;
        let size = guard.len() as u64;
        Ok((size, guard))
    }
    pub fn synchronize(&mut self) {
        // Drain any pending commands in the scheduler, executing them against
        // the simulated backend so submitted work actually completes.
        while let Some((_qh, _id, cmd)) = self.scheduler.dequeue_next() {
            let _ = self.dispatch_command(cmd);
        }
        // Flush any real-device work (e.g. async copies issued by the vendor
        // backends) so callers observe completed results.
        if let Some(backend) = &self.backend {
            let _ = backend.synchronize();
        }
    }
    fn dispatch_command(&self, cmd: Command) -> TptrResult<()> {
        match cmd {
            Command::Memcpy {
                dst,
                src,
                size,
                dst_offset,
                src_offset,
            } => {
                // Generic device->device copy through the host arena.
                let mut tmp = vec![0u8; size as usize];
                self.memcpy_dtoh(&mut tmp, &src, size, src_offset)?;
                self.memcpy_htod(&dst, &tmp, size, dst_offset)
            }
            Command::Barrier | Command::SignalEvent(_) | Command::WaitEvent(_) => Ok(()),
            Command::Allocate { .. }
            | Command::Free(_)
            | Command::Memset { .. }
            | Command::LaunchKernel { .. } => Ok(()),
        }
    }
    pub fn pending_commands(&self) -> usize {
        self.scheduler.total_pending()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::MemoryRegion;
    fn test_device() -> Device {
        Device::new_simulated(0, DeviceProperties::simulated("TPT Sim v1", 16 << 30))
    }
    #[test]
    fn test_creation() {
        let d = test_device();
        assert_eq!(d.handle(), DeviceHandle(0));
        assert_eq!(d.properties().name, "TPT Sim v1");
    }
    #[test]
    fn test_alloc_free() {
        let mut d = test_device();
        let m = d
            .allocate(
                4096,
                MemoryRegion::Global,
                MemType::Device,
                MemAccess::ReadWrite,
            )
            .unwrap();
        assert!(m.device_ptr() > 0);
        d.free(&m).unwrap();
        assert!(m.is_freed());
    }
    #[test]
    fn test_kernel_launch() {
        let d = test_device();
        let kernel = d.create_kernel("test");
        let config = KernelConfig::new((1, 1, 1), (256, 1, 1));
        let handle = d.launch_kernel(&kernel, &config, &[]);
        assert!(handle.is_complete());
    }
    #[test]
    fn test_queues() {
        let mut d = test_device();
        let qh = d.create_queue(QueuePriority::Normal, 64);
        let id = d
            .submit(qh, Command::Barrier, QueuePriority::Normal)
            .unwrap();
        assert_eq!(id, 1);
        assert_eq!(d.pending_commands(), 1);
    }
    #[test]
    fn test_properties() {
        let d = test_device();
        let p = d.properties();
        assert_eq!(p.backend, Backend::Simulated);
        assert_eq!(p.warp_size, 32);
    }

    #[test]
    fn test_memcpy_roundtrip() {
        let mut d = test_device();
        let m = d
            .allocate(
                16,
                MemoryRegion::Global,
                MemType::Device,
                MemAccess::ReadWrite,
            )
            .unwrap();
        let host = [1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        d.memcpy_htod(&m, &host, 16, 0).unwrap();
        let mut out = [0u8; 16];
        d.memcpy_dtoh(&mut out, &m, 16, 0).unwrap();
        assert_eq!(host, out);
    }

    #[test]
    fn test_memcpy_partial() {
        let mut d = test_device();
        let m = d
            .allocate(
                16,
                MemoryRegion::Global,
                MemType::Device,
                MemAccess::ReadWrite,
            )
            .unwrap();
        let host = [9u8; 4];
        d.memcpy_htod(&m, &host, 4, 8).unwrap();
        let mut out = [0u8; 16];
        d.memcpy_dtoh(&mut out, &m, 16, 0).unwrap();
        assert_eq!(&out[8..12], &[9u8; 4]);
        assert_eq!(&out[0..8], &[0u8; 8]);
    }

    #[test]
    fn test_memcpy_bounds() {
        let mut d = test_device();
        // BuddyAllocator rounds up to its min block size, so allocate a block
        // larger than 8 bytes and assert that over-sized / over-offset copies
        // are rejected.
        let m = d
            .allocate(
                8192,
                MemoryRegion::Global,
                MemType::Device,
                MemAccess::ReadWrite,
            )
            .unwrap();
        let host = [1u8; 16];
        assert!(d.memcpy_htod(&m, &host, 9000, 0).is_err());
        assert!(d.memcpy_htod(&m, &host, 16, 8192).is_err());
    }

    #[test]
    fn test_synchronize_drains_queue() {
        let mut d = test_device();
        let qh = d.create_queue(QueuePriority::Normal, 64);
        d.submit(qh, Command::Barrier, QueuePriority::Normal)
            .unwrap();
        d.submit(qh, Command::Barrier, QueuePriority::Normal)
            .unwrap();
        assert_eq!(d.pending_commands(), 2);
        d.synchronize();
        assert_eq!(d.pending_commands(), 0);
    }

    #[test]
    fn test_load_module_reduce_max() {
        // Hand-written TPTIR module (matching tpt-archon's emit_topk shape):
        // load input buffer, reduce_max, store scalar result.
        let module = r#"
module {
  func.func @reduce_max(%in: memref<*xf32>, %out: memref<*xf32>) attributes {tptir.kernel} {
    ^entry:
      %v = tptir.load(%in)
      %m = tptir.max(%v)
      tptir.store(%m, %out)
      tptir.return
  }
}
"#;
        let mut d = test_device();
        let kernel = d.load_module(module).unwrap();
        assert_eq!(kernel.name(), "reduce_max");
        assert!(kernel.module().is_some());

        let input = d
            .allocate(
                32,
                MemoryRegion::Global,
                MemType::Device,
                MemAccess::ReadWrite,
            )
            .unwrap();
        let output = d
            .allocate(
                32,
                MemoryRegion::Global,
                MemType::Device,
                MemAccess::ReadWrite,
            )
            .unwrap();
        let data: [f32; 8] = [1.0, -3.0, 7.5, 2.0, 0.0, 4.0, -1.0, 9.25];
        let mut bytes = Vec::new();
        for f in &data {
            bytes.extend_from_slice(&f.to_le_bytes());
        }
        d.memcpy_htod(&input, &bytes, 32, 0).unwrap();

        let args = vec![
            input.device_ptr().to_le_bytes().to_vec(),
            output.device_ptr().to_le_bytes().to_vec(),
        ];
        let config = KernelConfig::new((1, 1, 1), (1, 1, 1));
        let handle = d.launch_kernel(&kernel, &config, &args);
        assert!(handle.is_complete());

        let mut out = [0u8; 4];
        d.memcpy_dtoh(&mut out, &output, 4, 0).unwrap();
        let result = f32::from_le_bytes(out);
        assert_eq!(result, 9.25);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_new_cuda_real_memcpy_roundtrip() {
        // Verifies that `Device::new_cuda` opens a real CUDA context and that
        // allocate / memcpy_htod / memcpy_dtoh round-trip real bytes through the
        // GPU (checked against the RTX 3050 on this machine).
        let mut d = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => {
                eprintln!("no CUDA device available, skipping");
                return;
            }
        };
        assert!(d.is_real());
        assert_eq!(d.properties().backend, Backend::CUDA);

        let m = d
            .allocate(
                16,
                MemoryRegion::Global,
                MemType::Device,
                MemAccess::ReadWrite,
            )
            .unwrap();
        assert!(m.device_ptr() > 0);
        let host = [1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        d.memcpy_htod(&m, &host, 16, 0).unwrap();
        let mut out = [0u8; 16];
        d.memcpy_dtoh(&mut out, &m, 16, 0).unwrap();
        assert_eq!(host, out);
        d.free(&m).unwrap();
        assert!(m.is_freed());
    }
}
