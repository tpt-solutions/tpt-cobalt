//! TPT Runtime - Python Bindings (PyO3)
use pyo3::exceptions::{PyMemoryError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::sync::Mutex;
use tpt_gpu_runtime::{
    Device as CoreDevice, DeviceProperties, ErrorCode, Kernel, KernelConfig, KernelHandle,
    MemAccess, MemType, MemoryAllocation, MemoryRegion, QueuePriority, TptrError,
};

#[pymodule]
fn tptr(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDevice>()?;
    m.add_class::<PyMemoryAllocation>()?;
    m.add_class::<PyQueue>()?;
    m.add_class::<PyKernel>()?;
    m.add_class::<PyKernelConfig>()?;
    m.add_class::<PyKernelHandle>()?;
    m.add_class::<PyTptrError>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}

#[pyclass(name = "TptrError", extends = PyRuntimeError, from_py_object)]
#[derive(Debug, Clone)]
pub struct PyTptrError {
    #[pyo3(get)]
    code: String,
    #[pyo3(get)]
    message: String,
    #[pyo3(get)]
    source: String,
    #[pyo3(get)]
    context: Vec<(String, String)>,
}

fn map_err(err: TptrError) -> PyErr {
    match err.code {
        ErrorCode::AllocationFailure | ErrorCode::OutOfMemory => {
            PyMemoryError::new_err(format!("{}", err))
        }
        ErrorCode::InvalidKernel
        | ErrorCode::LaunchFailure
        | ErrorCode::ArgumentMismatch
        | ErrorCode::ConfigurationError => PyValueError::new_err(format!("{}", err)),
        _ => PyRuntimeError::new_err(format!("{}", err)),
    }
}

#[pyclass(name = "Device")]
pub struct PyDevice {
    inner: Mutex<CoreDevice>,
}

#[pymethods]
impl PyDevice {
    #[new]
    fn new(index: u32) -> PyResult<Self> {
        let props = DeviceProperties::simulated(&format!("TPT Device {}", index), 16 << 30);
        Ok(Self {
            inner: Mutex::new(CoreDevice::new_simulated(index as u64, props)),
        })
    }
    #[staticmethod]
    fn get_default() -> PyResult<Self> {
        Self::new(0)
    }
    #[staticmethod]
    fn enumerate() -> PyResult<Vec<String>> {
        Ok(vec!["TPT Device 0 (Simulated)".to_string()])
    }
    fn allocate(
        &self,
        size: u64,
        mem_type: Option<&str>,
        access: Option<&str>,
    ) -> PyResult<PyMemoryAllocation> {
        let mt = match mem_type.unwrap_or("device") {
            "host_pinned" => MemType::HostPinned,
            "managed" => MemType::Managed,
            _ => MemType::Device,
        };
        let acc = match access.unwrap_or("read_write") {
            "read" => MemAccess::ReadOnly,
            "write" => MemAccess::WriteOnly,
            _ => MemAccess::ReadWrite,
        };
        self.inner
            .lock()
            .unwrap()
            .allocate(size, MemoryRegion::Global, mt, acc)
            .map(PyMemoryAllocation::new)
            .map_err(map_err)
    }
    fn memcpy_htod(
        &self,
        dst: &PyMemoryAllocation,
        src: Vec<u8>,
        size: u64,
        dst_offset: Option<u64>,
    ) -> PyResult<()> {
        self.inner
            .lock()
            .unwrap()
            .memcpy_htod(&dst.inner, &src, size, dst_offset.unwrap_or(0))
            .map_err(map_err)
    }
    fn memcpy_dtoh(
        &self,
        src: &PyMemoryAllocation,
        size: u64,
        src_offset: Option<u64>,
    ) -> PyResult<Vec<u8>> {
        let mut buf = vec![0u8; size as usize];
        self.inner
            .lock()
            .unwrap()
            .memcpy_dtoh(&mut buf, &src.inner, size, src_offset.unwrap_or(0))
            .map_err(map_err)?;
        Ok(buf)
    }
    fn create_queue(&self, priority: Option<&str>) -> PyResult<PyQueue> {
        let pri = match priority.unwrap_or("normal") {
            "high" => QueuePriority::High,
            "low" => QueuePriority::Low,
            _ => QueuePriority::Normal,
        };
        let handle = self.inner.lock().unwrap().create_queue(pri, 1024);
        Ok(PyQueue { handle })
    }
    fn create_kernel(&self, name: &str) -> PyKernel {
        PyKernel {
            inner: self.inner.lock().unwrap().create_kernel(name),
        }
    }
    fn load_module(&self, tptir_text: &str) -> PyResult<PyKernel> {
        self.inner
            .lock()
            .unwrap()
            .load_module(tptir_text)
            .map(|k| PyKernel { inner: k })
            .map_err(map_err)
    }
    fn info(&self) -> std::collections::HashMap<String, String> {
        let p = self.inner.lock().unwrap().properties().clone();
        let mut m = std::collections::HashMap::new();
        m.insert("name".into(), p.name.clone());
        m.insert("total_memory".into(), p.total_memory.to_string());
        m.insert("backend".into(), p.backend.name().to_string());
        m
    }
    fn synchronize(&self) {
        self.inner.lock().unwrap().synchronize();
    }
}

#[pyclass(name = "MemoryAllocation", from_py_object)]
#[derive(Clone)]
pub struct PyMemoryAllocation {
    inner: MemoryAllocation,
}
impl PyMemoryAllocation {
    fn new(inner: MemoryAllocation) -> Self {
        Self { inner }
    }
}
#[pymethods]
impl PyMemoryAllocation {
    #[getter]
    fn handle(&self) -> u64 {
        self.inner.handle()
    }
    #[getter]
    fn size(&self) -> u64 {
        self.inner.size()
    }
    #[getter]
    fn device_ptr(&self) -> u64 {
        self.inner.device_ptr()
    }
    #[getter]
    fn is_freed(&self) -> bool {
        self.inner.is_freed()
    }
    fn __repr__(&self) -> String {
        format!(
            "MemoryAllocation(handle={}, size={}, ptr=0x{:x})",
            self.inner.handle(),
            self.inner.size(),
            self.inner.device_ptr()
        )
    }
}

#[pyclass(name = "CommandQueue")]
pub struct PyQueue {
    handle: tpt_gpu_runtime::command::QueueHandle,
}
#[pymethods]
impl PyQueue {
    #[getter]
    fn handle(&self) -> u64 {
        self.handle.0
    }
}
#[pyclass(name = "Kernel")]
pub struct PyKernel {
    inner: Kernel,
}
#[pymethods]
impl PyKernel {
    #[getter]
    fn name(&self) -> &str {
        self.inner.name()
    }
    fn launch(
        &self,
        device: &PyDevice,
        config: &PyKernelConfig,
        args: Option<Vec<Vec<u8>>>,
    ) -> PyKernelHandle {
        let args = args.unwrap_or_default();
        let handle = device
            .inner
            .lock()
            .unwrap()
            .launch_kernel(&self.inner, &config.inner, &args);
        PyKernelHandle { inner: handle }
    }
}

#[pyclass(name = "KernelConfig", from_py_object)]
#[derive(Clone)]
pub struct PyKernelConfig {
    inner: KernelConfig,
}
#[pymethods]
impl PyKernelConfig {
    #[new]
    #[pyo3(signature = (grid, block, shared_mem = None))]
    fn new(grid: (u32, u32, u32), block: (u32, u32, u32), shared_mem: Option<u32>) -> Self {
        let mut c = KernelConfig::new(grid, block);
        if let Some(sm) = shared_mem {
            c = c.with_shared_mem(sm);
        }
        Self { inner: c }
    }
    #[getter]
    fn grid_size(&self) -> (u32, u32, u32) {
        (
            self.inner.grid_size.x,
            self.inner.grid_size.y,
            self.inner.grid_size.z,
        )
    }
    #[getter]
    fn block_size(&self) -> (u32, u32, u32) {
        (
            self.inner.block_size.x,
            self.inner.block_size.y,
            self.inner.block_size.z,
        )
    }
    #[getter]
    fn shared_mem_bytes(&self) -> u32 {
        self.inner.shared_mem_bytes
    }
}

#[pyclass(name = "KernelHandle")]
pub struct PyKernelHandle {
    inner: KernelHandle,
}
#[pymethods]
impl PyKernelHandle {
    #[getter]
    fn id(&self) -> u64 {
        self.inner.id()
    }
    fn is_complete(&self) -> bool {
        self.inner.is_complete()
    }
    fn wait(&self) {
        self.inner.wait();
    }
}
