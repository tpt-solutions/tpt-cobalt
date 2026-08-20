use alloc::string::String;

/// A namespaced operation name of the form `"dialect.op"`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct OpName {
    pub dialect: String,
    pub op: String,
}

impl OpName {
    pub fn new(dialect: impl Into<String>, op: impl Into<String>) -> Self {
        OpName {
            dialect: dialect.into(),
            op: op.into(),
        }
    }

    /// Parse a `"dialect.op"` string into an [`OpName`].
    ///
    /// Returns `Err` if there is no single `.` separator, or if either side is
    /// empty.
    pub fn parse(s: &str) -> Result<Self, &'static str> {
        match s.split_once('.') {
            Some((d, o)) if !d.is_empty() && !o.is_empty() => Ok(OpName::new(d, o)),
            _ => Err("invalid OpName: expected \"dialect.op\""),
        }
    }
}

impl core::fmt::Display for OpName {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}.{}", self.dialect, self.op)
    }
}

// Core dialect
pub const CORE_ADD: &str = "core.add";
pub const CORE_MUL: &str = "core.mul";
pub const CORE_LOAD: &str = "core.load";
pub const CORE_STORE: &str = "core.store";

// GPU dialect
pub const TPT_GPU_LAUNCH: &str = "tpt_gpu.launch";
pub const TPT_GPU_THREAD_ID: &str = "tpt_gpu.thread_id";
pub const TPT_GPU_BARRIER: &str = "tpt_gpu.barrier";
pub const TPT_GPU_ALLOC: &str = "tpt_gpu.alloc";
pub const TPT_GPU_DEALLOC: &str = "tpt_gpu.dealloc";

// Crucible dialect
pub const TPT_CRUCIBLE_MAP_FLASH: &str = "tpt_crucible.map_flash";
pub const TPT_CRUCIBLE_ROUTE_FPGA: &str = "tpt_crucible.route_fpga";
pub const TPT_CRUCIBLE_ANALOG_CONV: &str = "tpt_crucible.analog_conv";

// Memory dialect
pub const TPT_MEMORY_SCOPE_BEGIN: &str = "tpt_memory.scope_begin";
pub const TPT_MEMORY_ALLOC: &str = "tpt_memory.alloc";
pub const TPT_MEMORY_SCOPE_END: &str = "tpt_memory.scope_end";
