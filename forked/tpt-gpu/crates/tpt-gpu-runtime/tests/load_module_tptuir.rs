//! End-to-end test for `Device::load_module_tptuir` (the runtime consumption
//! side of the TPT-UIR ingestion adapter). Builds a kernel region, writes it
//! as a `.tptuir` file via the adapter, then loads it through a simulated
//! device — proving the adapter is wired into `tpt-gpu-runtime`.

use tpt_gpu_runtime::device::{Device, DeviceProperties};

#[test]
fn load_module_tptuir_loads_kernel() {
    let region = tpt_gpu_compiler::ir::build_kernel_region(
        "matmul",
        tpt_gpu_compiler::ir::ElemType::F32,
        &[256],
    )
    .unwrap();

    let path = std::env::temp_dir().join(format!("rt_load_{}.tptuir", std::process::id()));
    tpt_gpu_uir_adapter::write_tptuir(&region, &path).expect("write_tptuir failed");

    let device = Device::new_simulated(0, DeviceProperties::simulated("test", 1 << 30));
    let _kernel = device
        .load_module_tptuir(&path)
        .expect("load_module_tptuir failed");

    let _ = std::fs::remove_file(&path);
}
