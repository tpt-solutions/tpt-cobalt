// fault.rs — GPU page fault handling and context recovery

use std::sync::Arc;

use anyhow::Result;
use tracing::{error, warn};

use crate::context::GpuContext;
use crate::mmio::{ctrl_bits, regs, Mmio};

/// Handle a GPU page fault.
/// In a full implementation this would:
///  1. Read the fault address from TPT_REG_FAULT_ADDR
///  2. Check the context's page table for a valid-but-unmapped range
///  3. Demand-page if valid; kill context if invalid
pub fn handle_page_fault(mmio: &Arc<Mmio>, ctx: &Arc<GpuContext>) -> Result<()> {
    // Fault address register (not yet defined in RTL — placeholder offset)
    let fault_lo = mmio.read32(0x200) as u64;
    let fault_hi = mmio.read32(0x204) as u64;
    let fault_va = (fault_hi << 32) | fault_lo;

    error!("GPU page fault at VA={fault_va:#012x}");

    // Tear down the faulting context
    ctx.teardown();

    // Attempt GPU recovery: reset and re-boot
    recover_gpu(mmio)?;

    warn!("GPU recovered after page fault (context killed)");
    Ok(())
}

/// Recover the GPU after a fault or watchdog event.
pub fn recover_gpu(mmio: &Arc<Mmio>) -> Result<()> {
    use crate::mmio::status_bits;

    mmio.write32(regs::SCHED_EN, 0);
    mmio.write32(regs::IRQ_MASK, 0);
    mmio.write32(regs::CTRL, ctrl_bits::RESET);

    std::thread::sleep(std::time::Duration::from_millis(10));

    mmio.write32(regs::CTRL, ctrl_bits::BOOT);

    let deadline = std::time::Instant::now() + std::time::Duration::from_millis(100);
    while std::time::Instant::now() < deadline {
        if mmio.read32(regs::STATUS) & status_bits::READY != 0 {
            mmio.write32(regs::IRQ_MASK, 0xFF);
            mmio.write32(regs::CTRL, ctrl_bits::BOOT | ctrl_bits::IRQ_EN);
            mmio.write32(regs::SCHED_EN, 1);
            return Ok(());
        }
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    anyhow::bail!("GPU recovery timeout")
}
