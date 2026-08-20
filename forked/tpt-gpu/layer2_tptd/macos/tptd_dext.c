/*============================================================================
 * tptd_dext.c — TPT GPU macOS DriverKit Extension Implementation
 *============================================================================
 * TPT GPU — Tensor Processing Technology
 * License: Apache License 2.0 (with Express Patent Grant)
 *
 * Implements TPTDriver (IOService) and TPTUserClient for the TPT GPU
 * PCIe device on macOS 12+ (Monterey) using the DriverKit framework.
 *
 * DriverKit extensions run in user space — all hardware access goes through
 * IOKit's memory descriptor and PCI device APIs.
 *============================================================================*/

#include "tptd_dext.h"
#include <DriverKit/OSSynchronize.h>
#include <DriverKit/IODispatchQueue.h>
#include <DriverKit/IOTimerDispatchSource.h>
#include <os/log.h>

#define LOG(fmt, ...) os_log(OS_LOG_DEFAULT, "tptd: " fmt, ##__VA_ARGS__)

/* Command ring constants */
#define CMDRING_ENTRIES  256u
#define CMDRING_SIZE     (CMDRING_ENTRIES * 64u)

/* ============================================================================
 * TPTDriver implementation
 * ========================================================================= */
#define super IOService
OSDefineMetaClassAndStructors(TPTDriver, IOService)

kern_return_t TPTDriver::Start(IOService *provider)
{
    kern_return_t ret;

    ret = super::Start(provider);
    if (ret != kIOReturnSuccess) return ret;

    m_pciDevice = OSDynamicCast(IOPCIDevice, provider);
    if (!m_pciDevice) {
        LOG("provider is not an IOPCIDevice");
        return kIOReturnNoDevice;
    }
    m_pciDevice->retain();

    /* Set bus mastering */
    m_pciDevice->SetBusMasterEnable(true);
    m_pciDevice->SetMemoryEnable(true);

    /* Map BAR0 (4 KiB MMIO) */
    ret = m_pciDevice->MapDeviceMemoryWithIndex(0, &m_bar0Map);
    if (ret != kIOReturnSuccess || !m_bar0Map) {
        LOG("Failed to map BAR0: %d", ret);
        return ret;
    }
    m_bar0Va = (void *)(uintptr_t)m_bar0Map->GetVirtualAddress();

    /* Read hardware info */
    uint32_t ver    = ReadReg32(TPT_REG_VERSION);
    uint32_t vram_lo = ReadReg32(TPT_REG_VRAM_LO);
    uint32_t vram_hi = ReadReg32(TPT_REG_VRAM_HI);
    m_vramBytes = ((uint64_t)vram_hi << 32) | vram_lo;

    LOG("TPT GPU version %u.%u, VRAM %llu MiB",
        ver >> 16, ver & 0xFFFF, m_vramBytes >> 20);

    ret = SetupCommandRing();
    if (ret != kIOReturnSuccess) return ret;

    ret = BootGpu();
    if (ret != kIOReturnSuccess) return ret;

    /* Register MSI-X interrupt (vector 0) */
    // ret = m_pciDevice->RegisterInterrupts(kIOInterruptTypeMessage, 1, ...);

    RegisterService();
    LOG("TPT GPU started");
    return kIOReturnSuccess;
}

kern_return_t TPTDriver::Stop(IOService *provider)
{
    WriteReg32(TPT_REG_SCHED_EN, 0);
    WriteReg32(TPT_REG_IRQ_MASK, 0);
    WriteReg32(TPT_REG_CTRL, 0);

    if (m_bar0Map) {
        m_bar0Map->release();
        m_bar0Map = nullptr;
    }
    if (m_pciDevice) {
        m_pciDevice->release();
        m_pciDevice = nullptr;
    }
    return super::Stop(provider);
}

kern_return_t TPTDriver::BootGpu()
{
    WriteReg32(TPT_REG_CTRL, TPT_CTRL_BOOT);

    /* Poll READY (10 ms) */
    for (int i = 0; i < 100; i++) {
        if (ReadReg32(TPT_REG_STATUS) & TPT_STATUS_READY) {
            WriteReg32(TPT_REG_IRQ_MASK, 0xFF);
            WriteReg32(TPT_REG_CTRL, TPT_CTRL_BOOT | TPT_CTRL_IRQ_EN);
            WriteReg32(TPT_REG_SCHED_EN, 1);
            return kIOReturnSuccess;
        }
        IOSleep(100);  /* 100 µs */
    }
    LOG("Boot timeout");
    return kIOReturnTimeout;
}

kern_return_t TPTDriver::SetupCommandRing()
{
    IOBufferMemoryDescriptor *ringBuf = nullptr;
    kern_return_t ret;

    ret = IOBufferMemoryDescriptor::Create(
              kIOMemoryDirectionInOut, CMDRING_SIZE, 0x1000, &ringBuf);
    if (ret != kIOReturnSuccess || !ringBuf) return kIOReturnNoMemory;

    /* Zero the ring */
    void *va = ringBuf->GetAddressRange().address;
    if (va) __builtin_memset(va, 0, CMDRING_SIZE);

    /* Get physical address for DMA */
    IODMACommandSpecification dmaSpec = {};
    dmaSpec.options      = kIODMACommandCreateOptionConcurrent;
    dmaSpec.maxAddressBits = 40;

    IODMACommand *dmaCmd = nullptr;
    ringBuf->CreateWithOptions(0, nullptr, nullptr, &dmaCmd);
    // In a real dext: use IODMACommand to get segments

    /* Program ring (simplified — use GetPhysicalAddress for contig allocation) */
    uint64_t phys = ringBuf->GetPhysicalAddressRange().address;
    WriteReg32(TPT_REG_CMDRING_LO,  (uint32_t)(phys & 0xFFFFFFFF));
    WriteReg32(TPT_REG_CMDRING_HI,  (uint32_t)(phys >> 32));
    WriteReg32(TPT_REG_CMDRING_CAP, CMDRING_ENTRIES);
    WriteReg32(TPT_REG_CMDRING_HEAD, 0);

    if (dmaCmd) dmaCmd->release();
    ringBuf->release();
    return kIOReturnSuccess;
}

uint32_t TPTDriver::ReadReg32(uint32_t offset)
{
    return TPT_DEXT_READ32(m_bar0Va, offset);
}

void TPTDriver::WriteReg32(uint32_t offset, uint32_t value)
{
    TPT_DEXT_WRITE32(m_bar0Va, offset, value);
}

kern_return_t TPTDriver::AllocVram(uint64_t size, uint64_t *phys_out)
{
    static const uint64_t ALIGN = 0x100000ULL;
    uint64_t aligned = (size + ALIGN - 1) & ~(ALIGN - 1);

    /* Simple bump allocator (replace with buddy allocator for production) */
    *phys_out = m_vramBump;
    m_vramBump += aligned;
    return kIOReturnSuccess;
}

/* ============================================================================
 * TPTUserClient implementation
 * ========================================================================= */
#undef super
#define super IOUserClient
OSDefineMetaClassAndStructors(TPTUserClient, IOUserClient)

kern_return_t TPTUserClient::Start(IOService *provider)
{
    kern_return_t ret = super::Start(provider);
    if (ret != kIOReturnSuccess) return ret;

    m_driver = OSDynamicCast(TPTDriver, provider);
    if (!m_driver) return kIOReturnNoDevice;
    m_driver->retain();

    return kIOReturnSuccess;
}

kern_return_t TPTUserClient::Stop(IOService *provider)
{
    if (m_driver) { m_driver->release(); m_driver = nullptr; }
    return super::Stop(provider);
}

kern_return_t TPTUserClient::ExternalMethod(
    uint64_t selector,
    IOUserClientMethodArguments *args,
    const IOUserClientMethodDispatch *dispatch,
    OSObject *target,
    void *reference)
{
    (void)dispatch; (void)target; (void)reference;

    switch ((TPTUserClientMethod)selector) {
    case kTPTMethodGetInfo:      return MethodGetInfo(args);
    case kTPTMethodAllocMem:     return MethodAllocMem(args);
    case kTPTMethodFreeMem:      return MethodFreeMem(args);
    case kTPTMethodSubmitCmd:    return MethodSubmitCmd(args);
    case kTPTMethodWaitComplete: return MethodWaitComplete(args);
    case kTPTMethodQueryPerf:    return MethodQueryPerf(args);
    case kTPTMethodResetGpu:     return MethodResetGpu(args);
    default:
        return kIOReturnUnsupported;
    }
}

kern_return_t TPTUserClient::MethodGetInfo(IOUserClientMethodArguments *args)
{
    if (!args->structureOutput || args->structureOutputSize < sizeof(tpt_info_t))
        return kIOReturnBadArgument;

    tpt_info_t *out = (tpt_info_t *)args->structureOutput->getBytesNoCopy();
    uint32_t ver = m_driver->ReadReg32(TPT_REG_VERSION);

    __builtin_memset(out, 0, sizeof(*out));
    out->version_major    = ver >> 16;
    out->version_minor    = ver & 0xFFFF;
    out->vram_bytes       = m_driver->VramBytes();
    out->num_sm           = m_driver->NumSm();
    out->num_warps_per_sm = 64;
    out->warp_lanes       = 32;
    out->num_ctas         = 16;
    out->caps             = TPT_CAP_TENSOR | TPT_CAP_FP64;
    args->structureOutputSize = sizeof(*out);
    return kIOReturnSuccess;
}

kern_return_t TPTUserClient::MethodAllocMem(IOUserClientMethodArguments *args)
{
    if (!args->structureInput || args->structureInputSize < sizeof(tpt_alloc_mem_t))
        return kIOReturnBadArgument;
    if (!args->structureOutput || args->structureOutputSize < sizeof(tpt_alloc_mem_t))
        return kIOReturnBadArgument;

    const tpt_alloc_mem_t *in  = (const tpt_alloc_mem_t *)args->structureInput->getBytesNoCopy();
    tpt_alloc_mem_t *out = (tpt_alloc_mem_t *)args->structureOutput->getBytesNoCopy();
    __builtin_memcpy(out, in, sizeof(*in));

    uint64_t phys = 0;
    kern_return_t ret = m_driver->AllocVram(in->size_bytes, &phys);
    if (ret != kIOReturnSuccess) return ret;

    out->phys_addr = phys;
    out->handle    = phys;
    args->structureOutputSize = sizeof(*out);
    return kIOReturnSuccess;
}

kern_return_t TPTUserClient::MethodFreeMem(IOUserClientMethodArguments *args)
{
    /* VRAM bump allocator has no free — production would use a real allocator */
    (void)args;
    return kIOReturnSuccess;
}

kern_return_t TPTUserClient::MethodSubmitCmd(IOUserClientMethodArguments *args)
{
    if (!args->structureInput  || args->structureInputSize  < sizeof(tpt_submit_cmd_t) ||
        !args->structureOutput || args->structureOutputSize < sizeof(tpt_submit_cmd_t))
        return kIOReturnBadArgument;

    const tpt_submit_cmd_t *in = (const tpt_submit_cmd_t *)args->structureInput->getBytesNoCopy();
    tpt_submit_cmd_t *out = (tpt_submit_cmd_t *)args->structureOutput->getBytesNoCopy();
    __builtin_memcpy(out, in, sizeof(*in));

    /* Advance command ring head */
    uint32_t head = m_driver->ReadReg32(TPT_REG_CMDRING_HEAD);
    m_driver->WriteReg32(TPT_REG_CMDRING_HEAD, (head + 1) % CMDRING_ENTRIES);

    static uint64_t s_seqNo = 0;
    out->seq_no = ++s_seqNo;
    args->structureOutputSize = sizeof(*out);
    return kIOReturnSuccess;
}

kern_return_t TPTUserClient::MethodWaitComplete(IOUserClientMethodArguments *args)
{
    if (!args->structureInput  || args->structureInputSize  < sizeof(tpt_wait_complete_t) ||
        !args->structureOutput || args->structureOutputSize < sizeof(tpt_wait_complete_t))
        return kIOReturnBadArgument;

    const tpt_wait_complete_t *in = (const tpt_wait_complete_t *)args->structureInput->getBytesNoCopy();
    tpt_wait_complete_t *out = (tpt_wait_complete_t *)args->structureOutput->getBytesNoCopy();
    __builtin_memcpy(out, in, sizeof(*in));

    /* Poll completion (simplified — production uses IODispatchSource + notification) */
    uint32_t elapsed = 0;
    uint32_t timeout = in->timeout_ms ? in->timeout_ms : 5000;
    while (elapsed < timeout) {
        uint32_t status = m_driver->ReadReg32(TPT_REG_STATUS);
        if (status & TPT_STATUS_IDLE) {
            out->status = TPT_WAIT_OK;
            args->structureOutputSize = sizeof(*out);
            return kIOReturnSuccess;
        }
        IOSleep(1);
        elapsed++;
    }
    out->status = TPT_WAIT_TIMEOUT;
    args->structureOutputSize = sizeof(*out);
    return kIOReturnSuccess;
}

kern_return_t TPTUserClient::MethodQueryPerf(IOUserClientMethodArguments *args)
{
    if (!args->structureOutput || args->structureOutputSize < sizeof(tpt_perf_counters_t))
        return kIOReturnBadArgument;

    tpt_perf_counters_t *out = (tpt_perf_counters_t *)args->structureOutput->getBytesNoCopy();
    out->inst_retired = ((uint64_t)m_driver->ReadReg32(TPT_REG_PERF_INST_HI) << 32) |
                         m_driver->ReadReg32(TPT_REG_PERF_INST_LO);
    out->core_cycles  = ((uint64_t)m_driver->ReadReg32(TPT_REG_PERF_CYCL_HI) << 32) |
                         m_driver->ReadReg32(TPT_REG_PERF_CYCL_LO);
    out->l1d_misses   = m_driver->ReadReg32(TPT_REG_PERF_L1D_MISS);
    out->l2_misses    = m_driver->ReadReg32(TPT_REG_PERF_L2_MISS);
    out->branch_mispred = 0;
    out->warp_stalls    = 0;
    args->structureOutputSize = sizeof(*out);
    return kIOReturnSuccess;
}

kern_return_t TPTUserClient::MethodResetGpu(IOUserClientMethodArguments *args)
{
    (void)args;
    m_driver->WriteReg32(TPT_REG_CTRL, TPT_CTRL_RESET);
    IOSleep(10);
    m_driver->WriteReg32(TPT_REG_CTRL, TPT_CTRL_BOOT | TPT_CTRL_IRQ_EN);
    return kIOReturnSuccess;
}
