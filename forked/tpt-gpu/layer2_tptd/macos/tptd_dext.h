/*============================================================================
 * tptd_dext.h — TPT GPU macOS DriverKit Extension
 *============================================================================
 * TPT GPU — Tensor Processing Technology
 * License: Apache License 2.0 (with Express Patent Grant)
 *
 * Declares IOService subclass (TPTDriver) and IOUserClient subclass
 * (TPTUserClient) for the TPT GPU PCIe device on macOS 12+.
 *
 * Build requirements:
 *   - Xcode 14+ with DriverKit SDK
 *   - Entitlement: com.apple.developer.driverkit.transport.pci
 *   - Entitlement: com.apple.developer.driverkit.family.gpu (TBD — request from Apple)
 *============================================================================*/

#pragma once

#include <DriverKit/IOService.h>
#include <DriverKit/IOUserClient.h>
#include <DriverKit/IOBufferMemoryDescriptor.h>
#include <PCIDriverKit/IOPCIDevice.h>

#include "../include/tpt_driver.h"

/*---------------------------------------------------------------------------
 * MMIO accessor helpers (volatile pointer to BAR0 mapping)
 *---------------------------------------------------------------------------*/
#define TPT_DEXT_READ32(base, off) \
    *((volatile uint32_t *)((uint8_t *)(base) + (off)))

#define TPT_DEXT_WRITE32(base, off, val) \
    (*((volatile uint32_t *)((uint8_t *)(base) + (off))) = (val))

/*---------------------------------------------------------------------------
 * UserClient method selectors (map to TPT_IOC_* codes)
 *---------------------------------------------------------------------------*/
typedef enum : uint64_t {
    kTPTMethodGetInfo       = 0,
    kTPTMethodAllocMem      = 1,
    kTPTMethodFreeMem       = 2,
    kTPTMethodMapMem        = 3,
    kTPTMethodUnmapMem      = 4,
    kTPTMethodSubmitCmd     = 5,
    kTPTMethodWaitComplete  = 6,
    kTPTMethodQueryPerf     = 7,
    kTPTMethodResetGpu      = 8,
    kTPTMethodCount         = 9,
} TPTUserClientMethod;

/*---------------------------------------------------------------------------
 * TPTDriver — IOService subclass
 *---------------------------------------------------------------------------*/
class TPTDriver : public IOService
{
    OSDeclareDefaultStructors(TPTDriver);

public:
    /* IOService lifecycle */
    virtual kern_return_t Start(IOService *provider) override;
    virtual kern_return_t Stop(IOService *provider) override;

    /* Called by TPTUserClient to perform MMIO reads/writes */
    uint32_t  ReadReg32(uint32_t offset);
    void      WriteReg32(uint32_t offset, uint32_t value);

    /* VRAM bump allocator */
    kern_return_t AllocVram(uint64_t size, uint64_t *phys_out);

    /* Accessors */
    uint64_t  VramBytes() const { return m_vramBytes; }
    uint32_t  NumSm()     const { return m_numSm; }

private:
    IOPCIDevice         *m_pciDevice   = nullptr;
    IOMemoryMap         *m_bar0Map     = nullptr;  /* BAR0 MMIO mapping */
    void                *m_bar0Va      = nullptr;
    uint64_t             m_vramBytes   = 0;
    uint32_t             m_numSm       = 1;
    uint64_t             m_vramBump    = 0x0010000ULL;

    OSAction            *m_irqAction   = nullptr;

    kern_return_t BootGpu();
    kern_return_t SetupCommandRing();

    /* MSI-X interrupt handler */
    void HandleInterrupt(OSAction *action, uint64_t timestamp);
    IMPL(TPTDriver, HandleInterrupt);
};

/*---------------------------------------------------------------------------
 * TPTUserClient — IOUserClient subclass
 *---------------------------------------------------------------------------*/
class TPTUserClient : public IOUserClient
{
    OSDeclareDefaultStructors(TPTUserClient);

public:
    virtual kern_return_t Start(IOService *provider) override;
    virtual kern_return_t Stop(IOService *provider) override;

    /* IOUserClient method dispatch */
    virtual kern_return_t ExternalMethod(
        uint64_t             selector,
        IOUserClientMethodArguments *args,
        const IOUserClientMethodDispatch *dispatch,
        OSObject *target,
        void *reference) override;

private:
    TPTDriver *m_driver = nullptr;

    kern_return_t MethodGetInfo(IOUserClientMethodArguments *args);
    kern_return_t MethodAllocMem(IOUserClientMethodArguments *args);
    kern_return_t MethodFreeMem(IOUserClientMethodArguments *args);
    kern_return_t MethodSubmitCmd(IOUserClientMethodArguments *args);
    kern_return_t MethodWaitComplete(IOUserClientMethodArguments *args);
    kern_return_t MethodQueryPerf(IOUserClientMethodArguments *args);
    kern_return_t MethodResetGpu(IOUserClientMethodArguments *args);
};
