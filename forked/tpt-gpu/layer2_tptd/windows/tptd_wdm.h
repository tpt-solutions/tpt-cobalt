/*============================================================================
 * tptd_wdm.h — TPT GPU Windows WDM Driver — Internal Types
 *============================================================================
 * TPT GPU — Tensor Processing Technology
 * License: Apache License 2.0 (with Express Patent Grant)
 *
 * Internal driver types and forward declarations.
 * Public ABI is in ../include/tpt_driver.h
 *============================================================================*/

#pragma once

#include <ntddk.h>
#include <wdm.h>
#include <wdmguid.h>
#include <initguid.h>

#include "../include/tpt_driver.h"

/*---------------------------------------------------------------------------
 * IOCTL control codes (METHOD_BUFFERED)
 *---------------------------------------------------------------------------*/
#define IOCTL_TPT_GET_INFO     CTL_CODE(FILE_DEVICE_UNKNOWN, 0x801, METHOD_BUFFERED, FILE_ANY_ACCESS)
#define IOCTL_TPT_ALLOC_MEM    CTL_CODE(FILE_DEVICE_UNKNOWN, 0x802, METHOD_BUFFERED, FILE_ANY_ACCESS)
#define IOCTL_TPT_FREE_MEM     CTL_CODE(FILE_DEVICE_UNKNOWN, 0x803, METHOD_BUFFERED, FILE_ANY_ACCESS)
#define IOCTL_TPT_MAP_MEM      CTL_CODE(FILE_DEVICE_UNKNOWN, 0x804, METHOD_BUFFERED, FILE_ANY_ACCESS)
#define IOCTL_TPT_UNMAP_MEM    CTL_CODE(FILE_DEVICE_UNKNOWN, 0x805, METHOD_BUFFERED, FILE_ANY_ACCESS)
#define IOCTL_TPT_SUBMIT_CMD   CTL_CODE(FILE_DEVICE_UNKNOWN, 0x806, METHOD_BUFFERED, FILE_ANY_ACCESS)
#define IOCTL_TPT_WAIT         CTL_CODE(FILE_DEVICE_UNKNOWN, 0x807, METHOD_BUFFERED, FILE_ANY_ACCESS)
#define IOCTL_TPT_QUERY_PERF   CTL_CODE(FILE_DEVICE_UNKNOWN, 0x808, METHOD_BUFFERED, FILE_ANY_ACCESS)
#define IOCTL_TPT_RESET_GPU    CTL_CODE(FILE_DEVICE_UNKNOWN, 0x809, METHOD_BUFFERED, FILE_READ_ACCESS)

/*---------------------------------------------------------------------------
 * Device interface GUID
 *---------------------------------------------------------------------------*/
// {4A7B9F3C-1234-5678-ABCD-EF0123456789}
DEFINE_GUID(GUID_TPT_DEVICE_INTERFACE,
    0x4a7b9f3c, 0x1234, 0x5678,
    0xab, 0xcd, 0xef, 0x01, 0x23, 0x45, 0x67, 0x89);

/*---------------------------------------------------------------------------
 * VRAM buffer tracking
 *---------------------------------------------------------------------------*/
#define TPT_MAX_BUFFERS 4096

typedef struct _TPT_BUFFER {
    LIST_ENTRY   ListEntry;
    UINT64       PhysAddr;       /* device-physical VRAM address */
    UINT64       SizeBytes;
    UINT32       Flags;
    UINT64       Handle;         /* opaque handle returned to caller */
    PMDL         pMdl;           /* optional MDL for CPU-accessible buffers */
    PVOID        pVa;            /* kernel VA if mapped */
} TPT_BUFFER, *PTPT_BUFFER;

/*---------------------------------------------------------------------------
 * Per-device extension (DeviceExtension)
 *---------------------------------------------------------------------------*/
typedef struct _TPT_DEVICE_EXT {
    /* PCIe resources */
    PHYSICAL_ADDRESS     Bar0PhysAddr;
    ULONG                Bar0Length;
    PVOID                Bar0Va;           /* mapped BAR0 MMIO */

    PHYSICAL_ADDRESS     Bar1PhysAddr;
    ULONGLONG            Bar1Length;       /* VRAM aperture */

    /* Interrupt */
    PKINTERRUPT          Interrupt;
    ULONG                InterruptVector;
    KIRQL                InterruptIrql;
    KAFFINITY            InterruptAffinity;

    /* DPC for deferred IRQ work */
    KDPC                 CompletionDpc;

    /* Command ring (physically contiguous, DMA-mapped) */
    PHYSICAL_ADDRESS     CmdRingPhys;
    PVOID                CmdRingVa;
    ULONG                CmdRingHead;

    /* Completion tracking */
    UINT64               SeqNoIssued;
    UINT64               SeqNoCompleted;
    KEVENT               CompletionEvent;

    /* VRAM allocator (bump pointer for bring-up) */
    UINT64               VramBump;
    KSPIN_LOCK           VramLock;

    /* Buffer list */
    LIST_ENTRY           BufferList;
    KSPIN_LOCK           BufferLock;
    LOOKASIDE_LIST_EX    BufferPool;

    /* Device state */
    BOOLEAN              DeviceReady;
    UINT32               NumSm;
    UINT64               VramBytes;
} TPT_DEVICE_EXT, *PTPT_DEVICE_EXT;

/*---------------------------------------------------------------------------
 * MMIO accessor macros (BAR0)
 *---------------------------------------------------------------------------*/
#define TPT_READ32(ext, off) \
    READ_REGISTER_ULONG((PULONG)((PUCHAR)(ext)->Bar0Va + (off)))

#define TPT_WRITE32(ext, off, val) \
    WRITE_REGISTER_ULONG((PULONG)((PUCHAR)(ext)->Bar0Va + (off)), (val))

/*---------------------------------------------------------------------------
 * Function prototypes
 *---------------------------------------------------------------------------*/
DRIVER_INITIALIZE         DriverEntry;
DRIVER_ADD_DEVICE         TptAddDevice;
DRIVER_UNLOAD             TptUnload;

__drv_dispatchType(IRP_MJ_PNP)
DRIVER_DISPATCH           TptDispatchPnp;

__drv_dispatchType(IRP_MJ_POWER)
DRIVER_DISPATCH           TptDispatchPower;

__drv_dispatchType(IRP_MJ_CREATE)
__drv_dispatchType(IRP_MJ_CLOSE)
DRIVER_DISPATCH           TptDispatchCreateClose;

__drv_dispatchType(IRP_MJ_DEVICE_CONTROL)
DRIVER_DISPATCH           TptDispatchIoctl;

KSERVICE_ROUTINE          TptInterruptHandler;

IO_DPC_ROUTINE            TptCompletionDpc;

NTSTATUS TptStartDevice(PDEVICE_OBJECT pDev, PCM_RESOURCE_LIST pResources);
VOID     TptStopDevice(PDEVICE_OBJECT pDev);

NTSTATUS TptIoctlGetInfo(PTPT_DEVICE_EXT pExt, PIRP pIrp, PIO_STACK_LOCATION pSl);
NTSTATUS TptIoctlAllocMem(PTPT_DEVICE_EXT pExt, PIRP pIrp, PIO_STACK_LOCATION pSl);
NTSTATUS TptIoctlFreeMem(PTPT_DEVICE_EXT pExt, PIRP pIrp, PIO_STACK_LOCATION pSl);
NTSTATUS TptIoctlSubmitCmd(PTPT_DEVICE_EXT pExt, PIRP pIrp, PIO_STACK_LOCATION pSl);
NTSTATUS TptIoctlWait(PTPT_DEVICE_EXT pExt, PIRP pIrp, PIO_STACK_LOCATION pSl);
NTSTATUS TptIoctlQueryPerf(PTPT_DEVICE_EXT pExt, PIRP pIrp, PIO_STACK_LOCATION pSl);
NTSTATUS TptIoctlResetGpu(PTPT_DEVICE_EXT pExt, PIRP pIrp, PIO_STACK_LOCATION pSl);
