/*============================================================================
 * tptd_wdm.c — TPT GPU Windows WDM Driver
 *============================================================================
 * TPT GPU — Tensor Processing Technology
 * License: Apache License 2.0 (with Express Patent Grant)
 *
 * WDM miniport for the TPT GPU PCIe device.
 * Implements DriverEntry, PnP, Power, and IOCTL dispatch.
 * Build with: WDK 11 + MSVC 2022
 *   msbuild tptd.vcxproj /p:Configuration=Release /p:Platform=x64
 *============================================================================*/

#include "tptd_wdm.h"

#pragma warning(disable: 4100)  /* unreferenced formal parameter */

/* PCIe IDs */
#define TPT_VENDOR_ID  0x1AC7
#define TPT_DEVICE_ID  0x0100

/* Command ring */
#define CMDRING_ENTRIES  256u
#define CMDRING_SIZE     (CMDRING_ENTRIES * 64u)  /* 16 KiB */

/* VRAM bump base (after command ring) */
#define VRAM_ALLOC_BASE  0x0010000ull
#define VRAM_ALIGN       0x0100000ull   /* 1 MiB */

/*============================================================================
 * DriverEntry
 *============================================================================*/
NTSTATUS DriverEntry(PDRIVER_OBJECT pDrvObj, PUNICODE_STRING pRegPath)
{
    UNREFERENCED_PARAMETER(pRegPath);

    pDrvObj->DriverExtension->AddDevice          = TptAddDevice;
    pDrvObj->DriverUnload                        = TptUnload;
    pDrvObj->MajorFunction[IRP_MJ_PNP]           = TptDispatchPnp;
    pDrvObj->MajorFunction[IRP_MJ_POWER]         = TptDispatchPower;
    pDrvObj->MajorFunction[IRP_MJ_CREATE]        = TptDispatchCreateClose;
    pDrvObj->MajorFunction[IRP_MJ_CLOSE]         = TptDispatchCreateClose;
    pDrvObj->MajorFunction[IRP_MJ_DEVICE_CONTROL] = TptDispatchIoctl;

    return STATUS_SUCCESS;
}

VOID TptUnload(PDRIVER_OBJECT pDrvObj)
{
    UNREFERENCED_PARAMETER(pDrvObj);
}

/*============================================================================
 * AddDevice — create FDO and attach to PDO stack
 *============================================================================*/
NTSTATUS TptAddDevice(PDRIVER_OBJECT pDrvObj, PDEVICE_OBJECT pPdo)
{
    NTSTATUS       status;
    PDEVICE_OBJECT pFdo;
    PTPT_DEVICE_EXT pExt;

    status = IoCreateDevice(pDrvObj, sizeof(TPT_DEVICE_EXT),
                            NULL, FILE_DEVICE_UNKNOWN,
                            FILE_DEVICE_SECURE_OPEN, FALSE, &pFdo);
    if (!NT_SUCCESS(status)) return status;

    pExt = (PTPT_DEVICE_EXT)pFdo->DeviceExtension;
    RtlZeroMemory(pExt, sizeof(*pExt));

    KeInitializeSpinLock(&pExt->VramLock);
    KeInitializeSpinLock(&pExt->BufferLock);
    InitializeListHead(&pExt->BufferList);
    KeInitializeEvent(&pExt->CompletionEvent, SynchronizationEvent, FALSE);
    KeInitializeDpc(&pExt->CompletionDpc, TptCompletionDpc, pFdo);

    ExInitializeLookasideListEx(&pExt->BufferPool, NULL, NULL,
                                 NonPagedPoolNx, 0,
                                 sizeof(TPT_BUFFER), 'TPTB', 0);

    pExt->VramBump = VRAM_ALLOC_BASE;

    IoAttachDeviceToDeviceStack(pFdo, pPdo);
    pFdo->Flags &= ~DO_DEVICE_INITIALIZING;
    return STATUS_SUCCESS;
}

/*============================================================================
 * PnP dispatch
 *============================================================================*/
NTSTATUS TptDispatchPnp(PDEVICE_OBJECT pDev, PIRP pIrp)
{
    PIO_STACK_LOCATION pSl = IoGetCurrentIrpStackLocation(pIrp);

    switch (pSl->MinorFunction) {
    case IRP_MN_START_DEVICE:
        return TptStartDevice(pDev, pSl->Parameters.StartDevice.AllocatedResourcesTranslated);

    case IRP_MN_STOP_DEVICE:
    case IRP_MN_REMOVE_DEVICE:
        TptStopDevice(pDev);
        pIrp->IoStatus.Status = STATUS_SUCCESS;
        IoCompleteRequest(pIrp, IO_NO_INCREMENT);
        return STATUS_SUCCESS;

    default:
        pIrp->IoStatus.Status = STATUS_SUCCESS;
        IoCompleteRequest(pIrp, IO_NO_INCREMENT);
        return STATUS_SUCCESS;
    }
}

/*============================================================================
 * StartDevice — map BAR0, boot GPU, register IRQ, install command ring
 *============================================================================*/
NTSTATUS TptStartDevice(PDEVICE_OBJECT pDev, PCM_RESOURCE_LIST pResources)
{
    PTPT_DEVICE_EXT pExt = (PTPT_DEVICE_EXT)pDev->DeviceExtension;
    PCM_FULL_RESOURCE_DESCRIPTOR pFull;
    PCM_PARTIAL_RESOURCE_DESCRIPTOR pDesc;
    ULONG i, j;
    NTSTATUS status;
    UINT32 ver, ctrl;

    if (!pResources) return STATUS_INSUFFICIENT_RESOURCES;

    for (i = 0; i < pResources->Count; i++) {
        pFull = &pResources->List[i];
        for (j = 0; j < pFull->PartialResourceList.Count; j++) {
            pDesc = &pFull->PartialResourceList.PartialDescriptors[j];
            if (pDesc->Type == CmResourceTypeMemory) {
                if (pExt->Bar0Va == NULL) {
                    /* First memory range = BAR0 (4 KiB MMIO) */
                    pExt->Bar0PhysAddr = pDesc->u.Memory.Start;
                    pExt->Bar0Length   = pDesc->u.Memory.Length;
                    pExt->Bar0Va = MmMapIoSpaceEx(pExt->Bar0PhysAddr,
                                                   pExt->Bar0Length,
                                                   PAGE_READWRITE | PAGE_NOCACHE);
                    if (!pExt->Bar0Va) return STATUS_INSUFFICIENT_RESOURCES;
                } else {
                    /* Second = BAR1 (VRAM aperture) */
                    pExt->Bar1PhysAddr = pDesc->u.Memory.Start;
                    pExt->Bar1Length   = pDesc->u.Memory.Length;
                }
            } else if (pDesc->Type == CmResourceTypeInterrupt) {
                pExt->InterruptVector   = pDesc->u.Interrupt.Vector;
                pExt->InterruptIrql     = (KIRQL)pDesc->u.Interrupt.Level;
                pExt->InterruptAffinity = pDesc->u.Interrupt.Affinity;
            }
        }
    }

    if (!pExt->Bar0Va) return STATUS_INSUFFICIENT_RESOURCES;

    /* Read hardware version */
    ver = TPT_READ32(pExt, TPT_REG_VERSION);
    pExt->VramBytes = ((UINT64)TPT_READ32(pExt, TPT_REG_VRAM_HI) << 32) |
                       TPT_READ32(pExt, TPT_REG_VRAM_LO);
    pExt->NumSm = 1;

    /* Allocate command ring (physically contiguous) */
    PHYSICAL_ADDRESS lowAddr  = {0};
    PHYSICAL_ADDRESS highAddr = {.QuadPart = 0xFFFFFFFFFFFFFFFFull};
    PHYSICAL_ADDRESS alignAddr = {.QuadPart = 0x1000};
    pExt->CmdRingVa = MmAllocateContiguousMemorySpecifyCache(
                          CMDRING_SIZE, lowAddr, highAddr, alignAddr,
                          MmWriteCombined);
    if (!pExt->CmdRingVa) return STATUS_INSUFFICIENT_RESOURCES;
    RtlZeroMemory(pExt->CmdRingVa, CMDRING_SIZE);
    pExt->CmdRingPhys = MmGetPhysicalAddress(pExt->CmdRingVa);

    /* Program ring into hardware */
    TPT_WRITE32(pExt, TPT_REG_CMDRING_LO,  (UINT32)(pExt->CmdRingPhys.QuadPart & 0xFFFFFFFF));
    TPT_WRITE32(pExt, TPT_REG_CMDRING_HI,  (UINT32)(pExt->CmdRingPhys.QuadPart >> 32));
    TPT_WRITE32(pExt, TPT_REG_CMDRING_CAP, CMDRING_ENTRIES);
    TPT_WRITE32(pExt, TPT_REG_CMDRING_HEAD, 0);

    /* Boot GPU */
    TPT_WRITE32(pExt, TPT_REG_CTRL, TPT_CTRL_BOOT);

    /* Poll READY (10 ms) */
    for (i = 0; i < 100; i++) {
        if (TPT_READ32(pExt, TPT_REG_STATUS) & TPT_STATUS_READY) break;
        KeStallExecutionProcessor(100);  /* 100 µs */
    }
    if (!(TPT_READ32(pExt, TPT_REG_STATUS) & TPT_STATUS_READY))
        return STATUS_DEVICE_NOT_READY;

    /* Register interrupt */
    status = IoConnectInterrupt(&pExt->Interrupt,
                                 TptInterruptHandler,
                                 pDev,
                                 NULL,
                                 pExt->InterruptVector,
                                 pExt->InterruptIrql,
                                 pExt->InterruptIrql,
                                 LevelSensitive,
                                 TRUE,
                                 pExt->InterruptAffinity,
                                 FALSE);
    if (!NT_SUCCESS(status)) return status;

    /* Enable interrupts and scheduler */
    TPT_WRITE32(pExt, TPT_REG_IRQ_MASK,  0xFF);
    TPT_WRITE32(pExt, TPT_REG_CTRL,      TPT_CTRL_BOOT | TPT_CTRL_IRQ_EN);
    TPT_WRITE32(pExt, TPT_REG_SCHED_EN,  1);

    pExt->DeviceReady = TRUE;
    return STATUS_SUCCESS;
}

VOID TptStopDevice(PDEVICE_OBJECT pDev)
{
    PTPT_DEVICE_EXT pExt = (PTPT_DEVICE_EXT)pDev->DeviceExtension;

    pExt->DeviceReady = FALSE;
    TPT_WRITE32(pExt, TPT_REG_SCHED_EN, 0);
    TPT_WRITE32(pExt, TPT_REG_IRQ_MASK, 0);
    TPT_WRITE32(pExt, TPT_REG_CTRL, 0);

    if (pExt->Interrupt) {
        IoDisconnectInterrupt(pExt->Interrupt);
        pExt->Interrupt = NULL;
    }
    if (pExt->CmdRingVa) {
        MmFreeContiguousMemory(pExt->CmdRingVa);
        pExt->CmdRingVa = NULL;
    }
    if (pExt->Bar0Va) {
        MmUnmapIoSpace(pExt->Bar0Va, pExt->Bar0Length);
        pExt->Bar0Va = NULL;
    }
}

/*============================================================================
 * Power dispatch (minimal — pass down)
 *============================================================================*/
NTSTATUS TptDispatchPower(PDEVICE_OBJECT pDev, PIRP pIrp)
{
    PoStartNextPowerIrp(pIrp);
    IoSkipCurrentIrpStackLocation(pIrp);
    return PoCallDriver(IoGetAttachedDeviceReference(pDev), pIrp);
}

/*============================================================================
 * Create/Close
 *============================================================================*/
NTSTATUS TptDispatchCreateClose(PDEVICE_OBJECT pDev, PIRP pIrp)
{
    UNREFERENCED_PARAMETER(pDev);
    pIrp->IoStatus.Status      = STATUS_SUCCESS;
    pIrp->IoStatus.Information = 0;
    IoCompleteRequest(pIrp, IO_NO_INCREMENT);
    return STATUS_SUCCESS;
}

/*============================================================================
 * IOCTL dispatch
 *============================================================================*/
NTSTATUS TptDispatchIoctl(PDEVICE_OBJECT pDev, PIRP pIrp)
{
    PTPT_DEVICE_EXT    pExt = (PTPT_DEVICE_EXT)pDev->DeviceExtension;
    PIO_STACK_LOCATION pSl  = IoGetCurrentIrpStackLocation(pIrp);
    NTSTATUS           status;

    if (!pExt->DeviceReady) {
        pIrp->IoStatus.Status = STATUS_DEVICE_NOT_READY;
        IoCompleteRequest(pIrp, IO_NO_INCREMENT);
        return STATUS_DEVICE_NOT_READY;
    }

    switch (pSl->Parameters.DeviceIoControl.IoControlCode) {
    case IOCTL_TPT_GET_INFO:    status = TptIoctlGetInfo(pExt, pIrp, pSl);    break;
    case IOCTL_TPT_ALLOC_MEM:   status = TptIoctlAllocMem(pExt, pIrp, pSl);  break;
    case IOCTL_TPT_FREE_MEM:    status = TptIoctlFreeMem(pExt, pIrp, pSl);   break;
    case IOCTL_TPT_SUBMIT_CMD:  status = TptIoctlSubmitCmd(pExt, pIrp, pSl); break;
    case IOCTL_TPT_WAIT:        status = TptIoctlWait(pExt, pIrp, pSl);      break;
    case IOCTL_TPT_QUERY_PERF:  status = TptIoctlQueryPerf(pExt, pIrp, pSl); break;
    case IOCTL_TPT_RESET_GPU:   status = TptIoctlResetGpu(pExt, pIrp, pSl);  break;
    default:
        status = STATUS_INVALID_DEVICE_REQUEST;
        pIrp->IoStatus.Status = status;
        pIrp->IoStatus.Information = 0;
        IoCompleteRequest(pIrp, IO_NO_INCREMENT);
        return status;
    }
    return status;
}

/*============================================================================
 * IRQ handler (DIRQL — keep minimal)
 *============================================================================*/
BOOLEAN TptInterruptHandler(PKINTERRUPT Interrupt, PVOID Context)
{
    UNREFERENCED_PARAMETER(Interrupt);
    PDEVICE_OBJECT  pDev = (PDEVICE_OBJECT)Context;
    PTPT_DEVICE_EXT pExt = (PTPT_DEVICE_EXT)pDev->DeviceExtension;
    UINT32 pend;

    pend = TPT_READ32(pExt, TPT_REG_IRQ_PEND);
    if (!pend) return FALSE;

    /* ACK at DIRQL, defer work to DPC */
    TPT_WRITE32(pExt, TPT_REG_IRQ_PEND, pend);
    KeInsertQueueDpc(&pExt->CompletionDpc, (PVOID)(ULONG_PTR)pend, NULL);
    return TRUE;
}

/*============================================================================
 * DPC — deferred IRQ processing
 *============================================================================*/
VOID TptCompletionDpc(PKDPC Dpc, PVOID Context, PVOID Arg1, PVOID Arg2)
{
    UNREFERENCED_PARAMETER(Dpc); UNREFERENCED_PARAMETER(Arg2);
    PDEVICE_OBJECT  pDev  = (PDEVICE_OBJECT)Context;
    PTPT_DEVICE_EXT pExt  = (PTPT_DEVICE_EXT)pDev->DeviceExtension;
    UINT32          pend  = (UINT32)(ULONG_PTR)Arg1;

    if (pend & TPT_IRQ_KERNEL_DONE) {
        pExt->SeqNoCompleted++;
        KeSetEvent(&pExt->CompletionEvent, IO_NO_INCREMENT, FALSE);
    }
}

/*============================================================================
 * IOCTL handlers
 *============================================================================*/
NTSTATUS TptIoctlGetInfo(PTPT_DEVICE_EXT pExt, PIRP pIrp, PIO_STACK_LOCATION pSl)
{
    tpt_info_t *pOut = (tpt_info_t *)pIrp->AssociatedIrp.SystemBuffer;
    UINT32 ver;

    if (pSl->Parameters.DeviceIoControl.OutputBufferLength < sizeof(*pOut)) {
        pIrp->IoStatus.Status = STATUS_BUFFER_TOO_SMALL;
        IoCompleteRequest(pIrp, IO_NO_INCREMENT);
        return STATUS_BUFFER_TOO_SMALL;
    }

    ver = TPT_READ32(pExt, TPT_REG_VERSION);
    RtlZeroMemory(pOut, sizeof(*pOut));
    pOut->version_major    = ver >> 16;
    pOut->version_minor    = ver & 0xFFFF;
    pOut->vram_bytes       = pExt->VramBytes;
    pOut->num_sm           = pExt->NumSm;
    pOut->num_warps_per_sm = 64;
    pOut->warp_lanes       = 32;
    pOut->num_ctas         = 16;
    pOut->caps             = TPT_CAP_TENSOR | TPT_CAP_FP64;

    pIrp->IoStatus.Status      = STATUS_SUCCESS;
    pIrp->IoStatus.Information = sizeof(*pOut);
    IoCompleteRequest(pIrp, IO_NO_INCREMENT);
    return STATUS_SUCCESS;
}

NTSTATUS TptIoctlAllocMem(PTPT_DEVICE_EXT pExt, PIRP pIrp, PIO_STACK_LOCATION pSl)
{
    tpt_alloc_mem_t *pInOut = (tpt_alloc_mem_t *)pIrp->AssociatedIrp.SystemBuffer;
    PTPT_BUFFER      pBuf;
    KIRQL            irql;
    UINT64           aligned;

    if (pSl->Parameters.DeviceIoControl.InputBufferLength  < sizeof(*pInOut) ||
        pSl->Parameters.DeviceIoControl.OutputBufferLength < sizeof(*pInOut)) {
        pIrp->IoStatus.Status = STATUS_BUFFER_TOO_SMALL;
        IoCompleteRequest(pIrp, IO_NO_INCREMENT);
        return STATUS_BUFFER_TOO_SMALL;
    }

    pBuf = (PTPT_BUFFER)ExAllocateFromLookasideListEx(&pExt->BufferPool);
    if (!pBuf) {
        pIrp->IoStatus.Status = STATUS_INSUFFICIENT_RESOURCES;
        IoCompleteRequest(pIrp, IO_NO_INCREMENT);
        return STATUS_INSUFFICIENT_RESOURCES;
    }

    RtlZeroMemory(pBuf, sizeof(*pBuf));
    pBuf->SizeBytes = pInOut->size_bytes;
    pBuf->Flags     = pInOut->flags;

    /* Bump allocate from VRAM */
    aligned = (pInOut->size_bytes + VRAM_ALIGN - 1) & ~(VRAM_ALIGN - 1);
    KeAcquireSpinLock(&pExt->VramLock, &irql);
    pBuf->PhysAddr   = pExt->VramBump;
    pExt->VramBump  += aligned;
    KeReleaseSpinLock(&pExt->VramLock, irql);

    pBuf->Handle = pBuf->PhysAddr;

    KeAcquireSpinLock(&pExt->BufferLock, &irql);
    InsertTailList(&pExt->BufferList, &pBuf->ListEntry);
    KeReleaseSpinLock(&pExt->BufferLock, irql);

    pInOut->handle    = pBuf->Handle;
    pInOut->phys_addr = pBuf->PhysAddr;

    pIrp->IoStatus.Status      = STATUS_SUCCESS;
    pIrp->IoStatus.Information = sizeof(*pInOut);
    IoCompleteRequest(pIrp, IO_NO_INCREMENT);
    return STATUS_SUCCESS;
}

NTSTATUS TptIoctlFreeMem(PTPT_DEVICE_EXT pExt, PIRP pIrp, PIO_STACK_LOCATION pSl)
{
    tpt_free_mem_t *pIn = (tpt_free_mem_t *)pIrp->AssociatedIrp.SystemBuffer;
    PLIST_ENTRY     pEntry;
    PTPT_BUFFER     pBuf = NULL;
    KIRQL           irql;

    if (pSl->Parameters.DeviceIoControl.InputBufferLength < sizeof(*pIn)) {
        pIrp->IoStatus.Status = STATUS_BUFFER_TOO_SMALL;
        IoCompleteRequest(pIrp, IO_NO_INCREMENT);
        return STATUS_BUFFER_TOO_SMALL;
    }

    KeAcquireSpinLock(&pExt->BufferLock, &irql);
    for (pEntry  = pExt->BufferList.Flink;
         pEntry != &pExt->BufferList;
         pEntry  = pEntry->Flink) {
        PTPT_BUFFER pCandidate = CONTAINING_RECORD(pEntry, TPT_BUFFER, ListEntry);
        if (pCandidate->Handle == pIn->handle) {
            pBuf = pCandidate;
            RemoveEntryList(pEntry);
            break;
        }
    }
    KeReleaseSpinLock(&pExt->BufferLock, irql);

    if (pBuf) ExFreeToLookasideListEx(&pExt->BufferPool, pBuf);

    pIrp->IoStatus.Status      = pBuf ? STATUS_SUCCESS : STATUS_INVALID_HANDLE;
    pIrp->IoStatus.Information = 0;
    IoCompleteRequest(pIrp, IO_NO_INCREMENT);
    return pIrp->IoStatus.Status;
}

NTSTATUS TptIoctlSubmitCmd(PTPT_DEVICE_EXT pExt, PIRP pIrp, PIO_STACK_LOCATION pSl)
{
    tpt_submit_cmd_t *pInOut = (tpt_submit_cmd_t *)pIrp->AssociatedIrp.SystemBuffer;
    UINT32 head;

    if (pSl->Parameters.DeviceIoControl.InputBufferLength  < sizeof(*pInOut) ||
        pSl->Parameters.DeviceIoControl.OutputBufferLength < sizeof(*pInOut)) {
        pIrp->IoStatus.Status = STATUS_BUFFER_TOO_SMALL;
        IoCompleteRequest(pIrp, IO_NO_INCREMENT);
        return STATUS_BUFFER_TOO_SMALL;
    }

    /* Copy descriptor into command ring slot */
    head = pExt->CmdRingHead;
    RtlCopyMemory((PUCHAR)pExt->CmdRingVa + (head * 64),
                  &pInOut->desc, sizeof(tpt_cmd_desc_t));

    pExt->CmdRingHead = (head + 1) % CMDRING_ENTRIES;
    TPT_WRITE32(pExt, TPT_REG_CMDRING_HEAD, pExt->CmdRingHead);

    pExt->SeqNoIssued++;
    pInOut->seq_no = pExt->SeqNoIssued;

    pIrp->IoStatus.Status      = STATUS_SUCCESS;
    pIrp->IoStatus.Information = sizeof(*pInOut);
    IoCompleteRequest(pIrp, IO_NO_INCREMENT);
    return STATUS_SUCCESS;
}

NTSTATUS TptIoctlWait(PTPT_DEVICE_EXT pExt, PIRP pIrp, PIO_STACK_LOCATION pSl)
{
    tpt_wait_complete_t *pInOut = (tpt_wait_complete_t *)pIrp->AssociatedIrp.SystemBuffer;
    LARGE_INTEGER timeout;
    NTSTATUS status;

    if (pSl->Parameters.DeviceIoControl.InputBufferLength  < sizeof(*pInOut) ||
        pSl->Parameters.DeviceIoControl.OutputBufferLength < sizeof(*pInOut)) {
        pIrp->IoStatus.Status = STATUS_BUFFER_TOO_SMALL;
        IoCompleteRequest(pIrp, IO_NO_INCREMENT);
        return STATUS_BUFFER_TOO_SMALL;
    }

    if (pExt->SeqNoCompleted >= pInOut->seq_no) {
        pInOut->status = TPT_WAIT_OK;
        pIrp->IoStatus.Status = STATUS_SUCCESS;
        pIrp->IoStatus.Information = sizeof(*pInOut);
        IoCompleteRequest(pIrp, IO_NO_INCREMENT);
        return STATUS_SUCCESS;
    }

    /* Convert ms timeout to 100-ns units (negative = relative) */
    timeout.QuadPart = pInOut->timeout_ms
                     ? -(LONGLONG)pInOut->timeout_ms * 10000LL
                     : (LONGLONG)0x7FFFFFFFFFFFFFFFLL;

    status = KeWaitForSingleObject(&pExt->CompletionEvent,
                                   Executive, KernelMode, FALSE,
                                   pInOut->timeout_ms ? &timeout : NULL);

    pInOut->status = (status == STATUS_TIMEOUT) ? TPT_WAIT_TIMEOUT : TPT_WAIT_OK;
    pIrp->IoStatus.Status      = STATUS_SUCCESS;
    pIrp->IoStatus.Information = sizeof(*pInOut);
    IoCompleteRequest(pIrp, IO_NO_INCREMENT);
    return STATUS_SUCCESS;
}

NTSTATUS TptIoctlQueryPerf(PTPT_DEVICE_EXT pExt, PIRP pIrp, PIO_STACK_LOCATION pSl)
{
    tpt_perf_counters_t *pOut = (tpt_perf_counters_t *)pIrp->AssociatedIrp.SystemBuffer;

    if (pSl->Parameters.DeviceIoControl.OutputBufferLength < sizeof(*pOut)) {
        pIrp->IoStatus.Status = STATUS_BUFFER_TOO_SMALL;
        IoCompleteRequest(pIrp, IO_NO_INCREMENT);
        return STATUS_BUFFER_TOO_SMALL;
    }

    pOut->inst_retired = ((UINT64)TPT_READ32(pExt, TPT_REG_PERF_INST_HI) << 32) |
                          TPT_READ32(pExt, TPT_REG_PERF_INST_LO);
    pOut->core_cycles  = ((UINT64)TPT_READ32(pExt, TPT_REG_PERF_CYCL_HI) << 32) |
                          TPT_READ32(pExt, TPT_REG_PERF_CYCL_LO);
    pOut->l1d_misses   = TPT_READ32(pExt, TPT_REG_PERF_L1D_MISS);
    pOut->l2_misses    = TPT_READ32(pExt, TPT_REG_PERF_L2_MISS);
    pOut->branch_mispred = 0;
    pOut->warp_stalls    = 0;

    pIrp->IoStatus.Status      = STATUS_SUCCESS;
    pIrp->IoStatus.Information = sizeof(*pOut);
    IoCompleteRequest(pIrp, IO_NO_INCREMENT);
    return STATUS_SUCCESS;
}

NTSTATUS TptIoctlResetGpu(PTPT_DEVICE_EXT pExt, PIRP pIrp, PIO_STACK_LOCATION pSl)
{
    UNREFERENCED_PARAMETER(pSl);
    TPT_WRITE32(pExt, TPT_REG_CTRL, TPT_CTRL_RESET);
    KeStallExecutionProcessor(10000);  /* 10 ms */
    TPT_WRITE32(pExt, TPT_REG_CTRL, TPT_CTRL_BOOT | TPT_CTRL_IRQ_EN);
    pExt->SeqNoIssued = pExt->SeqNoCompleted = 0;
    pIrp->IoStatus.Status      = STATUS_SUCCESS;
    pIrp->IoStatus.Information = 0;
    IoCompleteRequest(pIrp, IO_NO_INCREMENT);
    return STATUS_SUCCESS;
}
