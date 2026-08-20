/*============================================================================
 * tpt_driver.h — TPT GPU Driver ABI
 *============================================================================
 * TPT GPU — Tensor Processing Technology
 * License: Apache License 2.0 (with Express Patent Grant)
 *
 * This header defines the ABI shared between:
 *   - All OS kernel drivers  (Linux DRM, Windows WDM, macOS DriverKit)
 *   - The userspace daemon   (crates/tpt-gpu-driver-daemon)
 *   - The layer4 runtime     (crates/tpt-gpu-runtime — kernel launch / memory alloc)
 *
 * Layout:
 *   § 1  MMIO register map (PCIe BAR0)
 *   § 2  IOCTL command codes
 *   § 3  IOCTL argument structures
 *   § 4  Error codes
 *   § 5  Capability flags
 *   § 6  Interrupt status bits
 *============================================================================*/

#ifndef TPT_DRIVER_H
#define TPT_DRIVER_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

/*===========================================================================
 * § 1  MMIO Register Map  (BAR0, 4 KiB window, 32-bit registers)
 *===========================================================================
 * Offsets match tpt_csr.sv implementation.
 */

/** Control register: [0] BOOT, [1] RESET, [2] IRQ_EN */
#define TPT_REG_CTRL          0x000u
#  define TPT_CTRL_BOOT         (1u << 0)
#  define TPT_CTRL_RESET        (1u << 1)
#  define TPT_CTRL_IRQ_EN       (1u << 2)

/** Status register (read-only): [0] READY, [1] IDLE, [2] ERROR */
#define TPT_REG_STATUS        0x004u
#  define TPT_STATUS_READY      (1u << 0)
#  define TPT_STATUS_IDLE       (1u << 1)
#  define TPT_STATUS_ERROR      (1u << 2)

/** Interrupt pending register (write 1 to clear) */
#define TPT_REG_IRQ_PEND      0x008u
#  define TPT_IRQ_DONE          (1u << 0)
#  define TPT_IRQ_FAULT         (1u << 1)
#  define TPT_IRQ_TIMEOUT       (1u << 2)

/** Interrupt mask register */
#define TPT_REG_IRQ_MASK      0x00Cu

/** Doorbell — write warp index to dispatch */
#define TPT_REG_DOORBELL      0x014u

/** Scheduler enable: [0] enable */
#define TPT_REG_SCHED_EN      0x020u

/** Active warp count (read-only) */
#define TPT_REG_WARP_COUNT    0x024u

/** VRAM total size in bytes (lower 32 bits) */
#define TPT_REG_VRAM_LO       0x030u

/** VRAM total size in bytes (upper 32 bits) */
#define TPT_REG_VRAM_HI       0x034u

/** CTA count (read-only) */
#define TPT_REG_CTA_COUNT     0x038u

/** Hardware version: [31:16] major, [15:0] minor */
#define TPT_REG_VERSION       0x03Cu

/** Warp PC table base (write 40-bit PA lower 32 bits here) */
#define TPT_REG_WARP_PC_LO    0x040u

/** Warp PC table base upper 8 bits */
#define TPT_REG_WARP_PC_HI    0x044u

/** Command ring base PA (lower 32 bits) */
#define TPT_REG_CMDRING_LO    0x080u

/** Command ring base PA (upper 8 bits) */
#define TPT_REG_CMDRING_HI    0x084u

/** Command ring capacity (number of 64-byte descriptors) */
#define TPT_REG_CMDRING_CAP   0x088u

/** Command ring head (driver writes) */
#define TPT_REG_CMDRING_HEAD  0x08Cu

/** Command ring tail (hardware writes — read-only for driver) */
#define TPT_REG_CMDRING_TAIL  0x090u

/** Perf counter: instructions retired (lower 32 bits) */
#define TPT_REG_PERF_INST_LO  0x100u
#define TPT_REG_PERF_INST_HI  0x104u

/** Perf counter: core cycles */
#define TPT_REG_PERF_CYCL_LO  0x108u
#define TPT_REG_PERF_CYCL_HI  0x10Cu

/** Perf counter: L1D cache misses */
#define TPT_REG_PERF_L1D_MISS 0x110u

/** Perf counter: L2 cache misses */
#define TPT_REG_PERF_L2_MISS  0x114u

/*===========================================================================
 * § 2  IOCTL command codes
 *===========================================================================
 * Encoded as (type << 8) | number for cross-platform compat.
 * Linux uses _IOWR('T', n, struct), Windows uses IOCTL_TPT_* constants,
 * macOS uses IOUserClient method selectors — all map to these logical IDs.
 */

#define TPT_IOC_GET_INFO          0x5401u  /* get device info */
#define TPT_IOC_ALLOC_MEM         0x5402u  /* allocate VRAM buffer */
#define TPT_IOC_FREE_MEM          0x5403u  /* free VRAM buffer */
#define TPT_IOC_MAP_MEM           0x5404u  /* map VRAM to userspace VA */
#define TPT_IOC_UNMAP_MEM         0x5405u  /* unmap VRAM from userspace */
#define TPT_IOC_SUBMIT_CMD        0x5406u  /* submit kernel launch command */
#define TPT_IOC_WAIT_COMPLETE     0x5407u  /* wait for command completion */
#define TPT_IOC_QUERY_PERF        0x5408u  /* read hardware perf counters */
#define TPT_IOC_RESET_GPU         0x5409u  /* reset GPU (privileged) */
#define TPT_IOC_SET_PAGE_TABLE    0x540Au  /* install page table for context */

/*===========================================================================
 * § 3  IOCTL argument structures
 *===========================================================================*/

/** TPT_IOC_GET_INFO */
typedef struct tpt_info {
    uint32_t version_major;    /**< hardware major version */
    uint32_t version_minor;    /**< hardware minor version */
    uint64_t vram_bytes;       /**< total VRAM in bytes */
    uint32_t num_sm;           /**< streaming multiprocessors */
    uint32_t num_warps_per_sm; /**< warp pool depth per SM */
    uint32_t warp_lanes;       /**< SIMD lanes per warp */
    uint32_t num_ctas;         /**< max concurrent CTAs */
    uint32_t caps;             /**< capability flags (§ 5) */
    uint32_t _pad[3];
} tpt_info_t;

/** TPT_IOC_ALLOC_MEM */
typedef struct tpt_alloc_mem {
    uint64_t size_bytes;       /**< [in]  requested size */
    uint32_t flags;            /**< [in]  allocation flags */
#  define TPT_MEM_CACHED     (1u << 0)  /**< cacheable mapping */
#  define TPT_MEM_CONTIGUOUS (1u << 1)  /**< physically contiguous */
#  define TPT_MEM_PINNED     (1u << 2)  /**< pinned (no swap) */
    uint32_t _pad;
    uint64_t handle;           /**< [out] opaque buffer handle */
    uint64_t phys_addr;        /**< [out] device-physical base address */
} tpt_alloc_mem_t;

/** TPT_IOC_FREE_MEM */
typedef struct tpt_free_mem {
    uint64_t handle;           /**< buffer handle from alloc */
} tpt_free_mem_t;

/** TPT_IOC_MAP_MEM */
typedef struct tpt_map_mem {
    uint64_t handle;           /**< [in]  buffer handle */
    uint64_t offset;           /**< [in]  byte offset within buffer */
    uint64_t size_bytes;       /**< [in]  bytes to map (0 = whole buffer) */
    uint32_t prot;             /**< [in]  PROT_READ | PROT_WRITE */
    uint32_t _pad;
    uint64_t user_va;          /**< [out] userspace virtual address */
} tpt_map_mem_t;

/** TPT_IOC_UNMAP_MEM */
typedef struct tpt_unmap_mem {
    uint64_t user_va;          /**< [in] VA from map */
    uint64_t size_bytes;       /**< [in] size from map */
} tpt_unmap_mem_t;

/** Command descriptor (64 bytes, cache-line aligned) */
typedef struct __attribute__((aligned(64))) tpt_cmd_desc {
    uint32_t opcode;           /**< TPT_CMD_* */
#  define TPT_CMD_LAUNCH  0x01u   /**< launch kernel */
#  define TPT_CMD_COPY    0x02u   /**< DMA copy */
#  define TPT_CMD_FENCE   0x03u   /**< memory fence */
#  define TPT_CMD_SIGNAL  0x04u   /**< signal completion event */
    uint32_t flags;
    uint64_t kernel_phys_addr; /**< PA of kernel binary (.tptir) */
    uint32_t grid_x;           /**< CTA grid dimension X */
    uint32_t grid_y;           /**< CTA grid dimension Y */
    uint32_t grid_z;           /**< CTA grid dimension Z */
    uint32_t block_x;          /**< threads per CTA X */
    uint32_t block_y;
    uint32_t block_z;
    uint64_t arg_buf_phys;     /**< PA of argument buffer */
    uint32_t arg_buf_size;     /**< argument buffer size in bytes */
    uint32_t shared_mem_bytes; /**< dynamic shared memory per CTA */
    uint64_t completion_phys;  /**< PA to write 1 when done (0 = none) */
} tpt_cmd_desc_t;

_Static_assert(sizeof(tpt_cmd_desc_t) == 64, "tpt_cmd_desc must be 64 bytes");

/** TPT_IOC_SUBMIT_CMD */
typedef struct tpt_submit_cmd {
    tpt_cmd_desc_t desc;       /**< [in]  command descriptor */
    uint64_t       seq_no;     /**< [out] sequence number for wait */
} tpt_submit_cmd_t;

/** TPT_IOC_WAIT_COMPLETE */
typedef struct tpt_wait_complete {
    uint64_t seq_no;           /**< [in]  sequence number to wait for */
    uint32_t timeout_ms;       /**< [in]  timeout in ms (0 = infinite) */
    uint32_t status;           /**< [out] completion status */
#  define TPT_WAIT_OK      0u
#  define TPT_WAIT_TIMEOUT 1u
#  define TPT_WAIT_FAULT   2u
} tpt_wait_complete_t;

/** TPT_IOC_QUERY_PERF */
typedef struct tpt_perf_counters {
    uint64_t inst_retired;     /**< instructions retired */
    uint64_t core_cycles;      /**< core clock cycles */
    uint64_t l1d_misses;       /**< L1D cache misses */
    uint64_t l2_misses;        /**< L2 cache misses */
    uint64_t branch_mispred;   /**< branch mispredictions */
    uint64_t warp_stalls;      /**< warp stall cycles */
} tpt_perf_counters_t;

/** TPT_IOC_SET_PAGE_TABLE */
typedef struct tpt_set_page_table {
    uint64_t root_phys;        /**< [in] PA of page table root */
    uint32_t asid;             /**< [in] address space ID */
    uint32_t _pad;
} tpt_set_page_table_t;

/*===========================================================================
 * § 4  Error codes
 *===========================================================================*/

#define TPT_OK                 0
#define TPT_ERR_NOMEM         -1   /**< out of VRAM */
#define TPT_ERR_INVALID       -2   /**< invalid argument */
#define TPT_ERR_BUSY          -3   /**< device busy */
#define TPT_ERR_TIMEOUT       -4   /**< operation timed out */
#define TPT_ERR_FAULT         -5   /**< GPU page fault */
#define TPT_ERR_NODEV         -6   /**< device not found */
#define TPT_ERR_PERM          -7   /**< permission denied (privileged op) */
#define TPT_ERR_OVERFLOW      -8   /**< command ring full */
#define TPT_ERR_RESET         -9   /**< device was reset, context invalid */

/*===========================================================================
 * § 5  Capability flags  (tpt_info.caps)
 *===========================================================================*/

#define TPT_CAP_TENSOR        (1u << 0)  /**< tensor / MMA units present */
#define TPT_CAP_FP64          (1u << 1)  /**< FP64 ALU present */
#define TPT_CAP_ATOMICS       (1u << 2)  /**< global atomics */
#define TPT_CAP_PREEMPTION    (1u << 3)  /**< fine-grained preemption */
#define TPT_CAP_VIRTUAL_MEM   (1u << 4)  /**< hardware page table walker */
#define TPT_CAP_ECC           (1u << 5)  /**< ECC protected VRAM */
#define TPT_CAP_P2P           (1u << 6)  /**< peer-to-peer DMA */

/*===========================================================================
 * § 6  Interrupt status bits  (TPT_REG_IRQ_PEND)
 *===========================================================================*/

#define TPT_IRQ_KERNEL_DONE   (1u << 0)  /**< kernel completed */
#define TPT_IRQ_PAGE_FAULT    (1u << 1)  /**< GPU page fault */
#define TPT_IRQ_WATCHDOG      (1u << 2)  /**< hang watchdog fired */
#define TPT_IRQ_THERMAL       (1u << 3)  /**< thermal throttle event */
#define TPT_IRQ_DMA_DONE      (1u << 4)  /**< DMA copy finished */

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* TPT_DRIVER_H */
