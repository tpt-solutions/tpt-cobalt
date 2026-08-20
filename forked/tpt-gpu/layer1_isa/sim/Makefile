#==============================================================================
# Makefile — TPT ISA Simulation Build (Silicon-Ready)
#==============================================================================
# TPT GPU — Tensor Processing Technology
# License: Apache License 2.0 (with Express Patent Grant)
#==============================================================================
#
# Targets:
#   make sim       — Basic core simulation
#   make silicon   — Full silicon integration test (CSR, scheduler, caches)
#   make asm       — Assemble test programs
#   make clean     — Remove build artifacts
#==============================================================================

IVL = iverilog
VVP = vvp
PY  = python3
GTKWAVE = gtkwave

RTL_DIR = ../rtl
SIM_DIR = .
PROG_DIR = programs

RTL_SRC = \
    $(RTL_DIR)/tpt_pkg.sv \
    $(RTL_DIR)/tpt_decode.sv \
    $(RTL_DIR)/tpt_regfile.sv \
    $(RTL_DIR)/tpt_vregfile.sv \
    $(RTL_DIR)/tpt_pregfile.sv \
    $(RTL_DIR)/tpt_sr_file.sv \
    $(RTL_DIR)/tpt_alu.sv \
    $(RTL_DIR)/tpt_lsu.sv \
    $(RTL_DIR)/tpt_ctrl.sv \
    $(RTL_DIR)/tpt_tensor_unit.sv \
    $(RTL_DIR)/tpt_pipeline.sv \
    $(RTL_DIR)/tpt_warp_sched.sv \
    $(RTL_DIR)/tpt_branch_pred.sv \
    $(RTL_DIR)/tpt_icache.sv \
    $(RTL_DIR)/tpt_dcache.sv \
    $(RTL_DIR)/tpt_mmu.sv \
    $(RTL_DIR)/tpt_csr.sv \
    $(RTL_DIR)/tpt_core.sv \
    $(RTL_DIR)/tpt_gpu_top.sv

TB_SRC = $(SIM_DIR)/tpt_tb.sv
SILICON_TB = $(SIM_DIR)/tpt_silicon_tb.sv
VVP_OUT = tpt_sim.vvp
VCD_OUT = tpt_sim.vcd
SILICON_VVP = tpt_silicon_sim.vvp

all: sim

asm:
	$(PY) tpt_assemble.py $(PROG_DIR)/simple_add.asm

sim: asm
	$(IVL) -g2012 -o $(VVP_OUT) $(RTL_SRC) $(TB_SRC)
	$(VVP) $(VVP_OUT)

silicon:
	$(IVL) -g2012 -DSIMULATION -o $(SILICON_VVP) $(RTL_SRC) $(SILICON_TB)
	$(VVP) $(SILICON_VVP)

wave: $(VCD_OUT)
	$(GTKWAVE) $(VCD_OUT)

clean:
	rm -f $(VVP_OUT) $(VCD_OUT) $(SILICON_VVP)
	rm -rf work/
