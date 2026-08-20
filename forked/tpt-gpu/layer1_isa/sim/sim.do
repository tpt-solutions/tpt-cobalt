#==============================================================================
# sim.do — ModelSim/Questa Simulation Script for TPT Core
#==============================================================================
# TPT GPU — Tensor Processing Technology
# License: Apache License 2.0 (with Express Patent Grant)
#
# Usage: vsim -do sim.do
#==============================================================================

# Create working library
vlib work
vmap work work

# Compile the TPT package
vlog -sv ../rtl/tpt_pkg.sv

# Compile all RTL files
vlog -sv ../rtl/tpt_decode.sv
vlog -sv ../rtl/tpt_regfile.sv
vlog -sv ../rtl/tpt_alu.sv
vlog -sv ../rtl/tpt_lsu.sv
vlog -sv ../rtl/tpt_ctrl.sv
vlog -sv ../rtl/tpt_tensor_unit.sv
vlog -sv ../rtl/tpt_pipeline.sv
vlog -sv ../rtl/tpt_core.sv

# Compile the testbench
vlog -sv tpt_tb.sv

# Load the simulation
vsim -voptargs=+acc work.tpt_tb

# Add waveforms
log -r *

# Add key signals to wave window
add wave -divider "Clock & Reset"
add wave sim:/tpt_tb/clk
add wave sim:/tpt_tb/rst_n

add wave -divider "Core Signals"
add wave sim:/tpt_tb/dut/imem_addr_o
add wave sim:/tpt_tb/dut/imem_rdata_i
add wave sim:/tpt_tb/dut/decoded

add wave -divider "Pipeline Control"
add wave sim:/tpt_tb/dut/stall_fetch
add wave sim:/tpt_tb/dut/stall_decode
add wave sim:/tpt_tb/dut/flush_decode
add wave sim:/tpt_tb/dut/flush_execute

add wave -divider "Register File"
add wave sim:/tpt_tb/dut/rf_rdata0
add wave sim:/tpt_tb/dut/rf_rdata1

add wave -divider "ALU"
add wave sim:/tpt_tb/dut/alu_result
add wave sim:/tpt_tb/dut/alu_valid
add wave sim:/tpt_tb/dut/alu_zero
add wave sim:/tpt_tb/dut/alu_negative

add wave -divider "LSU"
add wave sim:/tpt_tb/dut/dmem_addr_o
add wave sim:/tpt_tb/dut/dmem_req_o
add wave sim:/tpt_tb/dut/dmem_we_o
add wave sim:/tpt_tb/dut/dmem_wdata_o
add wave sim:/tpt_tb/dut/dmem_rdata_i
add wave sim:/tpt_tb/dut/lsu_data

add wave -divider "Data Memory"
add wave sim:/tpt_tb/data_mem

# Run simulation
run -all

# Show results
view wave
