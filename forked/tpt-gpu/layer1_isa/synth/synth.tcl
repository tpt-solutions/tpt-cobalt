##==============================================================================
## synth.tcl — TPT GPU Synthesis Script (Synopsys Design Compiler)
##==============================================================================
## TPT GPU — Tensor Processing Technology
## License: Apache License 2.0 (with Express Patent Grant)
##
## Usage:
##   dc_shell -f synth.tcl | tee synth.log
##
## Environment variables expected:
##   TPT_ROOT  — repo root directory
##   TPT_PDK   — PDK library root (e.g. /pdk/tsmc7nm)
##   TPT_NUM_SM — number of SMs to synthesize (default 1)
##==============================================================================

##------------------------------------------------------------------------------
## 0. Environment setup
##------------------------------------------------------------------------------
if {![info exists env(TPT_ROOT)]}  { set env(TPT_ROOT)  "../.." }
if {![info exists env(TPT_PDK)]}   { set env(TPT_PDK)   "/pdk/tsmc7nm" }
if {![info exists env(TPT_NUM_SM)]} { set env(TPT_NUM_SM) 1 }

set RTL_DIR   "$env(TPT_ROOT)/layer1_isa/rtl"
set SDC_FILE  "$env(TPT_ROOT)/layer1_isa/synth/tpt_constraints.sdc"
set UPF_FILE  "$env(TPT_ROOT)/layer1_isa/upf/tpt_power.upf"
set OUT_DIR   "$env(TPT_ROOT)/layer1_isa/synth/out"
set NUM_SM    $env(TPT_NUM_SM)

file mkdir $OUT_DIR

##------------------------------------------------------------------------------
## 1. Library setup
##------------------------------------------------------------------------------
set_app_var target_library  "$env(TPT_PDK)/lib/sc7_n7_ss_0p63v_125c.db \
                              $env(TPT_PDK)/lib/sc7_n7_ff_0p77v_m40c.db"
set_app_var synthetic_library "dw_foundation.sldb"
set_app_var link_library      "* $target_library $synthetic_library \
                               $env(TPT_PDK)/lib/tpt_memories.db \
                               $env(TPT_PDK)/lib/tpt_io_cells.db"

set_app_var search_path       "$RTL_DIR $env(TPT_PDK)/lib"

##------------------------------------------------------------------------------
## 2. RTL file list (compile order: package first, leaves before top)
##------------------------------------------------------------------------------
set RTL_FILES {
    tpt_pkg.sv
    tpt_regfile.sv
    tpt_vregfile.sv
    tpt_pregfile.sv
    tpt_sr_file.sv
    tpt_decode.sv
    tpt_alu.sv
    tpt_lsu.sv
    tpt_tensor_unit.sv
    tpt_pipeline.sv
    tpt_ctrl.sv
    tpt_branch_pred.sv
    tpt_mmu.sv
    tpt_icache.sv
    tpt_dcache.sv
    tpt_l2cache.sv
    tpt_mem_ctrl.sv
    tpt_core.sv
    tpt_warp_sched.sv
    tpt_csr.sv
    tpt_gpu_top.sv
}

##------------------------------------------------------------------------------
## 3. Analyze (parse) RTL
##------------------------------------------------------------------------------
foreach f $RTL_FILES {
    analyze -format sverilog "$RTL_DIR/$f"
}

##------------------------------------------------------------------------------
## 4. Elaborate top-level with parameters
##------------------------------------------------------------------------------
elaborate tpt_gpu_top -parameters "NUM_SM=$NUM_SM"
current_design tpt_gpu_top

##------------------------------------------------------------------------------
## 5. Apply constraints
##------------------------------------------------------------------------------
source $SDC_FILE
set_verification_top

##------------------------------------------------------------------------------
## 6. Apply power intent (UPF multi-voltage)
##------------------------------------------------------------------------------
load_upf $UPF_FILE

##------------------------------------------------------------------------------
## 7. Pre-synthesis checks
##------------------------------------------------------------------------------
check_design
check_timing
report_clock_gating > "$OUT_DIR/pre_cg.rpt"

##------------------------------------------------------------------------------
## 8. Compile — three passes
##    Pass 1: high effort, no-boundary optimization
##    Pass 2: incremental timing fix-up
##    Pass 3: DFT scan insertion prep
##------------------------------------------------------------------------------
compile_ultra -no_autoungroup -timing_high_effort_script
compile_ultra -incremental -timing_high_effort_script

## DFT scan chain insertion (uncomment when DFT collateral is ready)
# set_dft_signal -view existing_dft -type ScanClock -port clk_i
# set_dft_signal -view existing_dft -type Reset     -active_state 0 -port rst_n_i
# set_scan_configuration -chain_count 8
# insert_dft
# compile_ultra -incremental -scan

##------------------------------------------------------------------------------
## 9. Post-synthesis reports
##------------------------------------------------------------------------------
report_timing -max_paths 50 -path full -delay max > "$OUT_DIR/timing_setup.rpt"
report_timing -max_paths 50 -path full -delay min > "$OUT_DIR/timing_hold.rpt"
report_area    > "$OUT_DIR/area.rpt"
report_power   > "$OUT_DIR/power.rpt"
report_cell    > "$OUT_DIR/cells.rpt"
report_resources > "$OUT_DIR/resources.rpt"
report_hierarchy -full > "$OUT_DIR/hierarchy.rpt"
report_qor     > "$OUT_DIR/qor.rpt"

##------------------------------------------------------------------------------
## 10. Write outputs
##------------------------------------------------------------------------------
# Gate-level netlist
write -format verilog -hier -output "$OUT_DIR/tpt_gpu_top_netlist.v"

# SDF (Standard Delay Format) for gate-level simulation
write_sdf -version 3.0 "$OUT_DIR/tpt_gpu_top.sdf"

# SDC for place-and-route (Cadence Innovus / Synopsys ICC2)
write_sdc "$OUT_DIR/tpt_gpu_top_pnr.sdc"

# SAIF for power analysis
# read_saif -input "$OUT_DIR/tpt_gpu_top.saif" -instance tpt_gpu_top
# report_power > "$OUT_DIR/power_saif.rpt"

## DEF floorplan hints (generated from Innovus, not DC — placeholder)
# write_floorplan -output "$OUT_DIR/tpt_floorplan.def"

##------------------------------------------------------------------------------
## 11. Verify constraints met
##------------------------------------------------------------------------------
set wns [get_attribute [get_timing_path -delay max] slack]
if { $wns < 0.0 } {
    puts "WARNING: Setup timing not met. WNS = $wns ns"
} else {
    puts "INFO: Setup timing PASSED. WNS = $wns ns"
}

puts "Synthesis complete. Reports in $OUT_DIR"
