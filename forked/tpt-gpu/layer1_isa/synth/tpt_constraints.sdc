##==============================================================================
## tpt_constraints.sdc — TPT GPU Silicon Timing Constraints
##==============================================================================
## TPT GPU — Tensor Processing Technology
## License: Apache License 2.0 (with Express Patent Grant)
##
## Target process: TSMC 7nm (N7) or 16nm FinFET (N16)
## Core clock:     1.0 GHz (1 ns period)
## GDDR6 PHY:      14 Gbps per pin (clocked separately by PHY PLL)
##==============================================================================

##------------------------------------------------------------------------------
## 1. Clock definitions
##------------------------------------------------------------------------------

# Core clock (from PLL output — clk_i top-level port)
create_clock -name CLK_CORE -period 1.000 [get_ports clk_i]

# GDDR6 PHY clock (500 MHz DDR — created inside the PHY macro, not here)
# create_clock -name CLK_PHY -period 2.000 [get_pins phy_inst/clk_out]

# Async MMIO clock from PCIe — 250 MHz
create_clock -name CLK_PCIE -period 4.000 [get_ports host_mmio_clk_i]

# Set clock uncertainty (jitter + skew)
set_clock_uncertainty -setup 0.050 [get_clocks CLK_CORE]
set_clock_uncertainty -hold  0.030 [get_clocks CLK_CORE]

# Clock transition (slew)
set_clock_transition 0.025 [get_clocks CLK_CORE]

##------------------------------------------------------------------------------
## 2. Clock groups (asynchronous domains)
##------------------------------------------------------------------------------

set_clock_groups -asynchronous \
    -group [get_clocks CLK_CORE] \
    -group [get_clocks CLK_PCIE]

##------------------------------------------------------------------------------
## 3. Input / Output delays
##   All delays relative to the core clock rising edge.
##   Assumes a board-level setup of ±0.2 ns trace skew.
##------------------------------------------------------------------------------

# Host MMIO inputs (PCIe clock domain — handled via CDC)
set_input_delay  -clock CLK_PCIE -max 1.500 [get_ports {host_mmio_addr_i[*] host_mmio_req_i host_mmio_we_i host_mmio_wdata_i[*]}]
set_input_delay  -clock CLK_PCIE -min 0.100 [get_ports {host_mmio_addr_i[*] host_mmio_req_i host_mmio_we_i host_mmio_wdata_i[*]}]

set_output_delay -clock CLK_PCIE -max 1.500 [get_ports {host_mmio_rdata_o[*] host_mmio_ack_o host_intr_o}]
set_output_delay -clock CLK_PCIE -min 0.100 [get_ports {host_mmio_rdata_o[*] host_mmio_ack_o host_intr_o}]

# PHY command outputs (synchronous to CLK_CORE, latched by PHY)
set_output_delay -clock CLK_CORE -max 0.250 [get_ports {phy_cmd_o[*] phy_bank_o[*] phy_row_o[*] phy_col_o[*] phy_wdata_o[*] phy_cs_n_o phy_cke_o}]
set_output_delay -clock CLK_CORE -min 0.050 [get_ports {phy_cmd_o[*] phy_bank_o[*] phy_row_o[*] phy_col_o[*] phy_wdata_o[*] phy_cs_n_o phy_cke_o}]

# PHY data inputs
set_input_delay  -clock CLK_CORE -max 0.350 [get_ports {phy_rdata_i[*] phy_rdata_valid_i}]
set_input_delay  -clock CLK_CORE -min 0.050 [get_ports {phy_rdata_i[*] phy_rdata_valid_i}]

##------------------------------------------------------------------------------
## 4. Timing exceptions
##------------------------------------------------------------------------------

# False paths across async clock boundaries (MMIO ↔ core)
set_false_path -from [get_clocks CLK_PCIE] -to [get_clocks CLK_CORE]
set_false_path -from [get_clocks CLK_CORE] -to [get_clocks CLK_PCIE]

# Multicycle path: memory controller state machine has 2-cycle paths
# for bank tracking (combinational → registered next cycle)
set_multicycle_path -setup 2 -from [get_cells *memctrl_inst/open_row*] -to [get_cells *memctrl_inst/state*]
set_multicycle_path -hold  1 -from [get_cells *memctrl_inst/open_row*] -to [get_cells *memctrl_inst/state*]

# Tensor unit MMA latency: 4-cycle pipeline inside tpt_tensor_unit
set_multicycle_path -setup 4 -from [get_cells *gen_sm*/core_inst/alu_inst*] -to [get_cells *gen_sm*/core_inst/alu_inst*result*]
set_multicycle_path -hold  3 -from [get_cells *gen_sm*/core_inst/alu_inst*] -to [get_cells *gen_sm*/core_inst/alu_inst*result*]

##------------------------------------------------------------------------------
## 5. Operating conditions and drive/load
##------------------------------------------------------------------------------

set_operating_conditions -max WORST -library sc7_n7_ss_0p63v_125c
set_operating_conditions -min BEST  -library sc7_n7_ff_0p77v_m40c

# Input drive (modelled as BUFX8 from PCB)
set_drive 0.1 [all_inputs]

# Output load (15 fF PCB pad + trace estimate)
set_load 0.015 [all_outputs]

##------------------------------------------------------------------------------
## 6. Design rules
##------------------------------------------------------------------------------

set_max_transition 0.100 [current_design]
set_max_fanout     32    [current_design]
set_max_capacitance 0.200 [current_design]

##------------------------------------------------------------------------------
## 7. Area / power targets
##------------------------------------------------------------------------------

# Die area budget: 10 mm² active (for 1-SM config at N7)
# set_max_area 10000000  ;# in library units (µm²) — uncomment to enforce

##------------------------------------------------------------------------------
## 8. Clock gating
##------------------------------------------------------------------------------

# Enable integrated clock gating cells for power reduction
set_clock_gating_style -positive_edge_logic integrated -sequential_cell latch

##------------------------------------------------------------------------------
## End of constraints
##------------------------------------------------------------------------------
