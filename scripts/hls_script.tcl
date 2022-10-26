############################################################
## This file is generated automatically by Vivado HLS.
## Please DO NOT edit it.
## Copyright (C) 1986-2019 Xilinx, Inc. All Rights Reserved.
############################################################
open_project HLS
set_top ultra_net
add_files src/ultranet.cpp -cflags "-std=c++11"
add_files -tb src/ultranet.cpp -cflags "-Wno-unknown-pragmas" -csimflags "-Wno-unknown-pragmas"
open_solution "solution1"
set_part {xczu3eg-sbva484-1-e}
create_clock -period 3.3 -name default
config_sdx -target none
config_rtl -encoding onehot -kernel_profile=0 -module_auto_prefix=0 -mult_keep_attribute=0 -reset control -reset_async=0 -reset_level low -verbose=0
config_export -format ip_catalog -rtl verilog -version 1.0 -vivado_optimization_level 2 -vivado_phys_opt place -vivado_report_level 0
set_clock_uncertainty 12.5%
#source "./HLS/solution1/directives.tcl"
#csim_design
#csynth_design
#cosim_design -trace_level all
#export_design -rtl verilog -format ip_catalog -version "1.0"
