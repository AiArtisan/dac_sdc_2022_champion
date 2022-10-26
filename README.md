# SEUer : DAC-SDC 2022 Champion

## Introduction

This is a repository for a hardware-efficient DNN accelerator on FPGA specialized in object detection and tracking. The design won first place in [the 59th IEEE/ACM Design Automation Conference System Design Contest (DAC-SDC)](https://byuccl.github.io/dac_sdc_2022/results/).

Designed by:

> Jingwei Zhang, Xinye Cao, Yu Zhang, Guoqing Li, Meng Zhang

> SEUer Group, Southeast University

[![picture](https://github.com/seujingwei/dac_sdc_2022_champion/raw/master/ranking.png)]()

## Overview

The DAC 2022 System Design Contest focused on low-power object detection on an embedded FPGA system. Contestants received a training dataset provided by DJI, and a hidden dataset used to evaluate the performance of the designs in terms of accuracy and power. Contestants competed to create the best performing design on a Ultra 96 v2 FPGA board. Grand cash awards were given to the top three teams. The award ceremony was held at the 2022 IEEE/ACM Design Automation Conference.

## Build the Project

1. **Generate HLS project by running:**

   ```shell
   cd ./scripts
   vivado_hls hls_script.tcl
   ```

2. **Generate Vivado project by running:**

   ```shell
   vivado -mode tcl -source rtl_script.tcl
   ```
