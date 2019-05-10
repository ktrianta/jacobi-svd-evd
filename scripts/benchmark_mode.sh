#!/usr/bin/env bash

scripts_dir="${0%/*}"

cpupower frequency-set -g performance
${scripts_dir}/disable_cpu_turbo.sh

#echo 0 > /sys/devices/system/cpu/cpu4/online
#echo 0 > /sys/devices/system/cpu/cpu5/online
#echo 0 > /sys/devices/system/cpu/cpu6/online
#echo 0 > /sys/devices/system/cpu/cpu7/online
