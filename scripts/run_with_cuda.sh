#!/bin/bash
# WSL CUDA 环境脚本
export LD_PRELOAD=/usr/lib/wsl/lib/libnvidia-ml.so.1:/usr/lib/wsl/lib/libcuda.so.1
"$@"