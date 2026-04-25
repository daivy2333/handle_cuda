<pir>
<meta>
name: handle_cuda
root: /home/daivy/projects/handle_cuda
profile: generic
lang: CPP,H,PY
</meta>
<units>
u0: tests/test_relu.cpp type=CPP role=lib module=tests
u1: tests/test_matmul.cpp type=CPP role=lib module=tests
u2: tests/test_conv2d.cpp type=CPP role=lib module=tests
u3: tests/test_bias_add.cpp type=CPP role=lib module=tests
u4: tests/test_maxpool2d.cpp type=CPP role=lib module=tests
u5: tests/test_softmax.cpp type=CPP role=lib module=tests
u6: include/cuda_ops.h type=H role=lib module=include
u7: include/cuda_util.h type=H role=lib module=include
u8: scripts/benchmark.py type=PY role=lib module=scripts
</units>
<dependency-pool>
d0: import:[numpy]
d1: import:[stdlib:py]
d2: import:[torch]
d3: include:[cmath]
d4: include:[cstddef]
d5: include:[cstdio]
d6: include:[cstdlib]
d7: include:[cstring]
d8: include:[cuda_ops.h]
d9: include:[cuda_runtime.h]
d10: include:[cuda_util.h]
d11: include:[gtest/gtest.h]
d12: include:[vector]
</dependency-pool>
<dependencies>
u0->refs:[d11 d10 d6 d8 d12 d3]
u1->refs:[d11 d10 d6 d8 d12 d3]
u2->refs:[d11 d10 d6 d8 d12 d3]
u3->refs:[d11 d10 d6 d8 d12 d3]
u4->refs:[d11 d10 d6 d8 d12 d3]
u5->refs:[d11 d10 d6 d8 d12 d3]
u6->refs:[d9 d4]
u7->refs:[d7 d9 d5 d6]
u8->refs:[d1 d2 d0]
</dependencies>
<symbols>
relative_error:u1 func
CudaBuffer:u7 func
allocate:u7 func
clear:u7 func
get_num_blocks:u7 func
host_to_device_async:u7 func
device_to_host:u7 func
device_to_device:u7 func
benchmark_matmul:u8 func
benchmark_relu:u8 func
benchmark_conv2d:u8 func
benchmark_softmax:u8 func
main:u8 func entry=true
</symbols>
<profiles>
  active: c-framework
  c-framework:
    confidence: 0.5
    tags:
      - domain:language-tooling
      - runtime:native
      - stack:c-framework
  system-c:
    confidence: 0.4
    tags:
      - domain:system
      - lang:c
      - runtime:native
</profiles>
</pir>