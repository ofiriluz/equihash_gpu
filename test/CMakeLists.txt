ADD_EXECUTABLE(blake2b_gpu_bench
    blake2b_gpu_bench.cpp
)

TARGET_LINK_LIBRARIES(blake2b_gpu_bench
    OpenCL
    b2
)