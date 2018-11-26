CMAKE_MINIMUM_REQUIRED(VERSION 3.7)
PROJECT(equihash_gpu)

INCLUDE_DIRECTORIES(include)

ADD_EXECUTABLE(equihash_gpu
    src/equihash/gpu/equihash_gpu_config.cpp
    src/equihash/gpu/equihash_gpu_solver.cpp
    src/equihash/gpu/equihash_gpu_util.cpp
    src/equihash/proof.cpp
    src/main.cpp
)

TARGET_LINK_LIBRARIES(equihash_gpu
    OpenCL
)

ADD_SUBDIRECTORY(test)