/**
 * @file equihash_gpu_config.h
 * @author ofir iluz (iluzofir@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2018-11-24
 * 
 * @copyright Copyright (c) 2018
 * 
 */

#ifndef EQUIHASHGPU_EQUIHASH_GPU_CONFIG_H_
#define EQUIHASHGPU_EQUIHASH_GPU_CONFIG_H_

// #define __CL_ENABLE_EXCEPTIONS 
#if defined(__APPLE__) || defined(__MACOSX)
    #include <OpenCL/cl.hpp>
#else
    #include <CL/cl.hpp>
#endif
#include <fstream>
#include <vector>
#include <iostream>
#include "equihash_gpu/equihash/gpu/equihash_gpu_util.h"

namespace Equihash
{
    class EquihashGPUConfig
    {
    private:
        // OpenCL information to be used
        bool is_configured_;
        cl::Context gpu_context_;
        std::vector<cl::Device> gpu_used_devices_;
        std::vector<cl::CommandQueue> gpu_devices_queues_;
        cl::Program compiled_gpu_program_;
        cl::Kernel equihash_hash_kernel_;
        cl::Kernel equihash_collision_detection_round_kernel_;
        cl::Kernel equihash_solutions_kernel_;

    private:
        std::string read_source(const std::string & path);

    public:
        EquihashGPUConfig();
        virtual ~EquihashGPUConfig();

        void initialize_configuration();
        void clear_configuration();
        bool prepare_program();
        cl::Context & get_context();
        cl::Program & get_program();
        cl::Kernel & get_equihash_hash_kernel();
        cl::Kernel & get_equihash_collision_detection_round_kernel();
        cl::Kernel & get_equihash_solutions_kernel();
        std::vector<cl::Device> & get_devices();
        std::vector<cl::CommandQueue> & get_device_queues();
    };
}

#endif