/**
 * @file equihash_gpu_config.cpp
 * @author ofir iluz (iluzofir@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2018-11-24
 * 
 * @copyright Copyright (c) 2018
 * 
 */

#include "equihash_gpu/equihash/gpu/equihash_gpu_config.h"

namespace Equihash
{
    EquihashGPUConfig::EquihashGPUConfig(): is_configured_(false)
    {

    }

    EquihashGPUConfig::~EquihashGPUConfig()
    {

    }

    void EquihashGPUConfig::initialize_configuration()
    {
        if(is_configured_)
        {
            return;
        }

        // Discover all the platforms
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);

        // For each platform discover the devices and add them to the device list
        // Note that all of the devices are joined together from nvidia and amd etc
        for(auto && platform : platforms) 
        {
            // Get devices
            std::vector<cl::Device> devices;
            platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

            // Add them to the devices list
            gpu_used_devices_.insert(
                std::end(gpu_used_devices_), std::begin(devices), std::end(devices));
        }

        // Create the context
        cl_int err;
        gpu_context_ = cl::Context(gpu_used_devices_, NULL, NULL, NULL, &err);
        std::cout << EquihashGPUUtils::get_cl_errno(err) << std::endl;
        is_configured_ = true;
    }

    void EquihashGPUConfig::clear_configuration()
    {
        if(!is_configured_)
        {
            return;
        }
        printf("QWEQW\n");
        gpu_used_devices_.clear();
        compiled_gpu_program_ = cl::Program();
        gpu_context_ = cl::Context();
        equihash_kernel_command_queue_ = cl::CommandQueue();
        equihash_hash_kernel_ = cl::Kernel();
        equihash_collision_detection_round_kernel_ = cl::Kernel();

        is_configured_ = false;
    }

    std::string EquihashGPUConfig::read_source(const std::string & path)
    {
        std::fstream stream(path);
        return std::string(std::istreambuf_iterator<char>(stream),
                                        (std::istreambuf_iterator<char>()));
    }

    bool EquihashGPUConfig::prepare_program()
    {
        cl_int err = CL_SUCCESS;
        // std::string source = read_source("/home/ofir/Desktop/Equihash/equihash_gpu/include/equihash_gpu/equihash/gpu/equihash.cl"); 
        std::fstream stream("/home/ofir/Desktop/Equihash/equihash_gpu/include/equihash_gpu/equihash/gpu/equihash.cl");
        std::string source = std::string(std::istreambuf_iterator<char>(stream),
                                        (std::istreambuf_iterator<char>()));

        // Create the program and load the .cl files
        compiled_gpu_program_ = cl::Program(gpu_context_, source, true, &err);
        std::cout << EquihashGPUUtils::get_cl_errno(err) << std::endl;
        if (err != CL_SUCCESS)
        {
            for (cl::Device dev : gpu_used_devices_)
            {
                // Check the build status
                cl_build_status status = compiled_gpu_program_.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(dev);
                // if (status != CL_BUILD_ERROR)
                //     continue;

                // Get the build log
                std::string name     = dev.getInfo<CL_DEVICE_NAME>();
                std::string buildlog = compiled_gpu_program_.getBuildInfo<CL_PROGRAM_BUILD_LOG>(dev);
                std::cerr << "Build log for " << name << ":" << std::endl
                            << buildlog << std::endl;
            }

            return false;
        }

        // Create the command queue for equihash
        equihash_kernel_command_queue_ = cl::CommandQueue(gpu_context_, CL_QUEUE_PROFILING_ENABLE, &err);
        std::cout << EquihashGPUUtils::get_cl_errno(err) << std::endl;
        if (err != CL_SUCCESS)
        {
            std::cout << "Could not create command queue" << std::endl;
            return false;
        }

        // Get the kernel for equihash hashing, collision and solutions
        equihash_hash_kernel_ = cl::Kernel(compiled_gpu_program_,"equihash_initialize_hash" , &err);
        std::cout << EquihashGPUUtils::get_cl_errno(err) << std::endl;
        if (err != CL_SUCCESS)
        {
            std::cout << "Could not retrieve hash kernel" << std::endl;
            return false;
        }
        equihash_collision_detection_round_kernel_ = cl::Kernel(compiled_gpu_program_,"equihash_collision_detection_round" , &err);
        std::cout << EquihashGPUUtils::get_cl_errno(err) << std::endl;
        if (err != CL_SUCCESS)
        {
            std::cout << "Could not retrieve collision kernel" << std::endl;
            return false;
        }
        // equihash_solutions_kernel_ = cl::Kernel(compiled_gpu_program_,EQUIHASH_GPU_KERNEL_SOLUTIONS_NAME , &err);
        // if (err != CL_SUCCESS)
        // {
        //     std::cout << "Could not retrieve solutions kernel" << std::endl;
        //     return false;
        // }

        return true;
    }

    cl::Context & EquihashGPUConfig::get_context()
    {
        return gpu_context_;
    }

    cl::Program & EquihashGPUConfig::get_program()
    {
        return compiled_gpu_program_;
    }

    cl::Kernel & EquihashGPUConfig::get_equihash_hash_kernel()
    {
        return equihash_hash_kernel_;
    }

    cl::Kernel & EquihashGPUConfig::get_equihash_collision_detection_round_kernel()
    {
        return equihash_collision_detection_round_kernel_;
    }

    cl::CommandQueue & EquihashGPUConfig::get_equihash_kernel_command_queue()
    {
        return equihash_kernel_command_queue_;
    }
}