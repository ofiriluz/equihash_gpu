#include <iostream>
#include <CL/cl.hpp>
#include <sstream>
#include <fstream>
#include <random>
#include <algorithm>
#include <functional>
#include <chrono>
#include <blake2.h>
#include <equihash_gpu/util/Timer.h>

const char *err_code (cl_int err_in)
{
    switch (err_in) {
        case CL_SUCCESS:
            return (char*)"CL_SUCCESS";
        case CL_DEVICE_NOT_FOUND:
            return (char*)"CL_DEVICE_NOT_FOUND";
        case CL_DEVICE_NOT_AVAILABLE:
            return (char*)"CL_DEVICE_NOT_AVAILABLE";
        case CL_COMPILER_NOT_AVAILABLE:
            return (char*)"CL_COMPILER_NOT_AVAILABLE";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:
            return (char*)"CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case CL_OUT_OF_RESOURCES:
            return (char*)"CL_OUT_OF_RESOURCES";
        case CL_OUT_OF_HOST_MEMORY:
            return (char*)"CL_OUT_OF_HOST_MEMORY";
        case CL_PROFILING_INFO_NOT_AVAILABLE:
            return (char*)"CL_PROFILING_INFO_NOT_AVAILABLE";
        case CL_MEM_COPY_OVERLAP:
            return (char*)"CL_MEM_COPY_OVERLAP";
        case CL_IMAGE_FORMAT_MISMATCH:
            return (char*)"CL_IMAGE_FORMAT_MISMATCH";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:
            return (char*)"CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case CL_BUILD_PROGRAM_FAILURE:
            return (char*)"CL_BUILD_PROGRAM_FAILURE";
        case CL_MAP_FAILURE:
            return (char*)"CL_MAP_FAILURE";
        case CL_MISALIGNED_SUB_BUFFER_OFFSET:
            return (char*)"CL_MISALIGNED_SUB_BUFFER_OFFSET";
        case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
            return (char*)"CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
        case CL_INVALID_VALUE:
            return (char*)"CL_INVALID_VALUE";
        case CL_INVALID_DEVICE_TYPE:
            return (char*)"CL_INVALID_DEVICE_TYPE";
        case CL_INVALID_PLATFORM:
            return (char*)"CL_INVALID_PLATFORM";
        case CL_INVALID_DEVICE:
            return (char*)"CL_INVALID_DEVICE";
        case CL_INVALID_CONTEXT:
            return (char*)"CL_INVALID_CONTEXT";
        case CL_INVALID_QUEUE_PROPERTIES:
            return (char*)"CL_INVALID_QUEUE_PROPERTIES";
        case CL_INVALID_COMMAND_QUEUE:
            return (char*)"CL_INVALID_COMMAND_QUEUE";
        case CL_INVALID_HOST_PTR:
            return (char*)"CL_INVALID_HOST_PTR";
        case CL_INVALID_MEM_OBJECT:
            return (char*)"CL_INVALID_MEM_OBJECT";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
            return (char*)"CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case CL_INVALID_IMAGE_SIZE:
            return (char*)"CL_INVALID_IMAGE_SIZE";
        case CL_INVALID_SAMPLER:
            return (char*)"CL_INVALID_SAMPLER";
        case CL_INVALID_BINARY:
            return (char*)"CL_INVALID_BINARY";
        case CL_INVALID_BUILD_OPTIONS:
            return (char*)"CL_INVALID_BUILD_OPTIONS";
        case CL_INVALID_PROGRAM:
            return (char*)"CL_INVALID_PROGRAM";
        case CL_INVALID_PROGRAM_EXECUTABLE:
            return (char*)"CL_INVALID_PROGRAM_EXECUTABLE";
        case CL_INVALID_KERNEL_NAME:
            return (char*)"CL_INVALID_KERNEL_NAME";
        case CL_INVALID_KERNEL_DEFINITION:
            return (char*)"CL_INVALID_KERNEL_DEFINITION";
        case CL_INVALID_KERNEL:
            return (char*)"CL_INVALID_KERNEL";
        case CL_INVALID_ARG_INDEX:
            return (char*)"CL_INVALID_ARG_INDEX";
        case CL_INVALID_ARG_VALUE:
            return (char*)"CL_INVALID_ARG_VALUE";
        case CL_INVALID_ARG_SIZE:
            return (char*)"CL_INVALID_ARG_SIZE";
        case CL_INVALID_KERNEL_ARGS:
            return (char*)"CL_INVALID_KERNEL_ARGS";
        case CL_INVALID_WORK_DIMENSION:
            return (char*)"CL_INVALID_WORK_DIMENSION";
        case CL_INVALID_WORK_GROUP_SIZE:
            return (char*)"CL_INVALID_WORK_GROUP_SIZE";
        case CL_INVALID_WORK_ITEM_SIZE:
            return (char*)"CL_INVALID_WORK_ITEM_SIZE";
        case CL_INVALID_GLOBAL_OFFSET:
            return (char*)"CL_INVALID_GLOBAL_OFFSET";
        case CL_INVALID_EVENT_WAIT_LIST:
            return (char*)"CL_INVALID_EVENT_WAIT_LIST";
        case CL_INVALID_EVENT:
            return (char*)"CL_INVALID_EVENT";
        case CL_INVALID_OPERATION:
            return (char*)"CL_INVALID_OPERATION";
        case CL_INVALID_GL_OBJECT:
            return (char*)"CL_INVALID_GL_OBJECT";
        case CL_INVALID_BUFFER_SIZE:
            return (char*)"CL_INVALID_BUFFER_SIZE";
        case CL_INVALID_MIP_LEVEL:
            return (char*)"CL_INVALID_MIP_LEVEL";
        case CL_INVALID_GLOBAL_WORK_SIZE:
            return (char*)"CL_INVALID_GLOBAL_WORK_SIZE";
        case CL_INVALID_PROPERTY:
            return (char*)"CL_INVALID_PROPERTY";

        default:
            return (char*)"UNKNOWN ERROR";
    }
}

int main(int argc, char ** argv)
{
    // Discover platforms
    std::vector<cl::Platform> platforms;
    cl::Platform * nvidiaPlatform = nullptr;
    cl::Platform::get(&platforms);

    // Find the NVIDIA platform to work with
    std::string platformName;
    for(auto && platform : platforms) 
    {
        platform.getInfo(CL_PLATFORM_NAME, &platformName);
        
        // Found nvidia cuda platform
        if(platformName.find("NVIDIA CUDA") != std::string::npos)
        {
            nvidiaPlatform = &platform;
            break;
        }
    }

    if(nvidiaPlatform)
    {
        // Get the platform devices
        std::vector<cl::Device> devices;
        nvidiaPlatform->getDevices(CL_DEVICE_TYPE_ALL, &devices);

        if(devices.size() > 0)
        {
            int err = 0;
            // Create the context
            cl::Context context(devices);

            // Create a dummy message to hash
            std::string message = "The quick brown fox jumps over the lazy dog The quick brown fox jumps over the lazy dog fox jumps over the lazy dog the lazy do";
            std::vector<uint8_t> outputHashResult(512);

            // Create a cl buffers that we will enqueue 
            cl::Buffer inMessage = cl::Buffer(context, std::begin(message), std::end(message), true, false &err);;
            cl::Buffer outHash(context, CL_MEM_WRITE_ONLY, 512, nullptr, &err);

            // Read the .cl file
            std::ifstream stream("/home/ofir/Desktop/Equihash/equihash_gpu/include/equihash_gpu/blake2b/gpu/blake2b.cl");
            std::string program = std::string(std::istreambuf_iterator<char>(stream),
                (std::istreambuf_iterator<char>()));            

            // Create the program and load the .cl file
            cl::Program testProgram(context, program, true, &err);
            if (err != CL_SUCCESS)
            {
                for (cl::Device dev : devices)
                {
                    // Check the build status
                    cl_build_status status = testProgram.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(dev);
                    if (status != CL_BUILD_ERROR)
                        continue;

                    // Get the build log
                    std::string name     = dev.getInfo<CL_DEVICE_NAME>();
                    std::string buildlog = testProgram.getBuildInfo<CL_PROGRAM_BUILD_LOG>(dev);
                    std::cerr << "Build log for " << name << ":" << std::endl
                                << buildlog << std::endl;
                }
            }

            // Get the command queue for the program
            cl::CommandQueue queue(context, 0, &err);

            // Create the gpu function 
            auto blake2b_kernel = cl::make_kernel<cl::Buffer, cl::Buffer, uint64_t, uint8_t>(testProgram, "blake2b_gpu_hash", &err);
            // Run the kernel K times just to simulate K hashes
            Timer kernel;
            blake2b_kernel(cl::EnqueueArgs(queue, 100000), inMessage, outHash, message.size(), 64);

            queue.finish();
            int64_t t1 = kernel.elapsed();
            std::cout << "Kernel Elapsed time = " << t1 << std::endl;

            // Copy the results to the host memory
            cl::copy(queue, outHash, std::begin(outputHashResult), std::end(outputHashResult));
            std::stringstream hex_stream;
            hex_stream << std::hex;
            
            for(int i=0;i<64;++i)
            {
                hex_stream << static_cast<int>(outputHashResult[i]);
            }

            std::string res = hex_stream.str();
            std::cout << "Hash result = \n" << res << std::endl;
            std::cout << "Hash size = \n" << res.length() << std::endl;

            // Do the same for blake cpu
            const char * msg = message.c_str();
            size_t s = message.length();
            uint8_t out[64];
            Timer blake;
            for(int i=0;i<100000;i++)
            {
                blake2b(out, msg, NULL, 64, s, 0);
            }
            int64_t t2 = blake.elapsed();
            std::cout << "Blake Elapsed time = " << t2 << std::endl;

            std::cout << "Simple bench rate diff [MS] = " << (t2-t1)/1000000 << std::endl; 
        }
    }
}