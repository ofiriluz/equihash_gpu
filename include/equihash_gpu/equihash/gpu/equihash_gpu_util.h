/**
 * @file equihash_gpu_util.h
 * @author ofir iluz (iluzofir@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2018-11-26
 * 
 * @copyright Copyright (c) 2018
 * 
 */

#include <string>
#include <CL/cl.hpp>

namespace Equihash
{
    class EquihashGPUUtils
    {
    public:
        static std::string get_cl_errno(cl_int err);
    };
}