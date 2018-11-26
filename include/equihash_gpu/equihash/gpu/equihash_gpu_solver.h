/**
 * @file equihash_gpu_solver.h
 * @author ofir iluz (iluzofir@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2018-11-24
 * 
 * @copyright Copyright (c) 2018
 * 
 */

#ifndef EQUIHASHGPU_EQUIHASH_GPU_SOLVER_H_
#define EQUIHASHGPU_EQUIHASH_GPU_SOLVER_H_

#include <stdint.h>
#include <iostream>
#include <array>
#include "equihash_gpu/equihash/equihash_solver.h"
#include "equihash_gpu/equihash/gpu/equihash_gpu_config.h"

#define SEED_SIZE 4 // 4x32bit
#define MAX_BUCKET_AMOUNT 5
#define LOCAL_WORK_GROUP_SIZE 64
#define MAX_NONCE 0xFFFFF

namespace Equihash
{
    // Structure to be used on the GPU aswell
    struct EquihashGPUContext
    {
        uint16_t N, K, S; // S = N/(K+1), calculated once instead of each thread
        uint8_t seed[SEED_SIZE+2]; // Later for nonce and index
        uint32_t bucket_size;
    };

    class EquihashGPUSolver : public IEquihashSolver
    {
    private:
        EquihashGPUConfig gpu_config_;
        EquihashGPUContext equihash_context_;

        // OpenCL buffers to be used
        cl::Buffer buckets_buffer_;
        cl::Buffer context_buffer_;
        cl::Buffer solutions_buffer_;

    private:
        void prepare_buffers();
        void enqueue_and_run_hash_kernel(uint32_t nonce);
        void enqueue_and_run_coliision_kernel();
        void enqueue_and_run_solutions_kernel();

    public:
        EquihashGPUSolver(uint16_t N, uint16_t K, uint32_t seed[SEED_SIZE]);
        virtual ~EquihashGPUSolver();

        virtual Proof find_proof() override;
        virtual bool verify_proof(const Proof & proof) override;
    };
}

#endif