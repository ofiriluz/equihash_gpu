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
#include <blake2.h>
#include <algorithm>

#define SEED_SIZE 4 // 4x32bit
#define MAX_BUCKET_AMOUNT 5
#define LOCAL_WORK_GROUP_SIZE 64
#define MAX_NONCE 0xFFFFF
#define HASH_BLOCK_SIZE 128

namespace Equihash
{
    // Structure to be used on the GPU aswell
    struct EquihashGPUContext
    {
        uint32_t N, K;
        uint32_t collision_bits_length;
        uint32_t collision_bytes_length;
        uint32_t hash_length;
        uint32_t indices_per_hash_output;
        uint32_t hash_output;
        uint32_t full_width;
        uint32_t init_size;
        uint32_t solution_size;

        uint32_t seed[SEED_SIZE]; // Later for nonce and index
    };

    struct BlakeGPU
    {
        uint64_t hash_state[8];
        uint8_t  buf[2*HASH_BLOCK_SIZE];
        uint32_t buflen;
        uint64_t t[2];
    };

    class EquihashGPUSolver : public IEquihashSolver
    {
    private:
        EquihashGPUConfig gpu_config_;
        EquihashGPUContext equihash_context_;

        // OpenCL buffers to be used
        cl::Buffer table_buffer_;
        cl::Buffer collision_table_buffer_;
        cl::Buffer collision_table_size_buffer_;
        cl::Buffer solutions_buffer_;
        cl::Buffer solutions_size_buffer_;
        cl::Buffer digest_buffer_;
        cl::Buffer context_buffer_;

    private:
        BlakeGPU create_initial_digest(size_t nonce);
        void initialize_context();
        void prepare_buffers();
        void enqueue_and_run_hash_kernel(size_t nonce);
        bool enqueue_and_run_coliision_detection_rounds_kernel();
        std::vector<Proof> enqueue_and_run_solutions_kernel(size_t nonce);

    public:
        EquihashGPUSolver(uint32_t N, uint32_t K, uint32_t seed[SEED_SIZE]);
        virtual ~EquihashGPUSolver();

        virtual std::vector<Proof> find_proof() override;
        virtual bool verify_proof(const Proof & proof) override;
    };
}

#endif