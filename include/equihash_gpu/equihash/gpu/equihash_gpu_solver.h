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

#define SEED_MAX_SIZE 128
typedef std::array<uint8_t, SEED_MAX_SIZE> Seed;

namespace Equihash
{
    class EquihashGPUSolver : public IEquihashSolver
    {
    private:
        EquihashGPUConfig gpu_config_;
        uint16_t N_, K_, S_; // S = N/(K+1), calculated once instead of each thread
        Seed seed_;

        // OpenCL buffers to be used
        cl::Buffer buckets_;
        cl::Buffer solutions_;

    private:
        void prepare_buffers();
        void enqueue_and_run_hash_kernel();
        void enqueue_and_run_coliision_kernel();
        void enqueue_and_run_solutions_kernel();

    public:
        EquihashGPUSolver(uint16_t N, uint16_t K, const Seed & seed);
        virtual ~EquihashGPUSolver();

        virtual Proof find_proof() override;
        virtual bool verify_proof(const Proof & proof) override;
    };
}

#endif