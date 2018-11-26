/**
 * @file equihash_gpu_solver.cpp
 * @author ofir iluz (iluzofir@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2018-11-24
 * 
 * @copyright Copyright (c) 2018
 * 
 */

#include "equihash_gpu/equihash/gpu/equihash_gpu_solver.h"

namespace Equihash
{
    EquihashGPUSolver::EquihashGPUSolver(uint16_t N, uint16_t K, uint32_t seed[SEED_SIZE])
    {
        equihash_context_.N = N;
        equihash_context_.K = K;
        equihash_context_.S = N / (K + 1);

        // For now copy all the seed, can be changed later
        memcpy(equihash_context_.seed, seed, SEED_SIZE);
        equihash_context_.seed[SEED_SIZE] = 0;
        equihash_context_.seed[SEED_SIZE+1] = 0;
    }

    EquihashGPUSolver::~EquihashGPUSolver()
    {

    }

    void EquihashGPUSolver::prepare_buffers()
    {
        // Create buckets buffer and solutions buffer to be used
        // Buckets amount equal to 2^S hashes, each bucket has K blocks of bits and can hold up to P slots
        // Each block of bits is of size S, to simplify things S is maxed to uint32_t
        // P is defined by default to be 5
        // Index of the iteration is stored at the start of each item at the bucket
        // Bucket structure : |COUNT|B1INDEX|B1BITS|B2INDEX|B2BITS|...|
        // Which means one bucket size = (sizeof(uint32_t) * (K+1))*P + sizeof(uint16_t)
        // Buffer will allocate bucket_size*2^S buckets
        equihash_context_.bucket_size = 
            (sizeof(uint32_t) * (equihash_context_.K + 1))*MAX_BUCKET_AMOUNT + sizeof(uint16_t);
        buckets_buffer_ = cl::Buffer(
            gpu_config_.get_context(),
            CL_MEM_READ_WRITE,
            equihash_context_.bucket_size*(((uint32_t)1) << equihash_context_.S)
        );

        // Construct the context buffer to be used, will be the same as the gpu structure
        context_buffer_ = cl::Buffer(
            gpu_config_.get_context(),
            CL_MEM_READ_ONLY,
            sizeof(EquihashGPUContext)
        );

        // Copy the context to the buffer
        uint8_t * context_pointer = reinterpret_cast<uint8_t*>(&equihash_context_);
        cl::copy(gpu_config_.get_equihash_kernel_command_queue(),
                 context_pointer, 
                 context_pointer + sizeof(EquihashGPUContext), 
                 context_buffer_);
    }

    void EquihashGPUSolver::enqueue_and_run_hash_kernel(uint32_t nonce)
    {
        std::vector<cl::Event> hash_event(1);

        // The global work size is equal to 2^S
        // Offset is set to 0, no need to alter the nonces
        cl::NDRange global_work_offset = 0;
        cl::NDRange global_work_size(1 << equihash_context_.S);
        cl::NDRange local_work_size(LOCAL_WORK_GROUP_SIZE);

        // Enqueue the buckets and the context to the kernel
        cl::CommandQueue & queue = gpu_config_.get_equihash_kernel_command_queue();
        cl::Kernel & hash_kernel = gpu_config_.get_equihash_hash_kernel();

        hash_kernel.setArg(0, context_buffer_);
        hash_kernel.setArg(1, buckets_buffer_);
        hash_kernel.setArg(2, nonce);

        // Run the kernel 
        queue.enqueueNDRangeKernel(hash_kernel, global_work_offset, 
                                   global_work_size, local_work_size, 
                                   nullptr, &hash_event[0]);

        // Wait for the kernel to end
        cl::WaitForEvents(hash_event);
    }

    void EquihashGPUSolver::enqueue_and_run_coliision_kernel()
    {

    }

    void EquihashGPUSolver::enqueue_and_run_solutions_kernel()
    {

    }

    Proof EquihashGPUSolver::find_proof()
    {
        uint32_t nonce = 1;
        while(nonce < MAX_NONCE)
        {
            nonce++;

            // Prepare the GPU buffers
            prepare_buffers();

            // Fill the buffer hashes, note that this will block and run in batches on the GPU
            enqueue_and_run_hash_kernel(nonce);
        }
    }

    bool EquihashGPUSolver::verify_proof(const Proof & proof)
    {

    }
}