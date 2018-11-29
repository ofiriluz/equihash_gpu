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
#include <unistd.h>
#include <stdlib.h>
namespace Equihash
{
    EquihashGPUSolver::EquihashGPUSolver(uint16_t N, uint16_t K, uint32_t seed[SEED_SIZE])
    {
        equihash_context_.N = N;
        equihash_context_.K = K;

        // For now copy all the seed, can be changed later
        memcpy(equihash_context_.seed, seed, SEED_SIZE);
        // memset(equihash_context_.seed, 0, SEED_SIZE+2);
        equihash_context_.seed[SEED_SIZE] = 0;
        equihash_context_.seed[SEED_SIZE+1] = 0;
    }

    EquihashGPUSolver::~EquihashGPUSolver()
    {

    }

    void EquihashGPUSolver::initialize_context()
    {
        // Number of bits to extract for each block - N / (K+1)
        equihash_context_.collision_bits_length = equihash_context_.N / (equihash_context_.K + 1);
        // Same as above but in bytes rounded up 
        equihash_context_.collision_bytes_length = (equihash_context_.collision_bits_length+7)/8;
        // Jump between each block of bits - (k+1) * collisionBytesLength
        equihash_context_.hash_length = (equihash_context_.K+1) * equihash_context_.collision_bytes_length;
        // Number of parts to dervice blocks from the hash - blake_bits(512) / N
        equihash_context_.indices_per_hash_output = 512 / equihash_context_.N;
        // Blake output size in bytes (64)
        equihash_context_.hash_output = sizeof(uint64_t)*8;
        // Width of a single row in the hash table - sizeof(uint32) * (2^(k-1)) + 2*collisionByteLength
        equihash_context_.full_width = sizeof(uint32_t) * (1 << (equihash_context_.K-1)) + 2*equihash_context_.collision_bytes_length;
        // Amount of rows on the hash table - 2^(collisionBitLength+1)
        equihash_context_.init_size = 1 << (equihash_context_.collision_bits_length+1);
        // The indices that gave the solution - 2^k * (n / (k + 1) + 1) / 8
        equihash_context_.solution_size = (1 << equihash_context_.K) * (equihash_context_.N / (equihash_context_.K + 1) + 1) / 8;
    }

    void EquihashGPUSolver::prepare_buffers()
    {
        cl::CommandQueue & queue = gpu_config_.get_device_queues()[0];
        cl_int zero = 0;
        // Create the hash table s
        table_buffer_ = cl::Buffer(
            gpu_config_.get_context(),
            CL_MEM_READ_WRITE,
            equihash_context_.init_size*equihash_context_.full_width
        );
        queue.enqueueFillBuffer(table_buffer_, zero, 0, 
            equihash_context_.init_size*equihash_context_.full_width);

        // TODO - Change this
        collision_table_buffer_ = cl::Buffer(
            gpu_config_.get_context(),
            CL_MEM_READ_WRITE,
            (equihash_context_.init_size/2)*equihash_context_.full_width
        );
        queue.enqueueFillBuffer(collision_table_buffer_, zero, 0, 
            (equihash_context_.init_size/2)*equihash_context_.full_width);

        collision_table_size_buffer_ = cl::Buffer(
            gpu_config_.get_context(),
            CL_MEM_READ_WRITE,
            sizeof(uint32_t)
        );
        queue.enqueueFillBuffer(collision_table_size_buffer_, zero, 0, 
            sizeof(uint32_t));

        // Construct the digests buffer for the hashes (256 bits)
        digest_buffer_ = cl::Buffer(
            gpu_config_.get_context(),
            CL_MEM_READ_ONLY,
            equihash_context_.hash_output
        );
        queue.enqueueFillBuffer(table_buffer_, zero, 0, 
            equihash_context_.hash_output);

        // Construct the context buffer to be used, will be the same as the gpu structure
        context_buffer_ = cl::Buffer(
            gpu_config_.get_context(),
            CL_MEM_READ_ONLY,
            sizeof(EquihashGPUContext)
        );

        // Copy the context to the buffer
        uint8_t * context_pointer = reinterpret_cast<uint8_t*>(&equihash_context_);
        cl::copy(queue,
                 context_pointer, 
                 context_pointer + sizeof(EquihashGPUContext), 
                 context_buffer_);
    }

    void EquihashGPUSolver::enqueue_and_run_hash_kernel(uint32_t nonce)
    {
        std::vector<cl::CommandQueue> & device_queues = gpu_config_.get_device_queues();
        size_t s = ((equihash_context_.init_size / equihash_context_.indices_per_hash_output) + 1)
                        / device_queues.size();

        std::cout << s << std::endl;        
        cl::Kernel & hash_kernel = gpu_config_.get_equihash_hash_kernel();

        hash_kernel.setArg(0, context_buffer_);
        hash_kernel.setArg(1, table_buffer_);
        hash_kernel.setArg(2, digest_buffer_);
        hash_kernel.setArg(3, equihash_context_.hash_output);
        hash_kernel.setArg(4, nonce);

        // Run the kernels
        for(size_t i=0;i<device_queues.size();i++)
        {
            device_queues[i].enqueueNDRangeKernel(hash_kernel, 
                                                 cl::NDRange(i*s),
                                                 cl::NDRange(s),
                                                 cl::NullRange);
            
            device_queues[i].flush();
        }

        // Wait for all to end
        for(size_t i=0;i<device_queues.size();i++)
        {
            device_queues[i].finish();
        }
        
        // std::cout << "Finished creating hashes" << std::endl;
        
    }

    void EquihashGPUSolver::enqueue_and_run_coliision_detection_rounds_kernel()
    {
        std::vector<cl::CommandQueue> & device_queues = gpu_config_.get_device_queues();
        size_t s = equihash_context_.init_size / device_queues.size();
        
        cl::Kernel & collision_detection_kernel = gpu_config_.get_equihash_collision_detection_round_kernel();
     
        cl::Buffer & working_table = table_buffer_;
        cl::Buffer & collision_table = collision_table_buffer_;
        cl::Buffer & temp = table_buffer_;
        cl_int zero = 0;
        uint32_t current_table_size = equihash_context_.init_size;

        // Go over K rounds, each time swapping the buffers
        for(uint8_t i=0;i<equihash_context_.K && current_table_size > 0;i++)
        {
            // Set the arguments
            collision_detection_kernel.setArg(0, context_buffer_);
            collision_detection_kernel.setArg(1, working_table);
            collision_detection_kernel.setArg(2, collision_table);
            collision_detection_kernel.setArg(3, collision_table_size_buffer_);
            collision_detection_kernel.setArg(4, current_table_size);
            collision_detection_kernel.setArg(5, i);

            device_queues[0].enqueueFillBuffer(collision_table_size_buffer_, 
                                                zero, 0, sizeof(uint32_t));
            device_queues[0].flush();
            device_queues[0].finish();

            for(size_t j=0;j<device_queues.size();j++)
            {
                device_queues[j].enqueueNDRangeKernel(collision_detection_kernel,
                                                      cl::NDRange(s*j),
                                                      cl::NDRange(s));
                device_queues[j].flush();
            }

            for(size_t j=0;j<device_queues.size();j++)
            {
                device_queues[j].finish();
            }

            // Swap the buffers and reset / copy the current size
            temp = working_table;
            working_table = collision_table;
            collision_table = temp;

            cl::copy(collision_table_size_buffer_, &current_table_size, (&current_table_size) + sizeof(uint32_t));
        }

        std::cout << "Finished Rounds" << std::endl;
    }

    void EquihashGPUSolver::enqueue_and_run_solutions_kernel()
    {

    }

    Proof EquihashGPUSolver::find_proof()
    {
        // Initialize the GPU config
        gpu_config_.initialize_configuration();
        gpu_config_.prepare_program();

        // Initialize the context for equihash
        initialize_context();

        // Prepare the GPU buffers
        prepare_buffers();

        uint32_t nonce = 1;
        while(nonce < MAX_NONCE)
        {
            nonce++;

            // Fill the buffer hashes, note that this will block and run in batches on the GPU
            enqueue_and_run_hash_kernel(nonce);

            // Perform the coliision detection
            // enqueue_and_run_coliision_detection_rounds_kernel();

            return Proof();
        }

        return Proof();
    }

    bool EquihashGPUSolver::verify_proof(const Proof & proof)
    {

    }
}