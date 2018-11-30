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
        memcpy(equihash_context_.seed, seed, sizeof(uint32_t)*SEED_SIZE);
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
        equihash_context_.hash_output = equihash_context_.indices_per_hash_output*equihash_context_.N / 8;
        // Width of a single row in the hash table - sizeof(uint32) * (2^(k-1)) + 2*collisionByteLength
        equihash_context_.full_width = sizeof(uint32_t) * (1 << (equihash_context_.K-1)) + 2*equihash_context_.collision_bytes_length;
        // Amount of rows on the hash table - 2^(collisionBitLength+1)
        equihash_context_.init_size = 1 << (equihash_context_.collision_bits_length+1);
        // The indices that gave the solution - 2^k * (n / (k + 1) + 1) / 8
        equihash_context_.solution_size = (1 << equihash_context_.K) * (equihash_context_.N / (equihash_context_.K + 1) + 1) / 8;
  
        printf(": n %d, k %d\n",              equihash_context_.N, equihash_context_.K);                 //  200, 9
        printf(": collisionBitLength %d\n",   equihash_context_.collision_bits_length);   //   20
        printf(": collisionByteLength %d\n",  equihash_context_.collision_bytes_length);  //    3
        printf(": hashLength %d\n",           equihash_context_.hash_length);           //   30
        printf(": indicesPerHashOutput %d\n", equihash_context_.indices_per_hash_output); //    2
        printf(": hashOutput %d\n",           equihash_context_.hash_output);           //   50
        printf(": fullWidth %d\n",            equihash_context_.full_width);            // 1030
        printf(": initSize %d (memory %u)\n",
            equihash_context_.init_size, equihash_context_.init_size * equihash_context_.full_width); // 2097152, 2160066560
        sleep(2);
    }

    void EquihashGPUSolver::prepare_buffers()
    {
        cl::CommandQueue & queue = gpu_config_.get_device_queues()[0];
        cl_int zero = 0;
        // Create the hash table s
        table_buffer_ = cl::Buffer(
            gpu_config_.get_context(),
            CL_MEM_READ_WRITE,
            equihash_context_.init_size*equihash_context_.full_width*2
        );
        queue.enqueueFillBuffer(table_buffer_, zero, 0, 
            equihash_context_.init_size*equihash_context_.full_width*2
        );

        table_size_buffer_ = cl::Buffer(
            gpu_config_.get_context(),
            CL_MEM_READ_WRITE,
            sizeof(uint32_t)
        );
        queue.enqueueFillBuffer(table_size_buffer_, zero, 0, 
            sizeof(uint32_t));

        // TODO - Change this to a more reasonable buffer
        collision_table_buffer_ = cl::Buffer(
            gpu_config_.get_context(),
            CL_MEM_READ_WRITE,
            equihash_context_.init_size*equihash_context_.full_width*2
        );
        queue.enqueueFillBuffer(collision_table_buffer_, zero, 0, 
            equihash_context_.init_size*equihash_context_.full_width*2
        );
            
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
            CL_MEM_READ_WRITE,
            sizeof(BlakeDummy)
        );
        queue.enqueueFillBuffer(table_buffer_, zero, 0, 
            sizeof(BlakeDummy));

        // Construct the context buffer to be used, will be the same as the gpu structure
        context_buffer_ = cl::Buffer(
            gpu_config_.get_context(),
            CL_MEM_READ_ONLY,
            sizeof(EquihashGPUContext)
        );

        // Copy the context to the buffer
        queue.enqueueWriteBuffer(context_buffer_, true, 0, sizeof(EquihashGPUContext), &equihash_context_);
    }

    void EquihashGPUSolver::enqueue_and_run_hash_kernel(uint32_t nonce)
    {
        std::cout << "Starting to create hashes" << std::endl;
        std::vector<cl::CommandQueue> & device_queues = gpu_config_.get_device_queues();

        // TODO - Switch this to a more proper version
        blake2b_state dummy_s;
        blake2b_param P[1];
        memset(P, 0, sizeof(blake2b_param));
        P->fanout = 1;
        P->depth = 1;
        P->digest_length = (512 / equihash_context_.N) * equihash_context_.N / 8;
        memcpy(P->personal, "ZcashPoW", 8);
        *(uint32_t *)(P->personal +  8) = htole32(equihash_context_.N);
        *(uint32_t *)(P->personal + 12) = htole32(equihash_context_.K);
        blake2b_init_param(&dummy_s, P);
        blake2b_update(&dummy_s, (uint8_t*)equihash_context_.seed, SEED_SIZE*sizeof(uint32_t));
        BlakeDummy dummy;
        memcpy(dummy.hash_state, dummy_s.h, sizeof(uint64_t)*8);
        memcpy(dummy.buf, dummy_s.buf, dummy_s.buflen);
        dummy.t[0] = dummy_s.t[0];
        dummy.t[1] = dummy_s.t[1];
        dummy.buflen = dummy_s.buflen;

        device_queues[0].enqueueWriteBuffer(digest_buffer_, true, 0, sizeof(BlakeDummy), &dummy);
        
        size_t s = ((equihash_context_.init_size / equihash_context_.indices_per_hash_output) + 1)
                        / device_queues.size();
        
        cl::Kernel & hash_kernel = gpu_config_.get_equihash_hash_kernel();

        hash_kernel.setArg(0, context_buffer_);
        hash_kernel.setArg(1, table_buffer_);
        hash_kernel.setArg(2, digest_buffer_);
        hash_kernel.setArg(3, nonce);

        // Run the kernels
        for(size_t i=0;i<device_queues.size();i++)
        {
            device_queues[i].enqueueNDRangeKernel(hash_kernel, 
                                                 cl::NDRange(i*s),
                                                 cl::NDRange(s),
                                                 cl::NullRange);
            
            cl_int err = device_queues[i].flush();
            std::cout << EquihashGPUUtils::get_cl_errno(err) << std::endl;
        }

        // Wait for all to end
        std::cout << "Done enqueing, waiting for finish" << std::endl;
        for(size_t i=0;i<device_queues.size();i++)
        {
            cl_int err = device_queues[i].finish();
            std::cout << EquihashGPUUtils::get_cl_errno(err) << std::endl;
        }
        
        std::cout << "Finished creating hashes" << std::endl;
    }

    void EquihashGPUSolver::enqueue_and_run_coliision_detection_rounds_kernel()
    {
        std::vector<cl::CommandQueue> & device_queues = gpu_config_.get_device_queues();
        
        cl::Kernel collision_detection_kernel = gpu_config_.get_equihash_collision_detection_round_kernel();
     
        cl::Buffer & working_table = table_buffer_;
        cl::Buffer & collision_table = collision_table_buffer_;
        cl::Buffer & temp = table_buffer_;
        cl_int zero = 0;
        uint32_t current_table_size = equihash_context_.init_size;

        // Go over K rounds, each time swapping the buffers
        cl_int err;
        for(size_t i=0;i<equihash_context_.K && current_table_size > 0;i++)
        {
            uint32_t kernel_size_per_queue = current_table_size / device_queues.size();
            
            std::cout << "Starting Kernel Round " << i+1 << "/" << equihash_context_.K << std::endl;

            std::cout << "Restarting the collision table size" << std::endl;
            device_queues[0].enqueueFillBuffer(collision_table_size_buffer_, 
                                                zero, 0, sizeof(uint32_t));
            device_queues[0].flush();
            device_queues[0].finish();

            // Set the arguments
            collision_detection_kernel.setArg(0, context_buffer_);
            collision_detection_kernel.setArg(1, working_table);
            collision_detection_kernel.setArg(2, collision_table);
            collision_detection_kernel.setArg(3, collision_table_size_buffer_);
            collision_detection_kernel.setArg(4, current_table_size);
            collision_detection_kernel.setArg(5, (uint8_t)i);

            std::cout << "Running the kernels" << std::endl;
            for(size_t j=0;j<device_queues.size();j++)
            {
                size_t queue_offset = kernel_size_per_queue*j;
                err = device_queues[j].enqueueNDRangeKernel(collision_detection_kernel,
                                                      cl::NDRange(queue_offset),
                                                      cl::NDRange(kernel_size_per_queue),
                                                      cl::NullRange);
                std::cout << EquihashGPUUtils::get_cl_errno(err) << std::endl;
                err = device_queues[j].flush();
                std::cout << EquihashGPUUtils::get_cl_errno(err) << std::endl;
            }
            std::cout << "Waiting for kernels to end" << std::endl;
            for(size_t j=0;j<device_queues.size();j++)
            {
                err = device_queues[j].finish();
                std::cout << EquihashGPUUtils::get_cl_errno(err) << std::endl;
            }
            if(err)
            {
                std::cout << "Err occured on round, stopping" << std::endl;
                break;
            }

            // Swap the buffers and reset / copy the current size
            temp = working_table;
            working_table = collision_table;
            collision_table = temp;

            device_queues[0].enqueueReadBuffer(collision_table_size_buffer_, true, 0, sizeof(uint32_t), &current_table_size);
            std::cout << "Round " << i+1 << " finished, collision size = " << current_table_size << std::endl;
        }

        std::cout << "Finished Rounds" << std::endl;
        // Keep the actual working table pointer for the solutions kernel
        table_buffer_view_ = &working_table;
    }

    void EquihashGPUSolver::enqueue_and_run_solutions_kernel()
    {
        std::vector<cl::CommandQueue> & device_queues = gpu_config_.get_device_queues();
        
        uint32_t size;
        cl_int zero = 0;
        uint32_t single_solution_size = sizeof(uint32_t) * (equihash_context_.K + 1);
        // cl::copy(collision_table_size_buffer_, &size, (&size) + sizeof(uint32_t));
        device_queues[0].enqueueReadBuffer(collision_table_size_buffer_, true, 0, sizeof(uint32_t), &size);

        // The solutions buffer can be max the current table size/2
        solutions_buffer_ = cl::Buffer(
            gpu_config_.get_context(),
            CL_MEM_READ_WRITE,
            (size/2)*single_solution_size
        );

        device_queues[0].enqueueFillBuffer(solutions_buffer_, zero, 0, (size/2)*single_solution_size);

        // TODO
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
            enqueue_and_run_coliision_detection_rounds_kernel();

            // Perform the final round and get the solutions
            // enqueue_and_run_solutions_kernel();
            return Proof();
        }

        return Proof();
    }

    bool EquihashGPUSolver::verify_proof(const Proof & proof)
    {

    }
}