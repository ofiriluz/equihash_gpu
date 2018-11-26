#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#pragma OPENCL EXTENSION cl_nv_pragma_unroll

#include "/home/ofir/Desktop/Equihash/equihash_gpu/include/equihash_gpu/blake2b/blake2b.cl"

#define SEED_SIZE 4
#define MAX_BUCKET_AMOUNT 5

typedef struct 
{
    uint16_t N, K, S; // S = N/(K+1), calculated once instead of each thread
    uint32_t seed[SEED_SIZE + 2];
    uint32_t bucket_size;
} equihash_context;

kernel void equihash_initialize_hash(global equihash_context * context,
                                     global uint32_t * buckets,
                                     const uint32_t nonce)
{   
    // The index to be used is the global work index
    private uint32_t index = get_global_id(0);
    private uint32_t bucket_index;
    private uint32_t hash[8]; // Assume that 32 bytes are enough (K shouldnt get to ~31)
    private uint8_t i;
    global uint32_t * chosen_bucket;
    global uint32_t * bucket_slot;

    // Fill the context seed nonce and index
    context->seed[SEED_SIZE] = nonce;
    context->seed[SEED_SIZE+1] = index;

    // Calculate the blake hash
    blake2b((global uint8_t*)context->seed, (uint64_t*)hash, sizeof(context->seed), sizeof(hash));

    // Calculate the bucket index
    bucket_index = hash[0] >> (32 - context->S);

    // Get the pointer to the bucket start
    chosen_bucket = &(buckets[bucket_index*context->bucket_size]);

    // Check if the count has reached the max, if it did, do not add
    if(chosen_bucket[0] < MAX_BUCKET_AMOUNT)
    {
        // Increase the bucket count atomiclly
        atomic_inc((uint16_t*)(chosen_bucket));

        // Get the reference to the bucket slot to add
        bucket_slot = &(chosen_bucket[sizeof(uint16_t) 
                                    + chosen_bucket[0]*sizeof(uint32_t)*(context->K+1)]);

        // Set the added index on the bucket as a reference
        bucket_slot[0] = index;

        // Fill the bucket bit blocks
        for(i=1;i<(equihash_context->K+1);i++)
        {
            bucket_slot[i] = hash[i] >> (32 - context->S);
        }
    }
}