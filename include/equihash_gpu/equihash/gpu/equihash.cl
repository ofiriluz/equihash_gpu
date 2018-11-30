#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#pragma OPENCL EXTENSION cl_nv_pragma_unroll

#include "/home/ofir/Desktop/Equihash/equihash_gpu/include/equihash_gpu/blake2b/blake2b.cl"

#define SEED_SIZE 4
#define MAX_BUCKET_AMOUNT 5
#define ENDIAN_SWAP(n) ((rotate(n & 0x00FF00FF, 24U)|(rotate(n, 8U) & 0x00FF00FF)))
typedef struct 
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

} equihash_context;

void compress_array(global uint8_t * in, 
                  const uint32_t size_in, 
                  global uint8_t * out, 
                  const uint32_t size_out,
                  const uint32_t bit_len,
                  const uint32_t byte_pad)
{

    // TODO
    // Make sure that the bit len is at least 8 bits
    // And make sure 32 bytes can hold it


    // Calculate the actual width to produce, padded to bytes
    private const uint32_t in_width = (bit_len + 7) / 8 + byte_pad;

    // Calculate the bits mask, to use to take the blocks
    private const uint32_t bit_len_mask = ((uint32_t)1 << bit_len) - 1;

    // Start accumilating bits until we reached collision bits amount
    // Once reached, add it to the blocks masked to big endian 
    uint32_t acc_bits = 0;
    uint32_t acc_value = 0;
    uint32_t j = 0;
    uint32_t i;
    uint32_t x;
 
    for(i=0;i<size_in;i++)
    {
        if(acc_bits < 8)
        {
            acc_value = acc_value << bit_len;
            for (x = byte_pad; x < in_width; x++) {
				acc_value = acc_value | (
					(
					    // Apply bit_len_mask across byte boundaries
					    in[j + x] & ((bit_len_mask >> (8 * (in_width - x - 1))) & 0xFF)
					) << (8 * (in_width - x - 1))); // Big-endian
			}
			j += in_width;
			acc_bits += bit_len;
        }

		acc_bits -= 8;
		out[i] = (acc_value >> acc_bits) & 0xFF;
    }    
}

void expand_array(private uint8_t * in, 
                  const uint32_t size_in, 
                  global uint8_t * out, 
                  const uint32_t size_out,
                  const uint32_t bit_len,
                  const uint32_t byte_pad)
{
    // TODO
    // Make sure that the bit len is at least 8 bits
    // And make sure 32 bytes can hold it


    // Calculate the actual width to produce, padded to bytes
    private const uint32_t out_width = (bit_len + 7) / 8 + byte_pad;

    // Calculate the bits mask, to use to take the blocks
    private const uint32_t bit_len_mask = ((uint32_t)1 << bit_len) - 1;

    // Start accumilating bits until we reached collision bits amount
    // Once reached, add it to the blocks masked to big endian 
    uint32_t acc_bits = 0;
    uint32_t acc_value = 0;
    uint32_t j = 0;
    uint32_t i, k;
    uint32_t x;
 
    for(i=0;i<size_in;i++)
    {
        acc_value = ((acc_value << 8) | in[i]);
    	acc_bits += 8;
 
        if (acc_bits >= bit_len)
        {
            acc_bits -= bit_len;
            for (x = 0; x < byte_pad; x++) 
            {
                out[j + x] = 0;
            }
            for (x = byte_pad; x < out_width; x++) 
            {
                out[j + x] = (
                    // Big-endian
                    acc_value >> (acc_bits + (8 * (out_width - x - 1)))
                ) & (
                    // Apply bit_len_mask across byte boundaries
                    (bit_len_mask >> (8 * (out_width - x - 1))) & 0xFF
                );
	        }

	        j += out_width;
	    }
    }
}

bool has_collision(global uint8_t * a, global uint8_t * b, uint32_t bytes)
{
    for (;bytes--;a++,b++) 
    {
        if ((*a) != (*b))
        {
            return false;
        }
    }
    return true;
}

bool distinct_indices(global uint8_t * a, global uint8_t * b, const uint32_t len, const uint32_t len_indices)
{
    private uint32_t i,j;
    for(i=0;i<len_indices;i++)
    {
        for(j=0;j<len_indices;j++)
        {
            if(has_collision(a + len + i*(sizeof(uint32_t)), b + len + j*(sizeof(uint32_t)), sizeof(uint32_t)))
            {
                return false;
            }
        }
    }

    return true;
}

bool indices_before(global uint8_t * a, global uint8_t * b, uint32_t len_indices)
{
    for (;len_indices--;a++,b++) 
    {
        if ((*a) < (*b))
        {
            return true;
        }
    }
    return false;
}

void copy_indices(global uint8_t * dest, global uint8_t * src, const uint32_t len_indices)
{
    private uint32_t i;
    #pragma unroll
    for(i=0;i<len_indices*sizeof(uint32_t);i++)
    {
        dest[i] = src[i];
    }
}

void combine_rows(global uint8_t * dest, 
                  global uint8_t * a, 
                  global uint8_t * b, 
                  const uint32_t len, 
                  const uint32_t len_indices, 
                  const uint32_t trim)
{
    private uint32_t i;

    #pragma unroll
    for (i = trim; i < len; i++)
    {
        dest[i-trim] = a[i] ^ b[i];
    }
    
    if(indices_before(a+len, b+len, len_indices))
    {
        copy_indices(dest + len - trim, a + len, len_indices);
        copy_indices(dest + len - trim + len_indices, b + len, len_indices);
    }
    else
    {
        copy_indices(dest + len - trim, b + len, len_indices);
        copy_indices(dest + len - trim + len_indices, a + len, len_indices);
    }
}

void big_endian_index_to_array(uint32_t i, global uint8_t * array)
{   
    // private uint32_t le_i = ENDIAN_SWAP(i);
    private uint32_t j;
    #pragma unroll
    for(j=0;j<sizeof(i);j++)
    {
        array[j] = (i << (8*j));
    }
}

void store_solution_indices(global uint8_t * hash, 
                            const uint32_t len, 
                            const uint32_t indices_len, 
                            const uint32_t bits_len, 
                            global uint8_t * row, 
                            const uint32_t max_len)
{
    uint32_t min_len = (bits_len + 1) * indices_len / (8 * sizeof(uint32_t));
    uint32_t padding = sizeof(uint32_t) - ((bits_len + 1 ) + 7 ) / 8;

    compress_array(hash + len, indices_len, row, min_len, bits_len + 1, padding);
}

bool is_zero(global uint8_t * a, uint32_t len)
{
    for(;len--;a++)
    {
        if(a[len] != 0)
        {
            return false;
        }
    }

    return true;
}

kernel void equihash_initialize_hash(global equihash_context * context,
                                     global uint8_t * hash_table,
                                     global blake2b_state * s,
                                     const uint32_t nonce)
{
    // The index to be used is the global work index
    private uint8_t digest[HASH_BLOCK_SIZE];
    private uint32_t index = get_global_id(0);
    private uint32_t le_i = ENDIAN_SWAP(index);
    private uint8_t i, j, k;
    private uint8_t amount_to_add;
    private uint32_t array_index;
    global uint8_t * start_row;
    global uint8_t * current_row;

    private blake2b_state s_priv = *s;
    blake2b_update_priv(&s_priv, (private uint8_t*)&index, sizeof(index));
    blake2b_finalize_hash_priv(&s_priv, (private uint64_t*)digest, context->hash_output);
    
    // Get the pointer to the first row in this set of rows
    start_row = hash_table + 
                (context->full_width*index)*context->indices_per_hash_output;

    // Go over each part of the hash and add it to the table in the fitting rows
    // The amount is limited to the table size
    amount_to_add = min(context->indices_per_hash_output, 
                    context->init_size - index*context->indices_per_hash_output);

    #pragma unroll
    for(i=0;i<amount_to_add;i++)
    {
        // Get the current row
        current_row = start_row + context->full_width*i;
        
        // Split the block and put it on the fitting row in the hash table
        expand_array(digest + (i*context->N/8), context->N/8, 
                     current_row, 
                     context->hash_length, context->collision_bits_length, 0);

        array_index = (index*context->indices_per_hash_output)+i;

        // Add the index to the row
        big_endian_index_to_array(array_index,
                                 current_row + context->hash_length);
    } 
}

kernel void equihash_collision_detection_round(constant equihash_context * context,
                                               global uint8_t * working_table,
                                               global uint8_t * collision_table,
                                               global uint32_t * collision_table_size,
                                               const uint32_t working_table_size,
                                               const uint8_t collision_round)
{
    // printf("TABLE S = %d\n", working_table_size);
    private uint32_t row_index = get_global_id(0);
    global uint8_t * row = working_table + (context->full_width*row_index);
    global uint8_t * selected_row;
    global uint8_t * target_row;
    private bool t, t2;
    private uint32_t i, j;
    private uint32_t target_row_index;

    // Calculate the current hash length and indices length for this round
    private uint32_t hash_len = context->hash_length - 
                                (collision_round*context->collision_bytes_length);
    
    // Length in bytes
    private uint32_t indices_len = (1 << collision_round)*sizeof(uint32_t);    

    // We go over the working row up until the end and find collision
    // This is a naive solution but paralled
    #pragma unroll
    for(i=row_index+1;i<working_table_size;i++)
    {
        selected_row = working_table + (context->full_width*i);
        if(has_collision(row, selected_row, context->collision_bytes_length)
            &&
           distinct_indices(row, selected_row, hash_len, indices_len/sizeof(uint32_t)))
        {
            // Acquire the index 
            target_row_index = atomic_inc(collision_table_size);
            target_row = collision_table + context->full_width*target_row_index;

            // Combine the rows into the collision table
            combine_rows(target_row, row, selected_row, hash_len, indices_len, context->collision_bytes_length);
        }
    }    
}   

kernel void equihash_solutions_detection(global equihash_context * context, 
                                         global uint8_t * hash_table,
                                         global uint8_t * collision_table,
                                         global uint8_t * solutions_table,
                                         global uint32_t * collision_table_size,
                                         global uint32_t * solutions_table_size,
                                         const uint32_t hash_table_size)
{

    private uint32_t row_index = get_global_id(0);
    global uint8_t * row = working_table + (context->full_width*row_index);
    global uint8_t * selected_row;
    global uint8_t * target_row;
    private uint32_t i;

    private uint32_t hash_len = context->hash_length - 
                                (context->K*context->collision_bytes_length);
    private uint32_t indices_len = (1 << context->K) * sizeof(uint32_t); 

    for(i=row_index+1;i<working_table_size;i++)
    {
        selected_row = working_table + (context->full_width*i);
        if(has_collision(row, selected_row, hash_len))
        {
            // Acquire the index 
            target_row_index = atomic_inc(collision_table_size);
            target_row = collision_table + context->full_width*target_row_index;

            combine_rows(target_row, row, selected_row, hash_len, indices_len, 0);

            if(is_zero(target_row, hash_len) 
               && 
              distinct_indices(row, selected_row, hash_len, indices_len))
            {
                // Solution is found, acquire its index atomiclly
                solution_index = atomic_inc(solutions_table_size);
                solution = solutions_table * sizeof(uint32_t)*(context->K+1)*solution_index;

                // Store the uncompressed solution
                store_solution_indices(target_row, hash_len, 2 * indices_len, context->collision_bits_length, solution, sizeof(uint32_t)*(context->K+1));
            }
        }
    }
}