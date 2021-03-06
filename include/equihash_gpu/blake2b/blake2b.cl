#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#pragma OPENCL EXTENSION cl_nv_pragma_unroll

#define OUT_HASH_LENGTH 64 // Only 64 is supported for now
#define HASH_BLOCK_SIZE 128
typedef unsigned char  uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int   uint32_t;
typedef unsigned long  uint64_t;
typedef          char   int8_t;
typedef          short  int16_t;
typedef          int    int32_t;
typedef          long   int64_t;

#define B2B_GET64(p)                            \
    (((uint64_t) ((uint8_t *) (p))[0]) ^        \
    (((uint64_t) ((uint8_t *) (p))[1]) << 8) ^  \
    (((uint64_t) ((uint8_t *) (p))[2]) << 16) ^ \
    (((uint64_t) ((uint8_t *) (p))[3]) << 24) ^ \
    (((uint64_t) ((uint8_t *) (p))[4]) << 32) ^ \
    (((uint64_t) ((uint8_t *) (p))[5]) << 40) ^ \
    (((uint64_t) ((uint8_t *) (p))[6]) << 48) ^ \
    (((uint64_t) ((uint8_t *) (p))[7]) << 56))


#define ROTR64(x, y)  (((x) >> (y)) ^ ((x) << (64 - (y))))

typedef struct
{
  uint64_t hash_state[8];
  uint8_t  buf[2*HASH_BLOCK_SIZE];
  uint32_t buflen;
  uint64_t t[2];
} blake2b_state;

constant uint8_t blake2b_sigma[12][16] =
{
  {  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15 } ,
  { 14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3 } ,
  { 11,  8, 12,  0,  5,  2, 15, 13, 10, 14,  3,  6,  7,  1,  9,  4 } ,
  {  7,  9,  3,  1, 13, 12, 11, 14,  2,  6,  5, 10,  4,  0, 15,  8 } ,
  {  9,  0,  5,  7,  2,  4, 10, 15, 14,  1, 11, 12,  6,  8,  3, 13 } ,
  {  2, 12,  6, 10,  0, 11,  8,  3,  4, 13,  7,  5, 15, 14,  1,  9 } ,
  { 12,  5,  1, 15, 14, 13,  4, 10,  0,  7,  6,  3,  9,  2,  8, 11 } ,
  { 13, 11,  7, 14, 12,  1,  3,  9,  5,  0, 15,  4,  8,  6,  2, 10 } ,
  {  6, 15, 14,  9, 11,  3,  0,  8, 12,  2, 13,  7,  1,  4, 10,  5 } ,
  { 10,  2,  8,  4,  7,  6,  1,  5, 15, 11,  9, 14,  3, 12, 13 , 0 } ,
  {  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15 } ,
  { 14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3 }
};

void blake2b_copy_glob(private uint8_t * dest, global uint8_t * src, uint32_t s)
{
  uint32_t i=0;
  for(i=0;i<s;i++)
  {
    dest[i] = src[i];
  }
}

void blake2b_copy_glob2(global uint8_t * dest, private uint8_t * src, uint32_t s)
{
  uint32_t i=0;
  for(i=0;i<s;i++)
  {
    dest[i] = src[i];
  }
}

void blake2b_copy_priv(private uint8_t * dest, private uint8_t * src, uint32_t s)
{
  uint32_t i=0;
  for(i=0;i<s;i++)
  {
    dest[i] = src[i];
  }
}

void blake2b_memset(private uint8_t * buf, uint8_t pat, uint32_t s)
{
  uint32_t i=0;
  for(i=0;i<s;i++)
  {
    buf[i] = pat;
  }
}

void blake2b_increment(private blake2b_state * state, uint32_t inc)
{
  state->t[0] += inc;
  state->t[1] += ( state->t[0] < inc );
}

// Perform the actual compression mixin on the given work vector and words
// Places the new state on the given current state
void blake2b_perform_compression(private blake2b_state * state, 
                                 private uint64_t * v, 
                                 private const uint64_t * m)
{
  private uint8_t i;

#define MIX(a, b, c, d, x, y) {   \
    v[a] = v[a] + v[b] + x;         \
    v[d] = ROTR64(v[d] ^ v[a], 32); \
    v[c] = v[c] + v[d];             \
    v[b] = ROTR64(v[b] ^ v[c], 24); \
    v[a] = v[a] + v[b] + y;         \
    v[d] = ROTR64(v[d] ^ v[a], 16); \
    v[c] = v[c] + v[d];             \
    v[b] = ROTR64(v[b] ^ v[c], 63); \
  }

#define ROUND(r) \
	MIX(0, 4, 8,  12, m[blake2b_sigma[r][0]],  m[blake2b_sigma[r][1]]); \
  MIX(1, 5, 9,  13, m[blake2b_sigma[r][2]],  m[blake2b_sigma[r][3]]); \
  MIX(2, 6, 10, 14, m[blake2b_sigma[r][4]],  m[blake2b_sigma[r][5]]); \
  MIX(3, 7, 11, 15, m[blake2b_sigma[r][6]],  m[blake2b_sigma[r][7]]); \
  MIX(0, 5, 10, 15, m[blake2b_sigma[r][8]],  m[blake2b_sigma[r][9]]); \
  MIX(1, 6, 11, 12, m[blake2b_sigma[r][10]], m[blake2b_sigma[r][11]]); \
  MIX(2, 7, 8,  13, m[blake2b_sigma[r][12]], m[blake2b_sigma[r][13]]); \
  MIX(3, 4, 9,  14, m[blake2b_sigma[r][14]], m[blake2b_sigma[r][15]]); 

  // Perform the mix 12 times
  ROUND(0);  
	ROUND(1);
	ROUND(2);
	ROUND(3);
	ROUND(4);
	ROUND(5);
	ROUND(6);
	ROUND(7);
	ROUND(8);
	ROUND(9);
	ROUND(10);
	ROUND(11);

#undef MIX
#undef ROUND

  // Store the resulting mixed hash in the current state 
  #pragma unroll 8
  for(i=0;i<8;++i)
  {
    state->hash_state[i] ^= v[i] ^ v[i + 8];
  } 
}

// Compression of the given chunk based on the current hash state
void blake2b_compress_chunk(private blake2b_state * state, 
                            private const uint8_t * chunk,
                            private uint8_t chunk_size,
                            private bool is_last_chunk)
{
  // Create the work vector and the 64 bit words
  private uint64_t work_vector[16];
  private uint64_t words[16] = {0};
  private uint8_t i;

  // Perform big to little endian on each word
  // Roll it out to max 16 (128)
  // #pragma unroll 16
  for(i=0;i<(chunk_size/8);++i)
  {
    words[i] = *((private uint64_t*)&chunk[i*8]);
  } 

  // Initialize the work vector with the current hash
  // Roll it out
  #pragma unroll 8
  for(i=0;i<8;++i)
  {
    work_vector[i] = state->hash_state[i];
  } 

  // Initialize the rest of the vector from the initializing vector
  work_vector[8] =  0x6A09E667F3BCC908;
  work_vector[9] =  0xBB67AE8584CAA73B;
  work_vector[10] = 0x3C6EF372FE94F82B;
  work_vector[11] = 0xA54FF53A5F1D36F1;
  work_vector[12] = 0x510E527FADE682D1;
  work_vector[13] = 0x9B05688C2B3E6C1F;
  work_vector[14] = 0x1F83D9ABFB41BD6B;
  work_vector[15] = 0x5BE0CD19137E2179;

  // Mix byte counter, Only support 64bit for now [TODO - Change bytes_compressed to two 64bit]
  work_vector[12] ^= state->t[0];
  work_vector[13] ^= state->t[1];

  if(is_last_chunk)
  {
    work_vector[14] ^= 0xFFFFFFFFFFFFFFFF;
  } 

  // Perform the mix compression
  blake2b_perform_compression(state, work_vector, words);
}

// Finalizes the hash and output it to the output buffer on big endian
void blake2b_finalize_hash_priv(private blake2b_state * state, private uint8_t * out_hash, uint8_t out_hash_size)
{
  uint32_t i;
  if(state->buflen > HASH_BLOCK_SIZE)
  {
    blake2b_increment(state, HASH_BLOCK_SIZE);
    blake2b_compress_chunk(state, state->buf , HASH_BLOCK_SIZE, false);
    state->buflen -= HASH_BLOCK_SIZE;
    blake2b_copy_priv(state->buf, state->buf + HASH_BLOCK_SIZE, HASH_BLOCK_SIZE);
  }

  blake2b_increment(state, state->buflen);
  blake2b_memset(state->buf + state->buflen, 0, 2*HASH_BLOCK_SIZE - state->buflen);
  blake2b_compress_chunk(state, state->buf , HASH_BLOCK_SIZE, true);
  blake2b_copy_priv(out_hash, (private uint8_t*)state->hash_state, out_hash_size);
}

// Finalizes the hash and output it to the output buffer on big endian
void blake2b_finalize_hash(private blake2b_state * state, global uint8_t * out_hash, uint8_t out_hash_size)
{
  uint32_t i;
  if(state->buflen > HASH_BLOCK_SIZE)
  {
    blake2b_increment(state, HASH_BLOCK_SIZE);
    blake2b_compress_chunk(state, state->buf , HASH_BLOCK_SIZE, false);
    state->buflen -= HASH_BLOCK_SIZE;
    blake2b_copy_priv(state->buf, state->buf + HASH_BLOCK_SIZE, HASH_BLOCK_SIZE);
  }

  blake2b_increment(state, state->buflen);
  blake2b_memset(state->buf + state->buflen, 0, 2*HASH_BLOCK_SIZE - state->buflen);
  blake2b_compress_chunk(state, state->buf , HASH_BLOCK_SIZE, true);
  blake2b_copy_glob2(out_hash, (private uint8_t*)state->hash_state, out_hash_size);
}

void blake2b_update(private blake2b_state * state,
                    global uint8_t * in_message, 
                    private uint32_t in_len)
{
  // Store bytes remaining to be more efficient instead of calculating each time
  global const uint8_t * chunk;
  private uint32_t left, fill;

  // Run in 128 bytes chunk and compress each chunk
  while(in_len > 0)
  {
    left = state->buflen;
    fill = 2*HASH_BLOCK_SIZE - left;

    if(in_len > fill)
    {
      blake2b_copy_glob(state->buf + left, in_message, fill);
      state->buflen += fill;
      blake2b_increment(state, HASH_BLOCK_SIZE);
      blake2b_compress_chunk(state, state->buf, HASH_BLOCK_SIZE, false);
      blake2b_copy_priv(state->buf, state->buf + HASH_BLOCK_SIZE, HASH_BLOCK_SIZE);
      state->buflen -= HASH_BLOCK_SIZE;
      in_message += fill;
      in_len -= fill;
    }
    else
    {
      blake2b_copy_glob(state->buf + left, in_message, in_len );
      state->buflen += ( uint32_t ) in_len; // Be lazy, do not compress
      in_message += in_len;
      in_len -= in_len;
    }
  }
}

void blake2b_update_priv(private blake2b_state * state,
                    private uint8_t * in_message, 
                    private uint32_t in_len)
{
  // Store bytes remaining to be more efficient instead of calculating each time
  private const uint8_t * chunk;
  private uint32_t left, fill;

  // Run in 128 bytes chunk and compress each chunk
  while(in_len > 0)
  {
    left = state->buflen;
    fill = 2*HASH_BLOCK_SIZE - left;

    if(in_len > fill)
    {
      blake2b_copy_priv(state->buf + left, in_message, fill);
      state->buflen += fill;
      blake2b_increment(state, HASH_BLOCK_SIZE);
      blake2b_compress_chunk(state, state->buf, HASH_BLOCK_SIZE, false);
      blake2b_copy_priv(state->buf, state->buf + HASH_BLOCK_SIZE, HASH_BLOCK_SIZE);
      state->buflen -= HASH_BLOCK_SIZE;
      in_message += fill;
      in_len -= fill;
    }
    else
    {
      blake2b_copy_priv(state->buf + left, in_message, in_len );
      state->buflen += ( uint32_t ) in_len; // Be lazy, do not compress
      in_message += in_len;
      in_len -= in_len;
    }
  }
}

// Initialization vector for blake2b hash state
void blake2b_init(private blake2b_state * state)
{
  state->hash_state[0] = 0x6A09E667F3BCC908 ^ 0x01010000 ^ OUT_HASH_LENGTH;
  state->hash_state[1] = 0xBB67AE8584CAA73B;
  state->hash_state[2] = 0x3C6EF372FE94F82B;
  state->hash_state[3] = 0xA54FF53A5F1D36F1;
  state->hash_state[4] = 0x510E527FADE682D1;
  state->hash_state[5] = 0x9B05688C2B3E6C1F;
  state->hash_state[6] = 0x1F83D9ABFB41BD6B;
  state->hash_state[7] = 0x5BE0CD19137E2179;

  state->t[0] = 0;
  state->t[1] = 0;
  state->buflen = 0;
}

// GPU Hash impl of blake2b, receives as input:
// in_message - The message to hash
// out_hash - The buffer to store the resulting hash
// message_size - The size of the given message to hash
// out_hash_size - The size of the output hash, should not exceed OUT_HASH_SIZE(64)
// Notes:
// Extra hash key is not supported yet
kernel void blake2b_gpu_hash(global const uint8_t * in_message, 
                             global uint8_t * out_hash, 
                             const uint32_t message_size,
                             const uint32_t out_hash_size)
{
  // Create the state to be used as local memory
  private blake2b_state state;

  // Initialize the state
  blake2b_init(&state);

  // Perform the update on the given message
  blake2b_update(&state, in_message, message_size);

  // Finalize the hash and write it out
  blake2b_finalize_hash(&state, out_hash, out_hash_size);
}