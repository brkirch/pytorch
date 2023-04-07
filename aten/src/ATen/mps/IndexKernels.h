#pragma once

namespace at {
namespace mps {

#define GET_IDX_TEMPLATE                                     \
"static inline uint3 get_idx(                              " \
"  uint tid,                                               " \
"  constant uint * iter_shape,                             " \
"  const uint num_dimensions,                              " \
"  constant packed_uint3 * strides) {{                     " \
"  uint3 data_offsets = 0;                                 " \
"  uint32_t idx = tid;                                     " \
"  for (uint32_t dim = 0; dim < num_dimensions; dim++) {{  " \
"      uint32_t remainder = idx % iter_shape[dim];         " \
"      idx /= iter_shape[dim];                             " \
"      data_offsets += remainder * strides[dim];           " \
"  }}                                                      " \
"  return data_offsets;                                    " \
"}}"

static const char * indexing_metal_shaders = GET_IDX_TEMPLATE
R"INDEX_METAL(
#include <metal_stdlib>
#include <metal_atomic>

using namespace metal;

#if __METAL_VERSION__ < 300
struct IndexAB {
    // Allow up to 16 indices
    metal::array<constant void *, 16>  indexArray [[ id(0) ]];
};
#else
struct IndexAB {
    constant int64_t* indexArray;
};
#endif

template<typename T>
kernel void index_select(
#if __METAL_VERSION__ >= 300
    constant IndexAB  * indexAB           [[buffer(0)]],
#else
    constant IndexAB  & indexAB           [[buffer(0)]],
#endif
    constant void     * indexSizes        [[buffer(1)]],
    constant void     * indexStrides      [[buffer(2)]],
    constant void     * inputData         [[buffer(4)]],
    device   void     * outputData        [[buffer(5)]],
    constant uint32_t & num_indices       [[buffer(6)]],
    constant uint     * iter_shape        [[buffer(7)]],
    constant uint     & num_dimensions    [[buffer(8)]],
    constant packed_uint3 * strides   [[buffer(9)]],

    uint thread_index [[thread_position_in_grid]]) {

    uint3 offsets = get_idx(thread_index, iter_shape, num_dimensions, strides);

    constant int64_t * index_sizes   = (constant int64_t *)indexSizes;
    constant int64_t * index_strides = (constant int64_t *)indexStrides;
    int64_t offset = 0;
    for (uint32_t i = 0; i < num_indices; i++) {
#if __METAL_VERSION__ >= 300
        constant int64_t* indexArray = indexAB[i].indexArray;
#else
        constant int64_t* indexArray = (constant int64_t*)indexAB.indexArray[i];
#endif
        int64_t index = indexArray[offsets.z / sizeof(int64_t)];
        if (index < 0) {
            index += index_sizes[i];
        }
        offset += index * index_strides[i];
     }
    device T * out = (device T*)((device char*)outputData + offsets.x);
    constant T * in  = (constant T*)((constant char*)inputData  + offsets.y + offset);
    *out = *in;
}

template<typename T>
kernel void index_put(
#if __METAL_VERSION__ >= 300
    constant IndexAB  * indexAB           [[buffer(0)]],
#else
    constant IndexAB  & indexAB           [[buffer(0)]],
#endif
    constant void     * indexSizes        [[buffer(1)]],
    constant void     * indexStrides      [[buffer(2)]],
    constant void     * inputData         [[buffer(4)]],
    device   void     * outputData        [[buffer(5)]],
    constant uint32_t & num_indices       [[buffer(6)]],

    constant uint  * iter_shape       [[buffer(7)]],
    constant uint & num_dimensions    [[buffer(8)]],
    constant packed_uint3 * strides   [[buffer(9)]],

    uint thread_index [[thread_position_in_grid]]) {

    uint3 offsets = get_idx(thread_index, iter_shape, num_dimensions, strides);


    constant int64_t * index_sizes   = (constant int64_t *)indexSizes;
    constant int64_t * index_strides = (constant int64_t *)indexStrides;
    int64_t offset = 0;
    for (uint32_t i = 0; i < num_indices; i++) {
#if __METAL_VERSION__ >= 300
        constant int64_t* indexArray = indexAB[i].indexArray;
#else
        constant int64_t* indexArray = (constant int64_t*)indexAB.indexArray[i];
#endif
        int64_t index = indexArray[offsets.z / sizeof(int64_t)];

        if (index < 0) {
            index += index_sizes[i];
        }
        offset += index * index_strides[i];
     }
    device T * out = (device T*)((device char*)outputData + offsets.x + offset);
    constant T * in  = (constant T*)((constant char*)inputData  + offsets.y);
    *out = *in;
}

#if __METAL_VERSION__ < 300
#define REGISTER_INDEX_OP(DTYPE_SIZE, DTYPE, INDEX_OP_TYPE)     \
template                                                        \
[[host_name("index_" #INDEX_OP_TYPE "_" #DTYPE_SIZE)]]          \
kernel void index_ ## INDEX_OP_TYPE<DTYPE>(                     \
    constant IndexAB  & indexAB           [[buffer(0)]],        \
    constant void     * indexSizes        [[buffer(1)]],        \
    constant void     * indexStrides      [[buffer(2)]],        \
    constant void     * inputData         [[buffer(4)]],        \
    device   void     * outputData        [[buffer(5)]],        \
    constant uint32_t & num_indices       [[buffer(6)]],        \
    constant uint     * iter_shape        [[buffer(7)]],        \
    constant uint     & num_dimensions    [[buffer(8)]],        \
    constant packed_uint3 * strides       [[buffer(9)]],        \
    uint thread_index [[thread_position_in_grid]]);
#else
#define REGISTER_INDEX_OP(DTYPE_SIZE, DTYPE, INDEX_OP_TYPE)     \
template                                                        \
[[host_name("index_" #INDEX_OP_TYPE "_" #DTYPE_SIZE)]]          \
kernel void index_ ## INDEX_OP_TYPE<DTYPE>(                     \
    constant IndexAB  * indexAB          [[buffer(0)]],         \
    constant void     * indexSizes       [[buffer(1)]],         \
    constant void     * indexStrides     [[buffer(2)]],         \
    constant void     * inputData        [[buffer(4)]],         \
    device   void     * outputData       [[buffer(5)]],         \
    constant uint32_t & num_indices      [[buffer(6)]],         \
    constant uint     * iter_shape       [[buffer(7)]],         \
    constant uint     & num_dimensions   [[buffer(8)]],         \
    constant packed_uint3 * strides      [[buffer(9)]],         \
    uint thread_index [[thread_position_in_grid]]);
#endif

#define REGISTER_INDEX_OP_ALL_DTYPES(INDEX_OP_TYPE)     \
    REGISTER_INDEX_OP(8bit,  char,  INDEX_OP_TYPE);     \
    REGISTER_INDEX_OP(16bit, short, INDEX_OP_TYPE);     \
    REGISTER_INDEX_OP(32bit, int,   INDEX_OP_TYPE);     \
    REGISTER_INDEX_OP(64bit, long,  INDEX_OP_TYPE);

REGISTER_INDEX_OP_ALL_DTYPES(select);
REGISTER_INDEX_OP_ALL_DTYPES(put);

kernel void kernel_index_offsets(constant packed_uint3 * strides         [[buffer(0)]],
                                 device uint3          * data_offsets    [[buffer(1)]],
                                 constant uint         * iter_shape      [[buffer(2)]],
                                 constant uint         & num_dimensions  [[buffer(3)]],
                                 constant uint         & num_offsets     [[buffer(4)]],
                                 uint thread_index [[thread_position_in_grid]]) {
    data_offsets[thread_index] = 0;
    uint32_t idx = thread_index;
    for (uint32_t dim = 0; dim < num_dimensions; dim++) {
        uint32_t remainder = idx % iter_shape[dim];
        idx /= iter_shape[dim];

        data_offsets[thread_index] += remainder * strides[dim];
    }
}

template<typename T, typename E>
kernel void index_put_accumulate_native_dtypes(
#if __METAL_VERSION__ >= 300
    constant IndexAB  * indexAB        [[buffer(0)]],
#else
    constant IndexAB  & indexAB        [[buffer(0)]],
#endif
    constant void     * indexSizes     [[buffer(1)]],
    constant void     * indexStrides   [[buffer(2)]],
    constant void     * inputData      [[buffer(4)]],
    device void       * outputData     [[buffer(5)]],
    constant uint32_t & num_indices    [[buffer(6)]],
    constant uint     * iter_shape     [[buffer(7)]],
    constant uint     & num_dimensions [[buffer(8)]],
    constant packed_uint3 * strides    [[buffer(9)]],
    uint thread_index [[thread_position_in_grid]]) {
    uint3 offsets = get_idx(thread_index, iter_shape, num_dimensions, strides);
    constant int64_t * index_sizes   = (constant int64_t *)indexSizes;
    constant int64_t * index_strides = (constant int64_t *)indexStrides;
    int64_t offset = 0;
    for (uint32_t i = 0; i < num_indices; i++) {
#if __METAL_VERSION__ >= 300
        constant int64_t* indexArray = indexAB[i].indexArray;
#else
        constant int64_t* indexArray = (constant int64_t*)indexAB.indexArray[i];
#endif
        int64_t index = indexArray[offsets.z / sizeof(int64_t)];
        if (index < 0) {
            index += index_sizes[i];
        }
        offset += index * index_strides[i];
    }
    device T * out = (device T*)((device char*)outputData + offsets.x + offset);
    constant E * in  = (constant E*)((constant char*)inputData  + offsets.y);
    atomic_fetch_add_explicit(out, *in, memory_order_relaxed);
}

template<typename T>
__attribute__((__always_inline__)) void atomic_fetch_add_relaxed(device void * addr, T value) {
    device atomic_uint* uintAddr = (device atomic_uint*)addr;
    uint expected = atomic_load_explicit(uintAddr, memory_order_relaxed);
    T updated = as_type<T>(expected) + value;
    while (!atomic_compare_exchange_weak_explicit(uintAddr, &expected, as_type<uint>(updated), memory_order_relaxed, memory_order_relaxed)) {
        updated = as_type<T>(expected) + value;
    }
}

template<typename T>
kernel void atomic_index_put_accumulate(
#if __METAL_VERSION__ >= 300
    constant IndexAB  * indexAB           [[buffer(0)]],
#else
    constant IndexAB  & indexAB           [[buffer(0)]],
#endif
    constant void     * indexSizes        [[buffer(1)]],
    constant void     * indexStrides      [[buffer(2)]],
    constant void     * inputData         [[buffer(4)]],
    device   void     * outputData        [[buffer(5)]],
    constant uint32_t & num_indices       [[buffer(6)]],
    constant uint     * iter_shape        [[buffer(7)]],
    constant uint     & num_dimensions    [[buffer(8)]],
    constant packed_uint3 * strides       [[buffer(9)]],
    uint thread_index [[thread_position_in_grid]]) {
    uint3 offsets = get_idx(thread_index, iter_shape, num_dimensions, strides);
    constant int64_t * index_sizes   = (constant int64_t *)indexSizes;
    constant int64_t * index_strides = (constant int64_t *)indexStrides;
    int64_t offset = 0;
    for (uint32_t i = 0; i < num_indices; i++) {
#if __METAL_VERSION__ >= 300
        constant int64_t* indexArray = indexAB[i].indexArray;
#else
        constant int64_t* indexArray = (constant int64_t*)indexAB.indexArray[i];
#endif
        int64_t index = indexArray[offsets.z / sizeof(int64_t)];
        if (index < 0) {
            index += index_sizes[i];
        }
        offset += index * index_strides[i];
    }
    device void * out = (device void*)((device char*)outputData + offsets.x + offset);
    constant T  * in  = (constant T*)((constant char*)inputData + offsets.y);
    atomic_fetch_add_relaxed<T>(out, *in);
}

template
[[host_name("index_put_accumulate_32bit_float")]]
kernel void atomic_index_put_accumulate<float>(
#if __METAL_VERSION__ >= 300
    constant IndexAB  * indexAB     [[buffer(0)]],
#else
    constant IndexAB  & indexAB     [[buffer(0)]],
#endif
    constant void    * indexSizes   [[buffer(1)]],
    constant void    * indexStrides [[buffer(2)]],
    constant void    * inputData    [[buffer(4)]],
    device   void    * outputData   [[buffer(5)]],
    constant uint32_t& num_indices  [[buffer(6)]],
    constant uint  * iter_shape     [[buffer(7)]],
    constant uint & num_dimensions  [[buffer(8)]],
    constant packed_uint3 * strides [[buffer(9)]],
    uint thread_index [[thread_position_in_grid]]);

template
[[host_name("index_put_accumulate_32bit_int")]]
kernel void index_put_accumulate_native_dtypes<atomic_int, int>(
#if __METAL_VERSION__ >= 300
    constant IndexAB  * indexAB       [[buffer(0)]],
#else
    constant IndexAB  & indexAB       [[buffer(0)]],
#endif
    constant void    * indexSizes     [[buffer(1)]],
    constant void    * indexStrides   [[buffer(2)]],
    constant void    * inputData      [[buffer(4)]],
    device   void    * outputData     [[buffer(5)]],
    constant uint32_t& num_indices    [[buffer(6)]],
    constant uint    * iter_shape     [[buffer(7)]],
    constant uint    & num_dimensions [[buffer(8)]],
    constant packed_uint3 * strides   [[buffer(9)]],
    uint thread_index [[thread_position_in_grid]]);
)INDEX_METAL";
}
}
