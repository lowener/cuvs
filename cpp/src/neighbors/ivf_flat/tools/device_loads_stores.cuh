#pragma once

#include <cuda_fp16.h>

#include <cuda/std/cstdint>  // uintX_t

namespace raft {

/**
 * @defgroup SmemStores Shared memory store operations
 * @{
 * @brief Stores to shared memory (both vectorized and non-vectorized forms)
 *        requires the given shmem pointer to be aligned by the vector
          length, like for float4 lds/sts shmem pointer should be aligned
          by 16 bytes else it might silently fail or can also give
          runtime error.
 * @param[out] addr shared memory address (should be aligned to vector size)
 * @param[in]  x    data to be stored at this address
 */
__device__ inline void sts(uint8_t* addr, const uint8_t& x)
{
  uint32_t x_int;
  x_int   = x;
  auto s1 = __cvta_generic_to_shared(reinterpret_cast<uint8_t*>(addr));
  asm volatile("st.shared.u8 [%0], {%1};" : : "l"(s1), "r"(x_int));
}
__device__ inline void sts(uint8_t* addr, const uint8_t (&x)[1])
{
  uint32_t x_int[1];
  x_int[0] = x[0];
  auto s1  = __cvta_generic_to_shared(reinterpret_cast<uint8_t*>(addr));
  asm volatile("st.shared.u8 [%0], {%1};" : : "l"(s1), "r"(x_int[0]));
}
__device__ inline void sts(uint8_t* addr, const uint8_t (&x)[2])
{
  uint32_t x_int[2];
  x_int[0] = x[0];
  x_int[1] = x[1];
  auto s2  = __cvta_generic_to_shared(reinterpret_cast<uint8_t*>(addr));
  asm volatile("st.shared.v2.u8 [%0], {%1, %2};" : : "l"(s2), "r"(x_int[0]), "r"(x_int[1]));
}
__device__ inline void sts(uint8_t* addr, const uint8_t (&x)[4])
{
  uint32_t x_int[4];
  x_int[0] = x[0];
  x_int[1] = x[1];
  x_int[2] = x[2];
  x_int[3] = x[3];
  auto s4  = __cvta_generic_to_shared(reinterpret_cast<uint8_t*>(addr));
  asm volatile("st.shared.v4.u8 [%0], {%1, %2, %3, %4};"
               :
               : "l"(s4), "r"(x_int[0]), "r"(x_int[1]), "r"(x_int[2]), "r"(x_int[3]));
}

__device__ inline void sts(int8_t* addr, const int8_t& x)
{
  int32_t x_int = x;
  auto s1       = __cvta_generic_to_shared(reinterpret_cast<int8_t*>(addr));
  asm volatile("st.shared.s8 [%0], {%1};" : : "l"(s1), "r"(x_int));
}
__device__ inline void sts(int8_t* addr, const int8_t (&x)[1])
{
  int32_t x_int[1];
  x_int[0] = x[0];
  auto s1  = __cvta_generic_to_shared(reinterpret_cast<int8_t*>(addr));
  asm volatile("st.shared.s8 [%0], {%1};" : : "l"(s1), "r"(x_int[0]));
}
__device__ inline void sts(int8_t* addr, const int8_t (&x)[2])
{
  int32_t x_int[2];
  x_int[0] = x[0];
  x_int[1] = x[1];
  auto s2  = __cvta_generic_to_shared(reinterpret_cast<int8_t*>(addr));
  asm volatile("st.shared.v2.s8 [%0], {%1, %2};" : : "l"(s2), "r"(x_int[0]), "r"(x_int[1]));
}
__device__ inline void sts(int8_t* addr, const int8_t (&x)[4])
{
  int32_t x_int[4];
  x_int[0] = x[0];
  x_int[1] = x[1];
  x_int[2] = x[2];
  x_int[3] = x[3];
  auto s4  = __cvta_generic_to_shared(reinterpret_cast<int8_t*>(addr));
  asm volatile("st.shared.v4.s8 [%0], {%1, %2, %3, %4};"
               :
               : "l"(s4), "r"(x_int[0]), "r"(x_int[1]), "r"(x_int[2]), "r"(x_int[3]));
}

__device__ inline void sts(uint32_t* addr, const uint32_t& x)
{
  auto s1 = __cvta_generic_to_shared(reinterpret_cast<uint32_t*>(addr));
  asm volatile("st.shared.u32 [%0], {%1};" : : "l"(s1), "r"(x));
}
__device__ inline void sts(uint32_t* addr, const uint32_t (&x)[1])
{
  auto s1 = __cvta_generic_to_shared(reinterpret_cast<uint32_t*>(addr));
  asm volatile("st.shared.u32 [%0], {%1};" : : "l"(s1), "r"(x[0]));
}
__device__ inline void sts(uint32_t* addr, const uint32_t (&x)[2])
{
  auto s2 = __cvta_generic_to_shared(reinterpret_cast<uint2*>(addr));
  asm volatile("st.shared.v2.u32 [%0], {%1, %2};" : : "l"(s2), "r"(x[0]), "r"(x[1]));
}
__device__ inline void sts(uint32_t* addr, const uint32_t (&x)[4])
{
  auto s4 = __cvta_generic_to_shared(reinterpret_cast<uint4*>(addr));
  asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};"
               :
               : "l"(s4), "r"(x[0]), "r"(x[1]), "r"(x[2]), "r"(x[3]));
}

__device__ inline void sts(int32_t* addr, const int32_t& x)
{
  auto s1 = __cvta_generic_to_shared(reinterpret_cast<int32_t*>(addr));
  asm volatile("st.shared.u32 [%0], {%1};" : : "l"(s1), "r"(x));
}
__device__ inline void sts(int32_t* addr, const int32_t (&x)[1])
{
  auto s1 = __cvta_generic_to_shared(reinterpret_cast<int32_t*>(addr));
  asm volatile("st.shared.u32 [%0], {%1};" : : "l"(s1), "r"(x[0]));
}
__device__ inline void sts(int32_t* addr, const int32_t (&x)[2])
{
  auto s2 = __cvta_generic_to_shared(reinterpret_cast<int2*>(addr));
  asm volatile("st.shared.v2.u32 [%0], {%1, %2};" : : "l"(s2), "r"(x[0]), "r"(x[1]));
}
__device__ inline void sts(int32_t* addr, const int32_t (&x)[4])
{
  auto s4 = __cvta_generic_to_shared(reinterpret_cast<int4*>(addr));
  asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};"
               :
               : "l"(s4), "r"(x[0]), "r"(x[1]), "r"(x[2]), "r"(x[3]));
}

__device__ inline void sts(half* addr, const half& x)
{
  auto s = __cvta_generic_to_shared(reinterpret_cast<uint16_t*>(addr));
  asm volatile("st.shared.u16 [%0], {%1};" : : "l"(s), "h"(*reinterpret_cast<const uint16_t*>(&x)));
}
__device__ inline void sts(half* addr, const half (&x)[1])
{
  auto s = __cvta_generic_to_shared(reinterpret_cast<uint16_t*>(addr));
  asm volatile("st.shared.u16 [%0], {%1};" : : "l"(s), "h"(*reinterpret_cast<const uint16_t*>(x)));
}
__device__ inline void sts(half* addr, const half (&x)[2])
{
  auto s = __cvta_generic_to_shared(reinterpret_cast<uint16_t*>(addr));
  asm volatile("st.shared.v2.u16 [%0], {%1, %2};"
               :
               : "l"(s),
                 "h"(*reinterpret_cast<const uint16_t*>(x)),
                 "h"(*reinterpret_cast<const uint16_t*>(x + 1)));
}
__device__ inline void sts(half* addr, const half (&x)[4])
{
  auto s = __cvta_generic_to_shared(reinterpret_cast<uint16_t*>(addr));
  asm volatile("st.shared.v4.u16 [%0], {%1, %2, %3, %4};"
               :
               : "l"(s),
                 "h"(*reinterpret_cast<const uint16_t*>(x)),
                 "h"(*reinterpret_cast<const uint16_t*>(x + 1)),
                 "h"(*reinterpret_cast<const uint16_t*>(x + 2)),
                 "h"(*reinterpret_cast<const uint16_t*>(x + 3)));
}
__device__ inline void sts(half* addr, const half (&x)[8])
{
  auto s = __cvta_generic_to_shared(reinterpret_cast<uint32_t*>(addr));
  half2 y[4];
  y[0].x = x[0];
  y[0].y = x[1];
  y[1].x = x[2];
  y[1].y = x[3];
  y[2].x = x[4];
  y[2].y = x[5];
  y[3].x = x[6];
  y[3].y = x[7];
  asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};"
               :
               : "l"(s),
                 "r"(*reinterpret_cast<uint32_t*>(y)),
                 "r"(*reinterpret_cast<uint32_t*>(y + 1)),
                 "r"(*reinterpret_cast<uint32_t*>(y + 2)),
                 "r"(*reinterpret_cast<uint32_t*>(y + 3)));
}

__device__ inline void sts(float* addr, const float& x)
{
  auto s1 = __cvta_generic_to_shared(reinterpret_cast<float*>(addr));
  asm volatile("st.shared.f32 [%0], {%1};" : : "l"(s1), "f"(x));
}
__device__ inline void sts(float* addr, const float (&x)[1])
{
  auto s1 = __cvta_generic_to_shared(reinterpret_cast<float*>(addr));
  asm volatile("st.shared.f32 [%0], {%1};" : : "l"(s1), "f"(x[0]));
}
__device__ inline void sts(float* addr, const float (&x)[2])
{
  auto s2 = __cvta_generic_to_shared(reinterpret_cast<float2*>(addr));
  asm volatile("st.shared.v2.f32 [%0], {%1, %2};" : : "l"(s2), "f"(x[0]), "f"(x[1]));
}
__device__ inline void sts(float* addr, const float (&x)[4])
{
  auto s4 = __cvta_generic_to_shared(reinterpret_cast<float4*>(addr));
  asm volatile("st.shared.v4.f32 [%0], {%1, %2, %3, %4};"
               :
               : "l"(s4), "f"(x[0]), "f"(x[1]), "f"(x[2]), "f"(x[3]));
}

__device__ inline void sts(double* addr, const double& x)
{
  auto s1 = __cvta_generic_to_shared(reinterpret_cast<double*>(addr));
  asm volatile("st.shared.f64 [%0], {%1};" : : "l"(s1), "d"(x));
}
__device__ inline void sts(double* addr, const double (&x)[1])
{
  auto s1 = __cvta_generic_to_shared(reinterpret_cast<double*>(addr));
  asm volatile("st.shared.f64 [%0], {%1};" : : "l"(s1), "d"(x[0]));
}
__device__ inline void sts(double* addr, const double (&x)[2])
{
  auto s2 = __cvta_generic_to_shared(reinterpret_cast<double2*>(addr));
  asm volatile("st.shared.v2.f64 [%0], {%1, %2};" : : "l"(s2), "d"(x[0]), "d"(x[1]));
}
/** @} */

/**
 * @defgroup SmemLoads Shared memory load operations
 * @{
 * @brief Loads from shared memory (both vectorized and non-vectorized forms)
          requires the given shmem pointer to be aligned by the vector
          length, like for float4 lds/sts shmem pointer should be aligned
          by 16 bytes else it might silently fail or can also give
          runtime error.
 * @param[out] x    the data to be loaded
 * @param[in]  addr shared memory address from where to load
 *                  (should be aligned to vector size)
 */

__device__ inline void lds(uint8_t& x, const uint8_t* addr)
{
  uint32_t x_int;
  auto s1 = __cvta_generic_to_shared(reinterpret_cast<const uint8_t*>(addr));
  asm volatile("ld.shared.u8 {%0}, [%1];" : "=r"(x_int) : "l"(s1));
  x = x_int;
}
__device__ inline void lds(uint8_t (&x)[1], const uint8_t* addr)
{
  uint32_t x_int[1];
  auto s1 = __cvta_generic_to_shared(reinterpret_cast<const uint8_t*>(addr));
  asm volatile("ld.shared.u8 {%0}, [%1];" : "=r"(x_int[0]) : "l"(s1));
  x[0] = x_int[0];
}
__device__ inline void lds(uint8_t (&x)[2], const uint8_t* addr)
{
  uint32_t x_int[2];
  auto s2 = __cvta_generic_to_shared(reinterpret_cast<const uint8_t*>(addr));
  asm volatile("ld.shared.v2.u8 {%0, %1}, [%2];" : "=r"(x_int[0]), "=r"(x_int[1]) : "l"(s2));
  x[0] = x_int[0];
  x[1] = x_int[1];
}
__device__ inline void lds(uint8_t (&x)[4], const uint8_t* addr)
{
  uint32_t x_int[4];
  auto s4 = __cvta_generic_to_shared(reinterpret_cast<const uint8_t*>(addr));
  asm volatile("ld.shared.v4.u8 {%0, %1, %2, %3}, [%4];"
               : "=r"(x_int[0]), "=r"(x_int[1]), "=r"(x_int[2]), "=r"(x_int[3])
               : "l"(s4));
  x[0] = x_int[0];
  x[1] = x_int[1];
  x[2] = x_int[2];
  x[3] = x_int[3];
}

__device__ inline void lds(int8_t& x, const int8_t* addr)
{
  int32_t x_int;
  auto s1 = __cvta_generic_to_shared(reinterpret_cast<const int8_t*>(addr));
  asm volatile("ld.shared.s8 {%0}, [%1];" : "=r"(x_int) : "l"(s1));
  x = x_int;
}
__device__ inline void lds(int8_t (&x)[1], const int8_t* addr)
{
  int32_t x_int[1];
  auto s1 = __cvta_generic_to_shared(reinterpret_cast<const int8_t*>(addr));
  asm volatile("ld.shared.s8 {%0}, [%1];" : "=r"(x_int[0]) : "l"(s1));
  x[0] = x_int[0];
}
__device__ inline void lds(int8_t (&x)[2], const int8_t* addr)
{
  int32_t x_int[2];
  auto s2 = __cvta_generic_to_shared(reinterpret_cast<const int8_t*>(addr));
  asm volatile("ld.shared.v2.s8 {%0, %1}, [%2];" : "=r"(x_int[0]), "=r"(x_int[1]) : "l"(s2));
  x[0] = x_int[0];
  x[1] = x_int[1];
}
__device__ inline void lds(int8_t (&x)[4], const int8_t* addr)
{
  int32_t x_int[4];
  auto s4 = __cvta_generic_to_shared(reinterpret_cast<const int8_t*>(addr));
  asm volatile("ld.shared.v4.s8 {%0, %1, %2, %3}, [%4];"
               : "=r"(x_int[0]), "=r"(x_int[1]), "=r"(x_int[2]), "=r"(x_int[3])
               : "l"(s4));
  x[0] = x_int[0];
  x[1] = x_int[1];
  x[2] = x_int[2];
  x[3] = x_int[3];
}

__device__ inline void lds(uint32_t (&x)[4], const uint32_t* addr)
{
  auto s4 = __cvta_generic_to_shared(reinterpret_cast<const uint32_t*>(addr));
  asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(x[0]), "=r"(x[1]), "=r"(x[2]), "=r"(x[3])
               : "l"(s4));
}

__device__ inline void lds(uint32_t (&x)[2], const uint32_t* addr)
{
  auto s2 = __cvta_generic_to_shared(reinterpret_cast<const uint32_t*>(addr));
  asm volatile("ld.shared.v2.u32 {%0, %1}, [%2];" : "=r"(x[0]), "=r"(x[1]) : "l"(s2));
}

__device__ inline void lds(uint32_t (&x)[1], const uint32_t* addr)
{
  auto s1 = __cvta_generic_to_shared(reinterpret_cast<const uint32_t*>(addr));
  asm volatile("ld.shared.u32 {%0}, [%1];" : "=r"(x[0]) : "l"(s1));
}

__device__ inline void lds(uint32_t& x, const uint32_t* addr)
{
  auto s1 = __cvta_generic_to_shared(reinterpret_cast<const uint32_t*>(addr));
  asm volatile("ld.shared.u32 {%0}, [%1];" : "=r"(x) : "l"(s1));
}

__device__ inline void lds(int32_t (&x)[4], const int32_t* addr)
{
  auto s4 = __cvta_generic_to_shared(reinterpret_cast<const int32_t*>(addr));
  asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(x[0]), "=r"(x[1]), "=r"(x[2]), "=r"(x[3])
               : "l"(s4));
}

__device__ inline void lds(int32_t (&x)[2], const int32_t* addr)
{
  auto s2 = __cvta_generic_to_shared(reinterpret_cast<const int32_t*>(addr));
  asm volatile("ld.shared.v2.u32 {%0, %1}, [%2];" : "=r"(x[0]), "=r"(x[1]) : "l"(s2));
}

__device__ inline void lds(int32_t (&x)[1], const int32_t* addr)
{
  auto s1 = __cvta_generic_to_shared(reinterpret_cast<const int32_t*>(addr));
  asm volatile("ld.shared.u32 {%0}, [%1];" : "=r"(x[0]) : "l"(s1));
}

__device__ inline void lds(int32_t& x, const int32_t* addr)
{
  auto s1 = __cvta_generic_to_shared(reinterpret_cast<const int32_t*>(addr));
  asm volatile("ld.shared.u32 {%0}, [%1];" : "=r"(x) : "l"(s1));
}

__device__ inline void lds(half& x, const half* addr)
{
  auto s = __cvta_generic_to_shared(reinterpret_cast<const uint16_t*>(addr));
  asm volatile("ld.shared.u16 {%0}, [%1];" : "=h"(*reinterpret_cast<uint16_t*>(&x)) : "l"(s));
}
__device__ inline void lds(half (&x)[1], const half* addr)
{
  auto s = __cvta_generic_to_shared(reinterpret_cast<const uint16_t*>(addr));
  asm volatile("ld.shared.u16 {%0}, [%1];" : "=h"(*reinterpret_cast<uint16_t*>(x)) : "l"(s));
}
__device__ inline void lds(half (&x)[2], const half* addr)
{
  auto s = __cvta_generic_to_shared(reinterpret_cast<const uint16_t*>(addr));
  asm volatile("ld.shared.v2.u16 {%0, %1}, [%2];"
               : "=h"(*reinterpret_cast<uint16_t*>(x)), "=h"(*reinterpret_cast<uint16_t*>(x + 1))
               : "l"(s));
}
__device__ inline void lds(half (&x)[4], const half* addr)
{
  auto s = __cvta_generic_to_shared(reinterpret_cast<const uint16_t*>(addr));
  asm volatile("ld.shared.v4.u16 {%0, %1, %2, %3}, [%4];"
               : "=h"(*reinterpret_cast<uint16_t*>(x)),
                 "=h"(*reinterpret_cast<uint16_t*>(x + 1)),
                 "=h"(*reinterpret_cast<uint16_t*>(x + 2)),
                 "=h"(*reinterpret_cast<uint16_t*>(x + 3))
               : "l"(s));
}
__device__ inline void lds(half (&x)[8], const half* addr)
{
  auto s = __cvta_generic_to_shared(reinterpret_cast<const uint32_t*>(addr));
  half2 y[4];
  asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(*reinterpret_cast<uint32_t*>(y)),
                 "=r"(*reinterpret_cast<uint32_t*>(y + 1)),
                 "=r"(*reinterpret_cast<uint32_t*>(y + 2)),
                 "=r"(*reinterpret_cast<uint32_t*>(y + 3))
               : "l"(s));
  x[0] = y[0].x;
  x[1] = y[0].y;
  x[2] = y[1].x;
  x[3] = y[1].y;
  x[4] = y[2].x;
  x[5] = y[2].y;
  x[6] = y[3].x;
  x[7] = y[3].y;
}
__device__ inline void lds(float& x, const float* addr)
{
  auto s1 = __cvta_generic_to_shared(reinterpret_cast<const float*>(addr));
  asm volatile("ld.shared.f32 {%0}, [%1];" : "=f"(x) : "l"(s1));
}
__device__ inline void lds(float (&x)[1], const float* addr)
{
  auto s1 = __cvta_generic_to_shared(reinterpret_cast<const float*>(addr));
  asm volatile("ld.shared.f32 {%0}, [%1];" : "=f"(x[0]) : "l"(s1));
}
__device__ inline void lds(float (&x)[2], const float* addr)
{
  auto s2 = __cvta_generic_to_shared(reinterpret_cast<const float2*>(addr));
  asm volatile("ld.shared.v2.f32 {%0, %1}, [%2];" : "=f"(x[0]), "=f"(x[1]) : "l"(s2));
}
__device__ inline void lds(float (&x)[4], const float* addr)
{
  auto s4 = __cvta_generic_to_shared(reinterpret_cast<const float4*>(addr));
  asm volatile("ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];"
               : "=f"(x[0]), "=f"(x[1]), "=f"(x[2]), "=f"(x[3])
               : "l"(s4));
}

__device__ inline void lds(float& x, float* addr)
{
  auto s1 = __cvta_generic_to_shared(reinterpret_cast<float*>(addr));
  asm volatile("ld.shared.f32 {%0}, [%1];" : "=f"(x) : "l"(s1));
}
__device__ inline void lds(float (&x)[1], float* addr)
{
  auto s1 = __cvta_generic_to_shared(reinterpret_cast<float*>(addr));
  asm volatile("ld.shared.f32 {%0}, [%1];" : "=f"(x[0]) : "l"(s1));
}
__device__ inline void lds(float (&x)[2], float* addr)
{
  auto s2 = __cvta_generic_to_shared(reinterpret_cast<float2*>(addr));
  asm volatile("ld.shared.v2.f32 {%0, %1}, [%2];" : "=f"(x[0]), "=f"(x[1]) : "l"(s2));
}
__device__ inline void lds(float (&x)[4], float* addr)
{
  auto s4 = __cvta_generic_to_shared(reinterpret_cast<float4*>(addr));
  asm volatile("ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];"
               : "=f"(x[0]), "=f"(x[1]), "=f"(x[2]), "=f"(x[3])
               : "l"(s4));
}
__device__ inline void lds(double& x, double* addr)
{
  auto s1 = __cvta_generic_to_shared(reinterpret_cast<double*>(addr));
  asm volatile("ld.shared.f64 {%0}, [%1];" : "=d"(x) : "l"(s1));
}
__device__ inline void lds(double (&x)[1], double* addr)
{
  auto s1 = __cvta_generic_to_shared(reinterpret_cast<double*>(addr));
  asm volatile("ld.shared.f64 {%0}, [%1];" : "=d"(x[0]) : "l"(s1));
}
__device__ inline void lds(double (&x)[2], double* addr)
{
  auto s2 = __cvta_generic_to_shared(reinterpret_cast<double2*>(addr));
  asm volatile("ld.shared.v2.f64 {%0, %1}, [%2];" : "=d"(x[0]), "=d"(x[1]) : "l"(s2));
}
/** @} */

/**
 * @defgroup GlobalLoads Global cached load operations
 * @{
 * @brief Load from global memory with caching at L1 level
 * @param[out] x    data to be loaded from global memory
 * @param[in]  addr address in global memory from where to load
 */
__device__ inline void ldg(float& x, const float* addr)
{
  asm volatile("ld.global.cg.f32 %0, [%1];" : "=f"(x) : "l"(addr));
}
__device__ inline void ldg(float (&x)[1], const float* addr)
{
  asm volatile("ld.global.cg.f32 %0, [%1];" : "=f"(x[0]) : "l"(addr));
}
__device__ inline void ldg(float (&x)[2], const float* addr)
{
  asm volatile("ld.global.cg.v2.f32 {%0, %1}, [%2];" : "=f"(x[0]), "=f"(x[1]) : "l"(addr));
}
__device__ inline void ldg(float (&x)[4], const float* addr)
{
  asm volatile("ld.global.cg.v4.f32 {%0, %1, %2, %3}, [%4];"
               : "=f"(x[0]), "=f"(x[1]), "=f"(x[2]), "=f"(x[3])
               : "l"(addr));
}
__device__ inline void ldg(half& x, const half* addr)
{
  asm volatile("ld.global.cg.u16 {%0}, [%1];"
               : "=h"(*reinterpret_cast<uint16_t*>(&x))
               : "l"(reinterpret_cast<const uint16_t*>(addr)));
}
__device__ inline void ldg(half (&x)[1], const half* addr)
{
  asm volatile("ld.global.cg.u16 {%0}, [%1];"
               : "=h"(*reinterpret_cast<uint16_t*>(x))
               : "l"(reinterpret_cast<const uint16_t*>(addr)));
}
__device__ inline void ldg(half (&x)[2], const half* addr)
{
  asm volatile("ld.global.cg.v2.u16 {%0, %1}, [%2];"
               : "=h"(*reinterpret_cast<uint16_t*>(x)), "=h"(*reinterpret_cast<uint16_t*>(x + 1))
               : "l"(reinterpret_cast<const uint16_t*>(addr)));
}
__device__ inline void ldg(half (&x)[4], const half* addr)
{
  asm volatile("ld.global.cg.v4.u16 {%0, %1, %2, %3}, [%4];"
               : "=h"(*reinterpret_cast<uint16_t*>(x)),
                 "=h"(*reinterpret_cast<uint16_t*>(x + 1)),
                 "=h"(*reinterpret_cast<uint16_t*>(x + 2)),
                 "=h"(*reinterpret_cast<uint16_t*>(x + 3))
               : "l"(reinterpret_cast<const uint16_t*>(addr)));
}

__device__ inline void ldg(half (&x)[8], const half* addr)
{
  half2 y[4];
  asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(*reinterpret_cast<uint32_t*>(y)),
                 "=r"(*reinterpret_cast<uint32_t*>(y + 1)),
                 "=r"(*reinterpret_cast<uint32_t*>(y + 2)),
                 "=r"(*reinterpret_cast<uint32_t*>(y + 3))
               : "l"(reinterpret_cast<const uint32_t*>(addr)));
  x[0] = y[0].x;
  x[1] = y[0].y;
  x[2] = y[1].x;
  x[3] = y[1].y;
  x[4] = y[2].x;
  x[5] = y[2].y;
  x[6] = y[3].x;
  x[7] = y[3].y;
}
__device__ inline void ldg(double& x, const double* addr)
{
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(x) : "l"(addr));
}
__device__ inline void ldg(double (&x)[1], const double* addr)
{
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(x[0]) : "l"(addr));
}
__device__ inline void ldg(double (&x)[2], const double* addr)
{
  asm volatile("ld.global.cg.v2.f64 {%0, %1}, [%2];" : "=d"(x[0]), "=d"(x[1]) : "l"(addr));
}

__device__ inline void ldg(uint32_t (&x)[4], const uint32_t* const& addr)
{
  asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(x[0]), "=r"(x[1]), "=r"(x[2]), "=r"(x[3])
               : "l"(addr));
}

__device__ inline void ldg(uint32_t (&x)[2], const uint32_t* const& addr)
{
  asm volatile("ld.global.cg.v2.u32 {%0, %1}, [%2];" : "=r"(x[0]), "=r"(x[1]) : "l"(addr));
}

__device__ inline void ldg(uint32_t (&x)[1], const uint32_t* const& addr)
{
  asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(x[0]) : "l"(addr));
}

__device__ inline void ldg(uint32_t& x, const uint32_t* const& addr)
{
  asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(x) : "l"(addr));
}

__device__ inline void ldg(int32_t (&x)[4], const int32_t* const& addr)
{
  asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(x[0]), "=r"(x[1]), "=r"(x[2]), "=r"(x[3])
               : "l"(addr));
}

__device__ inline void ldg(int32_t (&x)[2], const int32_t* const& addr)
{
  asm volatile("ld.global.cg.v2.u32 {%0, %1}, [%2];" : "=r"(x[0]), "=r"(x[1]) : "l"(addr));
}

__device__ inline void ldg(int32_t (&x)[1], const int32_t* const& addr)
{
  asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(x[0]) : "l"(addr));
}

__device__ inline void ldg(int32_t& x, const int32_t* const& addr)
{
  asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(x) : "l"(addr));
}

__device__ inline void ldg(uint8_t (&x)[4], const uint8_t* const& addr)
{
  uint32_t x_int[4];
  asm volatile("ld.global.cg.v4.u8 {%0, %1, %2, %3}, [%4];"
               : "=r"(x_int[0]), "=r"(x_int[1]), "=r"(x_int[2]), "=r"(x_int[3])
               : "l"(addr));
  x[0] = x_int[0];
  x[1] = x_int[1];
  x[2] = x_int[2];
  x[3] = x_int[3];
}

__device__ inline void ldg(uint8_t (&x)[2], const uint8_t* const& addr)
{
  uint32_t x_int[2];
  asm volatile("ld.global.cg.v2.u8 {%0, %1}, [%2];" : "=r"(x_int[0]), "=r"(x_int[1]) : "l"(addr));
  x[0] = x_int[0];
  x[1] = x_int[1];
}

__device__ inline void ldg(uint8_t (&x)[1], const uint8_t* const& addr)
{
  uint32_t x_int;
  asm volatile("ld.global.cg.u8 %0, [%1];" : "=r"(x_int) : "l"(addr));
  x[0] = x_int;
}

__device__ inline void ldg(uint8_t& x, const uint8_t* const& addr)
{
  uint32_t x_int;
  asm volatile("ld.global.cg.u8 %0, [%1];" : "=r"(x_int) : "l"(addr));
  x = x_int;
}

__device__ inline void ldg(int8_t (&x)[4], const int8_t* const& addr)
{
  int x_int[4];
  asm volatile("ld.global.cg.v4.s8 {%0, %1, %2, %3}, [%4];"
               : "=r"(x_int[0]), "=r"(x_int[1]), "=r"(x_int[2]), "=r"(x_int[3])
               : "l"(addr));
  x[0] = x_int[0];
  x[1] = x_int[1];
  x[2] = x_int[2];
  x[3] = x_int[3];
}

__device__ inline void ldg(int8_t (&x)[2], const int8_t* const& addr)
{
  int x_int[2];
  asm volatile("ld.global.cg.v2.s8 {%0, %1}, [%2];" : "=r"(x_int[0]), "=r"(x_int[1]) : "l"(addr));
  x[0] = x_int[0];
  x[1] = x_int[1];
}

__device__ inline void ldg(int8_t& x, const int8_t* const& addr)
{
  int x_int;
  asm volatile("ld.global.cg.s8 %0, [%1];" : "=r"(x_int) : "l"(addr));
  x = x_int;
}

__device__ inline void ldg(int8_t (&x)[1], const int8_t* const& addr)
{
  int x_int;
  asm volatile("ld.global.cg.s8 %0, [%1];" : "=r"(x_int) : "l"(addr));
  x[0] = x_int;
}
} // namespace raft