#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#include <cuda/std/climits>
#include <cuda/std/cstddef>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

namespace raft {


  template <typename math_, int VecLen>
  struct IOType {};
  template <>
  struct IOType<bool, 1> {
    static_assert(sizeof(bool) == sizeof(int8_t), "IOType bool size assumption failed");
    typedef int8_t Type;
  };
  template <>
  struct IOType<bool, 2> {
    typedef int16_t Type;
  };
  template <>
  struct IOType<bool, 4> {
    typedef int32_t Type;
  };
  template <>
  struct IOType<bool, 8> {
    typedef int2 Type;
  };
  template <>
  struct IOType<bool, 16> {
    typedef int4 Type;
  };
  template <>
  struct IOType<int8_t, 1> {
    typedef int8_t Type;
  };
  template <>
  struct IOType<int8_t, 2> {
    typedef int16_t Type;
  };
  template <>
  struct IOType<int8_t, 4> {
    typedef int32_t Type;
  };
  template <>
  struct IOType<int8_t, 8> {
    typedef int2 Type;
  };
  template <>
  struct IOType<int8_t, 16> {
    typedef int4 Type;
  };
  template <>
  struct IOType<uint8_t, 1> {
    typedef uint8_t Type;
  };
  template <>
  struct IOType<uint8_t, 2> {
    typedef uint16_t Type;
  };
  template <>
  struct IOType<uint8_t, 4> {
    typedef uint32_t Type;
  };
  template <>
  struct IOType<uint8_t, 8> {
    typedef uint2 Type;
  };
  template <>
  struct IOType<uint8_t, 16> {
    typedef uint4 Type;
  };
  template <>
  struct IOType<int16_t, 1> {
    typedef int16_t Type;
  };
  template <>
  struct IOType<int16_t, 2> {
    typedef int32_t Type;
  };
  template <>
  struct IOType<int16_t, 4> {
    typedef int2 Type;
  };
  template <>
  struct IOType<int16_t, 8> {
    typedef int4 Type;
  };
  template <>
  struct IOType<uint16_t, 1> {
    typedef uint16_t Type;
  };
  template <>
  struct IOType<uint16_t, 2> {
    typedef uint32_t Type;
  };
  template <>
  struct IOType<uint16_t, 4> {
    typedef uint2 Type;
  };
  template <>
  struct IOType<uint16_t, 8> {
    typedef uint4 Type;
  };
  template <>
  struct IOType<__half, 1> {
    typedef __half Type;
  };
  template <>
  struct IOType<__half, 2> {
    typedef __half2 Type;
  };
  template <>
  struct IOType<__half, 4> {
    typedef uint2 Type;
  };
  template <>
  struct IOType<__half, 8> {
    typedef uint4 Type;
  };
  template <>
  struct IOType<__half2, 1> {
    typedef __half2 Type;
  };
  template <>
  struct IOType<__half2, 2> {
    typedef uint2 Type;
  };
  template <>
  struct IOType<__half2, 4> {
    typedef uint4 Type;
  };
  template <>
  struct IOType<int32_t, 1> {
    typedef int32_t Type;
  };
  template <>
  struct IOType<int32_t, 2> {
    typedef uint2 Type;
  };
  template <>
  struct IOType<int32_t, 4> {
    typedef uint4 Type;
  };
  template <>
  struct IOType<uint32_t, 1> {
    typedef uint32_t Type;
  };
  template <>
  struct IOType<uint32_t, 2> {
    typedef uint2 Type;
  };
  template <>
  struct IOType<uint32_t, 4> {
    typedef uint4 Type;
  };
  template <>
  struct IOType<float, 1> {
    typedef float Type;
  };
  template <>
  struct IOType<float, 2> {
    typedef float2 Type;
  };
  template <>
  struct IOType<float, 4> {
    typedef float4 Type;
  };
  template <>
  struct IOType<int64_t, 1> {
    typedef int64_t Type;
  };
  template <>
  struct IOType<int64_t, 2> {
    typedef uint4 Type;
  };
  /*template <>
  struct IOType<uint64_t, 1> {
    typedef uint64_t Type;
  };
  template <>
  struct IOType<uint64_t, 2> {
    typedef uint4 Type;
  };*/
  template <>
  struct IOType<unsigned long long, 1> {
    typedef unsigned long long Type;
  };
  template <>
  struct IOType<unsigned long long, 2> {
    typedef uint4 Type;
  };
  template <>
  struct IOType<double, 1> {
    typedef double Type;
  };
  template <>
  struct IOType<double, 2> {
    typedef double2 Type;
  };
} // namespace raft;
