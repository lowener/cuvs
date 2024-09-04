#pragma once

/**
 * Utility code involving integer arithmetic
 *
 */

#include <raft/core/detail/macros.hpp>
#include <cuda/std/limits>
#include <cuda/std/type_traits>
#include <cuda/std/cfloat>
#include <cuda/std/climits>

// TODO: temporary define
#define FLT_MIN           1.1754943e-38f
#define FLT_MAX           3.4028234e38f
#define SCHAR_MIN         (-128)
#define SCHAR_MAX         127
#define UCHAR_MIN         0
#define UCHAR_MAX         255

namespace raft {

template <typename I>
constexpr inline auto is_a_power_of_two(I val) noexcept
  -> cuda::std::enable_if_t<cuda::std::is_integral<I>::value, bool>
{
  return (val != 0) && (((val - 1) & val) == 0);
}



/**
 * @brief Minimum Minimum of two or more values.
 *
 * The CUDA Math API has overloads for all combinations of float/double. We provide similar
 * functionality while wrapping around std::min, which only supports arguments of the same type.
 * However, though the CUDA Math API supports combinations of unsigned and signed integers, this is
 * very error-prone so we do not support that and require the user to cast instead. (e.g the min of
 * -1 and 1u is 1u...)
 *
 * When no overload matches, we provide a generic implementation but require that both types be the
 * same (and that the less-than operator be defined).
 * @{
 */
template <
  typename T1,
  typename T2,
  cuda::std::enable_if_t<CUDA_CONDITION_ELSE_TRUE(RAFT_DEPAREN(
                     ((!cuda::std::is_same_v<T1, __half> && !cuda::std::is_same_v<T2, __half>) ||
                      (!cuda::std::is_same_v<T1, nv_bfloat16> && !cuda::std::is_same_v<T2, nv_bfloat16>)))),
                   int> = 0>
RAFT_INLINE_FUNCTION auto min(const T1& x, const T2& y)
{
    return (y < x) ? y : x;
}

#if defined(_RAFT_HAS_CUDA)
template <typename T>
RAFT_DEVICE_INLINE_FUNCTION typename cuda::std::enable_if_t<cuda::std::is_same_v<T, __half>, __half> min(T x,
                                                                                             T y)
{
#if (__CUDA_ARCH__ >= 530)
  return ::__hmin(x, y);
#else
  // Fail during template instantiation if the compute capability doesn't support this operation
  static_assert(sizeof(T) != sizeof(T), "__half is only supported on __CUDA_ARCH__ >= 530");
  return T{};
#endif
}

template <typename T>
RAFT_DEVICE_INLINE_FUNCTION typename cuda::std::enable_if_t<cuda::std::is_same_v<T, nv_bfloat16>, nv_bfloat16>
min(T x, T y)
{
#if (__CUDA_ARCH__ >= 800)
  return ::__hmin(x, y);
#else
  // Fail during template instantiation if the compute capability doesn't support this operation
  static_assert(sizeof(T) != sizeof(T), "nv_bfloat16 is only supported on __CUDA_ARCH__ >= 800");
  return T{};
#endif
}
#endif


/**
 * @brief The CUDA Math API has overloads for all combinations of float/double. We provide similar
 * functionality while wrapping around std::max, which only supports arguments of the same type.
 * However, though the CUDA Math API supports combinations of unsigned and signed integers, this is
 * very error-prone so we do not support that and require the user to cast instead. (e.g the max of
 * -1 and 1u is 4294967295u...)
 *
 * When no overload matches, we provide a generic implementation but require that both types be the
 * same (and that the less-than operator be defined).
 * @{
 */
template <
  typename T1,
  typename T2,
  cuda::std::enable_if_t<CUDA_CONDITION_ELSE_TRUE(RAFT_DEPAREN(
                     ((!cuda::std::is_same_v<T1, __half> && !cuda::std::is_same_v<T2, __half>) ||
                      (!cuda::std::is_same_v<T1, nv_bfloat16> && !cuda::std::is_same_v<T2, nv_bfloat16>)))),
                   int> = 0>
RAFT_INLINE_FUNCTION auto max(const T1& x, const T2& y)
{
    return (x < y) ? y : x;
}

#if defined(_RAFT_HAS_CUDA)
template <typename T>
RAFT_DEVICE_INLINE_FUNCTION typename cuda::std::enable_if_t<cuda::std::is_same_v<T, __half>, __half> max(T x,
                                                                                             T y)
{
#if (__CUDA_ARCH__ >= 530)
  return ::__hmax(x, y);
#else
  // Fail during template instantiation if the compute capability doesn't support this operation
  static_assert(sizeof(T) != sizeof(T), "__half is only supported on __CUDA_ARCH__ >= 530");
  return T{};
#endif
}

template <typename T>
RAFT_DEVICE_INLINE_FUNCTION typename cuda::std::enable_if_t<cuda::std::is_same_v<T, nv_bfloat16>, nv_bfloat16>
max(T x, T y)
{
#if (__CUDA_ARCH__ >= 800)
  return ::__hmax(x, y);
#else
  // Fail during template instantiation if the compute capability doesn't support this operation
  static_assert(sizeof(T) != sizeof(T), "nv_bfloat16 is only supported on __CUDA_ARCH__ >= 800");
  return T{};
#endif
}
#endif

/** Many-argument overload to avoid verbose nested calls or use with variadic arguments */
template <typename T1, typename T2, typename... Args>
RAFT_INLINE_FUNCTION auto max(const T1& x, const T2& y, Args&&... args)
{
  return raft::max(x, raft::max(y, cuda::std::forward<Args>(args)...));
}

/** One-argument overload for convenience when using with variadic arguments */
template <typename T>
constexpr RAFT_INLINE_FUNCTION auto max(const T& x)
{
  return x;
}

#if defined(_RAFT_HAS_CUDA)
template <typename T>
RAFT_DEVICE_INLINE_FUNCTION typename cuda::std::enable_if_t<cuda::std::is_same_v<T, __half>, __half> max(T x)
{
#if (__CUDA_ARCH__ >= 530)
  return x;
#else
  // Fail during template instantiation if the compute capability doesn't support this operation
  static_assert(sizeof(T) != sizeof(T), "__half is only supported on __CUDA_ARCH__ >= 530");
  return T{};
#endif
}

template <typename T>
RAFT_DEVICE_INLINE_FUNCTION typename cuda::std::enable_if_t<cuda::std::is_same_v<T, nv_bfloat16>, nv_bfloat16>
max(T x)
{
#if (__CUDA_ARCH__ >= 800)
  return x;
#else
  // Fail during template instantiation if the compute capability doesn't support this operation
  static_assert(sizeof(T) != sizeof(T), "nv_bfloat16 is only supported on __CUDA_ARCH__ >= 800");
  return T{};
#endif
}
#endif


template <typename T>
constexpr T lower_bound()
{
  static_assert(false, "Overload missing for lower_bound");
  /*if constexpr (std::numeric_limits<T>::has_infinity && std::numeric_limits<T>::is_signed) {
    return -std::numeric_limits<T>::infinity();
  }
  return std::numeric_limits<T>::lowest();*/
}

template <>
constexpr float lower_bound()
{
  return FLT_MIN;
}

template <>
constexpr uint8_t lower_bound()
{
  return UCHAR_MIN;
}

template <>
constexpr int8_t lower_bound()
{
  return SCHAR_MIN;
}

template <typename T>
constexpr T upper_bound()
{

  static_assert(false, "Overload missing for upper_bound");
  /*if constexpr (std::numeric_limits<T>::has_infinity) { return std::numeric_limits<T>::infinity(); }
  return std::numeric_limits<T>::max();*/
}
template <>
constexpr float upper_bound()
{
  return FLT_MAX;
}

template <>
constexpr uint8_t upper_bound()
{
  return UCHAR_MAX;
}

template <>
constexpr int8_t upper_bound()
{
  return SCHAR_MAX;
}
} // namespace raft