#pragma once

#include <cuda/std/climits>
#include <cuda/std/cstddef>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

namespace cuvs::tools {
/** number of threads per warp */
const int WarpSize = 32;
/** True CUDA alignment of a type (adapted from CUB) */
template <typename T>
struct cuda_alignment {
  struct Pad {
    T val;
    char byte;
  };

  static constexpr int bytes = sizeof(Pad) - sizeof(T);
};

template <typename LargeT, typename UnitT>
struct is_multiple {
  static constexpr int large_align_bytes = cuda_alignment<LargeT>::bytes;
  static constexpr int unit_align_bytes  = cuda_alignment<UnitT>::bytes;
  static constexpr bool value =
    (sizeof(LargeT) % sizeof(UnitT) == 0) && (large_align_bytes % unit_align_bytes == 0);
};

template <typename T>
struct is_shuffleable {
  static constexpr bool value =
    cuda::std::is_same_v<T, int> || cuda::std::is_same_v<T, unsigned int> ||
    cuda::std::is_same_v<T, long> || cuda::std::is_same_v<T, unsigned long> ||
    cuda::std::is_same_v<T, long long> || cuda::std::is_same_v<T, unsigned long long> ||
    cuda::std::is_same_v<T, float> || cuda::std::is_same_v<T, double>;
};

template <typename T>
inline constexpr bool is_shuffleable_v = is_shuffleable<T>::value;

/**
 * @brief Shuffle the data inside a warp
 * @tparam T the data type
 * @param val value to be shuffled
 * @param srcLane lane from where to shuffle
 * @param width lane width
 * @param mask mask of participating threads (Volta+)
 * @return the shuffled data
 */
template <typename T>
__device__ inline cuda::std::enable_if_t<is_shuffleable_v<T>, T> shfl(T val,
                                                                      int srcLane,
                                                                      int width     = WarpSize,
                                                                      uint32_t mask = 0xffffffffu)
{
#if CUDART_VERSION >= 9000
  return __shfl_sync(mask, val, srcLane, width);
#else
  return __shfl(val, srcLane, width);
#endif
}

/// Overload of shfl for data types not supported by the CUDA intrinsics
template <typename T>
__device__ inline cuda::std::enable_if_t<!is_shuffleable_v<T>, T> shfl(T val,
                                                                       int srcLane,
                                                                       int width     = WarpSize,
                                                                       uint32_t mask = 0xffffffffu)
{
  using UnitT = cuda::std::conditional_t<
    is_multiple_v<T, int>,
    unsigned int,
    cuda::std::conditional_t<is_multiple_v<T, short>, unsigned short, unsigned char>>;

  constexpr int n_words = sizeof(T) / sizeof(UnitT);

  T output;
  UnitT* output_alias = reinterpret_cast<UnitT*>(&output);
  UnitT* input_alias  = reinterpret_cast<UnitT*>(&val);

  unsigned int shuffle_word;
  shuffle_word    = shfl((unsigned int)input_alias[0], srcLane, width, mask);
  output_alias[0] = shuffle_word;

#pragma unroll
  for (int i = 1; i < n_words; ++i) {
    shuffle_word    = shfl((unsigned int)input_alias[i], srcLane, width, mask);
    output_alias[i] = shuffle_word;
  }

  return output;
}
}  // namespace cuvs::tools
