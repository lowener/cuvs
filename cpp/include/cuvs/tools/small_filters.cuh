#pragma once

#include <cuda/std/type_traits>
#include <raft/core/detail/macros.hpp>

namespace cuvs::neighbors::filtering {

struct none_ivf_sample_filter {
  /* A filter that filters nothing. This is the default behavior. */
  inline _RAFT_HOST_DEVICE bool operator()(
    // query index
    const uint32_t query_ix,
    // the current inverted list index
    const uint32_t cluster_ix,
    // the index of the current sample inside the current inverted list
    const uint32_t sample_ix) const
  {
    return true;
  }
};

template <typename filter_t, typename = void>
struct takes_three_args : cuda::std::false_type {};
template <typename filter_t>
struct takes_three_args<
  filter_t,
  cuda::std::void_t<decltype(cuda::std::declval<filter_t>()(uint32_t{}, uint32_t{}, uint32_t{}))>>
  : cuda::std::true_type {};

template <typename index_t, typename filter_t>
struct ivf_to_sample_filter {
  const index_t* const* inds_ptrs_;
  const filter_t next_filter_;

  ivf_to_sample_filter(const index_t* const* inds_ptrs, const filter_t next_filter)
    : inds_ptrs_{inds_ptrs}, next_filter_{next_filter}
  {
  }

  /** If the original filter takes three arguments, then don't modify the arguments.
   * If the original filter takes two arguments, then we are using `inds_ptr_` to obtain the sample
   * index.
   */
  inline _RAFT_HOST_DEVICE bool operator()(
    // query index
    const uint32_t query_ix,
    // the current inverted list index
    const uint32_t cluster_ix,
    // the index of the current sample inside the current inverted list
    const uint32_t sample_ix) const
  {
    if constexpr (takes_three_args<filter_t>::value) {
      return next_filter_(query_ix, cluster_ix, sample_ix);
    } else {
      return next_filter_(query_ix, inds_ptrs_[cluster_ix][sample_ix]);
    }
  }
};

}  // namespace cuvs::neighbors::filtering