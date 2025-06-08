/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#define F_OFD_SETLKW 36
#include <jitify2.hpp>

#include "../detail/ann_utils.cuh"
#include "../ivf_common.cuh"
#include "../sample_filter.cuh"
#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/common.hpp>
#include <cuvs/neighbors/ivf_flat.hpp>
#include <cuvs/tools/operators.cuh>
#include <raft/core/logger-ext.hpp>  // RAFT_LOG_TRACE
#include <raft/core/operators.hpp>
#include <raft/matrix/detail/select_warpsort.cuh>
#include <raft/util/cuda_rt_essentials.hpp>  // RAFT_CUDA_TRY
#include <raft/util/device_loads_stores.cuh>
#include <raft/util/integer_utils.hpp>
#include <raft/util/pow2_utils.cuh>
#include <raft/util/vectorized.cuh>
#include <rmm/cuda_stream_view.hpp>

#include "jit_preprocessed_files/neighbors/ivf_flat/ivf_flat_interleaved_scan_kernel.cuh.jit.hpp"

namespace cuvs::neighbors::ivf_flat::detail {

using namespace cuvs::spatial::knn::detail;  // NOLINT

constexpr int kThreadsPerBlock = 128;

auto RAFT_WEAK_FUNCTION is_local_topk_feasible(uint32_t k) -> bool
{
  return k <= raft::matrix::detail::select::warpsort::kMaxCapacity;
}
/**
 *  Configure the gridDim.x to maximize GPU occupancy, but reduce the output size
 */
template <typename T>
uint32_t configure_launch_x(uint32_t numQueries, uint32_t n_probes, int32_t sMemSize, T func)
{
  int dev_id;
  RAFT_CUDA_TRY(cudaGetDevice(&dev_id));
  int num_sms;
  RAFT_CUDA_TRY(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev_id));
  int num_blocks_per_sm = 0;
  RAFT_CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &num_blocks_per_sm, func, kThreadsPerBlock, sMemSize));

  size_t min_grid_size = num_sms * num_blocks_per_sm;
  size_t min_grid_x    = raft::ceildiv<size_t>(min_grid_size, numQueries);
  return min_grid_x > n_probes ? n_probes : static_cast<uint32_t>(min_grid_x);
}

template <int Capacity,
          int Veclen,
          bool Ascending,
          bool ComputeNorm,
          typename T,
          typename AccT,
          typename IdxT,
          typename IvfSampleFilterT,
          typename Lambda,
          typename PostLambda>
void launch_kernel(Lambda lambda,
                   PostLambda post_process,
                   const index<T, IdxT>& index,
                   const T* queries,
                   const uint32_t* coarse_index,
                   const uint32_t num_queries,
                   const uint32_t queries_offset,
                   const uint32_t n_probes,
                   const uint32_t k,
                   const uint32_t max_samples,
                   const uint32_t* chunk_indices,
                   IvfSampleFilterT sample_filter,
                   uint32_t* neighbors,
                   float* distances,
                   uint32_t& grid_dim_x,
                   rmm::cuda_stream_view stream)
{
  RAFT_EXPECTS(Veclen == index.veclen(),
               "Configured Veclen does not match the index interleaving pattern.");

  /*auto program = jitify2::Program("my_program", interleaved_scan_kernel_string)
    ->preprocess({"-std=c++14"});
    ->compile("interleaved_scan_kernel")
    ->link()
    ->load();*/
  // static jitify2::ProgramCache<> kernel_cache(/*max_size = */ 100, program);
  //  jitify2::Program program = kernel_cache.program(interleaved_scan_kernel_string);
  using jitify2::ProgramCache;
  using jitify2::reflection::reflect;
  using jitify2::reflection::Template;
  using jitify2::reflection::type_of;  //  type_of(*d_data)

  static ProgramCache<> scan_kernel_cache(
    /*max_size = */ 100, *neighbors_ivf_flat_ivf_flat_interleaved_scan_kernel_cuh_jit);

  auto kernel_inst = scan_kernel_cache.get_kernel(
    /*Template("cuvs::neighbors::ivf_flat::detail::interleaved_scan_kernel")
      .instantiate(Capacity,
                   Veclen,
                   Ascending,
                   ComputeNorm,
                   reflect<T>(),
                   reflect<AccT>(),
                   reflect<IdxT>(),
                   reflect<IvfSampleFilterT>(),
                   reflect<Lambda>(),
                   reflect<PostLambda>()));*/

    Template("cuvs::neighbors::ivf_flat::detail::my_test_kernel")
      .instantiate(reflect<T>(), reflect<IvfSampleFilterT>()));

  /*constexpr auto kKernel   = interleaved_scan_kernel<Capacity,
                                                     Veclen,
                                                     Ascending,
                                                     ComputeNorm,
                                                     T,
                                                     AccT,
                                                     IdxT,
                                                     IvfSampleFilterT,
                                                     Lambda,
                                                     PostLambda>;*/
  const int max_query_smem = 16384;
  int query_smem_elems     = std::min<int>(max_query_smem / sizeof(T),
                                       raft::Pow2<Veclen * raft::WarpSize>::roundUp(index.dim()));
  int smem_size            = query_smem_elems * sizeof(T);

  if constexpr (Capacity > 0) {
    constexpr int kSubwarpSize = std::min<int>(Capacity, raft::WarpSize);
    auto block_merge_mem =
      raft::matrix::detail::select::warpsort::calc_smem_size_for_block_wide<float, IdxT>(
        kThreadsPerBlock / kSubwarpSize, k);
    smem_size += std::max<int>(smem_size, block_merge_mem);
  }

  // power-of-two less than cuda limit (for better addr alignment)
  constexpr uint32_t kMaxGridY = 32768;

  /*if (grid_dim_x == 0) {
    grid_dim_x = configure_launch_x(std::min(kMaxGridY, num_queries), n_probes, smem_size, kKernel);
    return;
  }*/

  for (uint32_t query_offset = 0; query_offset < num_queries; query_offset += kMaxGridY) {
    uint32_t grid_dim_y = std::min<uint32_t>(kMaxGridY, num_queries - query_offset);
    dim3 grid_dim(grid_dim_x, grid_dim_y, 1);
    dim3 block_dim(kThreadsPerBlock);
    RAFT_LOG_TRACE(
      "Launching the ivf-flat interleaved_scan_kernel (%d, %d, 1) x (%d, 1, 1), n_probes = %d, "
      "smem_size = %d",
      grid_dim.x,
      grid_dim.y,
      block_dim.x,
      n_probes,
      smem_size);
    // configure_1d_max_occupancy
    // Operator cuFunction*
    kernel_inst->configure_1d_max_occupancy(num_queries, smem_size, 0, stream = stream)
      ->launch(/*lambda,
              post_process,*/
               query_smem_elems,
               queries,
               coarse_index,
               index.data_ptrs().data_handle(),
               index.list_sizes().data_handle(),
               queries_offset + query_offset,
               n_probes,
               k,
               max_samples,
               chunk_indices,
               index.dim(),
               sample_filter,
               neighbors,
               distances);
    /*kKernel<<<grid_dim, block_dim, smem_size, stream>>>(lambda,
                                                        post_process,
                                                        query_smem_elems,
                                                        queries,
                                                        coarse_index,
                                                        index.data_ptrs().data_handle(),
                                                        index.list_sizes().data_handle(),
                                                        queries_offset + query_offset,
                                                        n_probes,
                                                        k,
                                                        max_samples,
                                                        chunk_indices,
                                                        index.dim(),
                                                        sample_filter,
                                                        neighbors,
                                                        distances);*/
    queries += grid_dim_y * index.dim();
    if constexpr (Capacity > 0) {
      neighbors += grid_dim_y * grid_dim_x * k;
      distances += grid_dim_y * grid_dim_x * k;
    } else {
      distances += grid_dim_y * max_samples;
    }
    chunk_indices += grid_dim_y * n_probes;
    coarse_index += grid_dim_y * n_probes;
  }
}

template <int Veclen, typename T, typename AccT>
struct euclidean_dist {
  __device__ __forceinline__ void operator()(AccT& acc, AccT x, AccT y)
  {
    const auto diff = x - y;
    acc += diff * diff;
  }
};

template <int Veclen>
struct euclidean_dist<Veclen, uint8_t, uint32_t> {
  __device__ __forceinline__ void operator()(uint32_t& acc, uint32_t x, uint32_t y)
  {
    if constexpr (Veclen > 1) {
      const auto diff = __vabsdiffu4(x, y);
      acc             = raft::dp4a(diff, diff, acc);
    } else {
      const auto diff = __usad(x, y, 0u);
      acc += diff * diff;
    }
  }
};

template <int Veclen>
struct euclidean_dist<Veclen, int8_t, int32_t> {
  __device__ __forceinline__ void operator()(int32_t& acc, int32_t x, int32_t y)
  {
    if constexpr (Veclen > 1) {
      // Note that we enforce here that the unsigned version of dp4a is used, because the difference
      // between two int8 numbers can be greater than 127 and therefore represented as a negative
      // number in int8. Casting from int8 to int32 would yield incorrect results, while casting
      // from uint8 to uint32 is correct.
      const auto diff = __vabsdiffs4(x, y);
      acc             = raft::dp4a(diff, diff, static_cast<uint32_t>(acc));
    } else {
      const auto diff = x - y;
      acc += diff * diff;
    }
  }
};

template <int Veclen, typename T, typename AccT>
struct inner_prod_dist {
  __device__ __forceinline__ void operator()(AccT& acc, AccT x, AccT y)
  {
    if constexpr (Veclen > 1 && (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>)) {
      acc = raft::dp4a(x, y, acc);
    } else {
      acc += x * y;
    }
  }
};

/** Select the distance computation function and forward the rest of the arguments. */
template <int Capacity,
          int Veclen,
          bool Ascending,
          typename T,
          typename AccT,
          typename IdxT,
          typename IvfSampleFilterT,
          typename... Args>
void launch_with_fixed_consts(cuvs::distance::DistanceType metric, Args&&... args)
{
  switch (metric) {
    case cuvs::distance::DistanceType::L2Expanded:
    case cuvs::distance::DistanceType::L2Unexpanded:
      return launch_kernel<Capacity,
                           Veclen,
                           Ascending,
                           false,
                           T,
                           AccT,
                           IdxT,
                           IvfSampleFilterT,
                           euclidean_dist<Veclen, T, AccT>,
                           cuvs::tools::identity_op>({}, {}, std::forward<Args>(args)...);
    case cuvs::distance::DistanceType::L2SqrtExpanded:
    case cuvs::distance::DistanceType::L2SqrtUnexpanded:
      return launch_kernel<Capacity,
                           Veclen,
                           Ascending,
                           false,
                           T,
                           AccT,
                           IdxT,
                           IvfSampleFilterT,
                           euclidean_dist<Veclen, T, AccT>,
                           cuvs::tools::sqrt_op>({}, {}, std::forward<Args>(args)...);
    case cuvs::distance::DistanceType::InnerProduct:
      return launch_kernel<Capacity,
                           Veclen,
                           Ascending,
                           false,
                           T,
                           AccT,
                           IdxT,
                           IvfSampleFilterT,
                           inner_prod_dist<Veclen, T, AccT>,
                           cuvs::tools::identity_op>({}, {}, std::forward<Args>(args)...);
    case cuvs::distance::DistanceType::CosineExpanded:
      // NB: "Ascending" is reversed because the post-processing step is done after that sort
      return launch_kernel<Capacity,
                           Veclen,
                           !Ascending,
                           true,
                           T,
                           AccT,
                           IdxT,
                           IvfSampleFilterT,
                           inner_prod_dist<Veclen, T, AccT>>(
        {},
        raft::compose_op(cuvs::tools::add_const_op<float>{1.0f},
                         cuvs::tools::mul_const_op<float>{-1.0f}),
        std::forward<Args>(args)...);
    // NB: update the description of `knn::ivf_flat::build` when adding here a new metric.
    default: RAFT_FAIL("The chosen distance metric is not supported (%d)", int(metric));
  }
}

/**
 * Lift the `capacity` and `veclen` parameters to the template level,
 * forward the rest of the arguments unmodified to `launch_interleaved_scan_kernel`.
 */
template <typename T,
          typename AccT,
          typename IdxT,
          typename IvfSampleFilterT,
          int Capacity = raft::matrix::detail::select::warpsort::kMaxCapacity,
          int Veclen   = std::max<int>(1, 16 / sizeof(T))>
struct select_interleaved_scan_kernel {
  /**
   * Recursively reduce the `Capacity` and `Veclen` parameters until they match the
   * corresponding runtime arguments.
   * By default, this recursive process starts with maximum possible values of the
   * two parameters and ends with both values equal to 1.
   */
  template <typename... Args>
  static inline void run(int k_max, int veclen, bool select_min, Args&&... args)
  {
    if constexpr (Capacity > 0) {
      if (k_max == 0 || k_max > Capacity) {
        return select_interleaved_scan_kernel<T, AccT, IdxT, IvfSampleFilterT, 0, Veclen>::run(
          k_max, veclen, select_min, std::forward<Args>(args)...);
      }
    }
    if constexpr (Capacity > 1) {
      if (k_max * 2 <= Capacity) {
        return select_interleaved_scan_kernel<T,
                                              AccT,
                                              IdxT,
                                              IvfSampleFilterT,
                                              Capacity / 2,
                                              Veclen>::run(k_max,
                                                           veclen,
                                                           select_min,
                                                           std::forward<Args>(args)...);
      }
    }
    if constexpr (Veclen > 1) {
      if (veclen % Veclen != 0) {
        return select_interleaved_scan_kernel<T, AccT, IdxT, IvfSampleFilterT, Capacity, 1>::run(
          k_max, 1, select_min, std::forward<Args>(args)...);
      }
    }
    // NB: this is the limitation of the warpsort structures that use a huge number of
    //     registers (used in the main kernel here).
    RAFT_EXPECTS(Capacity == 0 || k_max == Capacity,
                 "Capacity must be either 0 or a power-of-two not bigger than the maximum "
                 "allowed size matrix::detail::select::warpsort::kMaxCapacity (%d).",
                 raft::matrix::detail::select::warpsort::kMaxCapacity);
    RAFT_EXPECTS(
      veclen == Veclen,
      "Veclen must be power-of-two not bigger than the maximum allowed size for this data type.");
    if (select_min) {
      launch_with_fixed_consts<Capacity, Veclen, true, T, AccT, IdxT, IvfSampleFilterT>(
        std::forward<Args>(args)...);
    } else {
      launch_with_fixed_consts<Capacity, Veclen, false, T, AccT, IdxT, IvfSampleFilterT>(
        std::forward<Args>(args)...);
    }
  }
};

/**
 * @brief Configure and launch an appropriate template instance of the interleaved scan kernel.
 *
 * @tparam T value type
 * @tparam AccT accumulated type
 * @tparam IdxT type of the indices
 *
 * @param index previously built ivf-flat index
 * @param[in] queries device pointer to the query vectors [batch_size, dim]
 * @param[in] coarse_query_results device pointer to the cluster (list) ids [batch_size, n_probes]
 * @param n_queries batch size
 * @param[in] queries_offset
 *   An offset of the current query batch. It is used for feeding sample_filter with the
 *   correct query index.
 * @param metric type of the measured distance
 * @param n_probes number of nearest clusters to query
 * @param k number of nearest neighbors.
 *            NB: the maximum value of `k` is limited statically by `kMaxCapacity`.
 * @param select_min whether to select nearest (true) or furthest (false) points w.r.t. the given
 * metric.
 * @param[out] neighbors device pointer to the result indices for each query and cluster
 * [batch_size, grid_dim_x, k]
 * @param[out] distances device pointer to the result distances for each query and cluster
 * [batch_size, grid_dim_x, k]
 * @param[inout] grid_dim_x number of blocks launched across all n_probes clusters;
 *               (one block processes one or more probes, hence: 1 <= grid_dim_x <= n_probes)
 * @param stream
 * @param sample_filter
 *   A filter that selects samples for a given query. Use an instance of none_ivf_sample_filter to
 *   provide a green light for every sample.
 */
template <typename T, typename AccT, typename IdxT, typename IvfSampleFilterT>
void ivfflat_interleaved_scan(const index<T, IdxT>& index,
                              const T* queries,
                              const uint32_t* coarse_query_results,
                              const uint32_t n_queries,
                              const uint32_t queries_offset,
                              const cuvs::distance::DistanceType metric,
                              const uint32_t n_probes,
                              const uint32_t k,
                              const uint32_t max_samples,
                              const uint32_t* chunk_indices,
                              const bool select_min,
                              IvfSampleFilterT sample_filter,
                              uint32_t* neighbors,
                              float* distances,
                              uint32_t& grid_dim_x,
                              rmm::cuda_stream_view stream)
{
  const int capacity = raft::bound_by_power_of_two(k);

  auto filter_adapter = cuvs::neighbors::filtering::ivf_to_sample_filter(
    index.inds_ptrs().data_handle(), sample_filter);
  select_interleaved_scan_kernel<T, AccT, IdxT, decltype(filter_adapter)>::run(capacity,
                                                                               index.veclen(),
                                                                               select_min,
                                                                               metric,
                                                                               index,
                                                                               queries,
                                                                               coarse_query_results,
                                                                               n_queries,
                                                                               queries_offset,
                                                                               n_probes,
                                                                               k,
                                                                               max_samples,
                                                                               chunk_indices,
                                                                               filter_adapter,
                                                                               neighbors,
                                                                               distances,
                                                                               grid_dim_x,
                                                                               stream);
}

}  // namespace cuvs::neighbors::ivf_flat::detail
