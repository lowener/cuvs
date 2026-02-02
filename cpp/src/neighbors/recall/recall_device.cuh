/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/neighbors/common.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/map.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>

namespace cuvs::neighbors {
namespace detail {

/**
 * @brief Optimized CUDA kernel to compute recall using shared memory and block reductions.
 *
 * This kernel processes one query per block, using shared memory to cache expected indices
 * and block-level reductions to minimize atomic operations.
 *
 * @tparam IdxT Index type
 * @tparam BlockSize Number of threads per block
 * @param n_queries Number of queries (rows)
 * @param k Number of neighbors per query (columns)
 * @param expected_indices Ground truth neighbor indices [n_queries, k]
 * @param actual_indices Actual neighbor indices from ANN search [n_queries, k]
 * @param match_count Output: number of matches found per block [gridDim.x]
 */
template <typename IdxT, int BlockSize>
__global__ void compute_recall_kernel_optimized(size_t n_queries,
                                                size_t k,
                                                const IdxT* expected_indices,
                                                const IdxT* actual_indices,
                                                size_t* match_count)
{
  extern __shared__ char shared_mem[];
  IdxT* expected_shared = reinterpret_cast<IdxT*>(shared_mem);
  size_t* block_matches = reinterpret_cast<size_t*>(expected_shared + k);

  size_t query_idx = blockIdx.x;
  if (query_idx >= n_queries) return;

  size_t tid = threadIdx.x;

  // Initialize block match counter
  if (tid == 0) { *block_matches = 0; }
  __syncthreads();

  // Load expected indices into shared memory cooperatively
  size_t expected_offset = query_idx * k;
  for (size_t i = tid; i < k; i += BlockSize) {
    expected_shared[i] = expected_indices[expected_offset + i];
  }
  __syncthreads();

  // Each thread processes multiple actual neighbors
  size_t actual_offset = query_idx * k;
  size_t local_matches = 0;

  for (size_t i = tid; i < k; i += BlockSize) {
    IdxT act_idx = actual_indices[actual_offset + i];

    // Search in shared memory (much faster than global memory)
    for (size_t j = 0; j < k; ++j) {
      if (act_idx == expected_shared[j]) {
        local_matches++;
        break;
      }
    }
  }

  // Use warp-level reduction first
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    local_matches += __shfl_down_sync(0xffffffff, local_matches, offset);
  }

  // First thread in each warp writes to shared memory
  __shared__ size_t warp_matches[BlockSize / 32];
  int lane    = tid % warpSize;
  int warp_id = tid / warpSize;

  if (lane == 0) { warp_matches[warp_id] = local_matches; }
  __syncthreads();

  // Final reduction by first warp
  if (warp_id == 0) {
    local_matches = (tid < (BlockSize / 32)) ? warp_matches[lane] : 0;

    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
      local_matches += __shfl_down_sync(0xffffffff, local_matches, offset);
    }

    if (tid == 0) { atomicAdd(match_count, local_matches); }
  }
}

/**
 * @brief Compute recall on device using GPU kernel.
 *
 * @tparam IdxT Index type
 * @tparam ExtentsT Matrix extents type
 * @param handle RAFT resources handle
 * @param expected_indices Ground truth neighbor indices [n_queries, k]
 * @param actual_indices Actual neighbor indices from ANN search [n_queries, k]
 * @return recall value as float
 */
template <typename IdxT, typename ExtentsT>
float recall_device(
  raft::resources const& handle,
  raft::device_matrix_view<const IdxT, ExtentsT, raft::row_major> expected_indices,
  raft::device_matrix_view<const IdxT, ExtentsT, raft::row_major> actual_indices)
{
  // Validate input dimensions
  RAFT_EXPECTS(expected_indices.extent(0) == actual_indices.extent(0),
               "Number of queries must match between expected and actual indices");
  RAFT_EXPECTS(expected_indices.extent(1) == actual_indices.extent(1),
               "Number of neighbors (k) must match between expected and actual indices");

  size_t n_queries   = expected_indices.extent(0);
  size_t k           = expected_indices.extent(1);
  size_t total_count = n_queries * k;

  if (total_count == 0) { return 0.0f; }

  auto stream = raft::resource::get_cuda_stream(handle);

  // Get device properties to check shared memory limits
  int device;
  RAFT_CUDA_TRY(cudaGetDevice(&device));
  cudaDeviceProp prop;
  RAFT_CUDA_TRY(cudaGetDeviceProperties(&prop, device));

  constexpr int threads_per_block = 256;
  size_t shared_mem_size          = k * sizeof(IdxT) + sizeof(size_t);

  size_t match_count_host = 0;

  // Use optimized shared memory kernel if k is small enough
  if (shared_mem_size <= prop.sharedMemPerBlock) {
    // Allocate device memory for match count
    auto match_count_device = raft::make_device_scalar<size_t>(handle, 0);

    // One block per query for better occupancy
    const int num_blocks = n_queries;

    compute_recall_kernel_optimized<IdxT, threads_per_block>
      <<<num_blocks, threads_per_block, shared_mem_size, stream>>>(
        n_queries,
        k,
        expected_indices.data_handle(),
        actual_indices.data_handle(),
        match_count_device.data_handle());

    RAFT_CUDA_TRY(cudaPeekAtLastError());

    // Copy result back to host
    raft::copy(&match_count_host, match_count_device.data_handle(), 1, stream);
    raft::resource::sync_stream(handle, stream);
  } else {
    // Fallback: use thrust reduction for very large k
    // This approach is used when k is too large for shared memory
    auto lambda = [=] __device__(size_t idx) -> size_t {
      size_t query_idx = idx / k;
      if (query_idx >= n_queries) return 0;

      IdxT act_idx           = actual_indices[idx];
      size_t expected_offset = query_idx * k;

      // Linear search in global memory
      for (size_t j = 0; j < k; ++j) {
        if (act_idx == expected_indices[expected_offset + j]) { return 1; }
      }
      return 0;
    };

    // Use device-wide reduction
    auto policy         = raft::resource::get_thrust_policy(handle);
    auto counting_iter  = thrust::make_counting_iterator<size_t>(0);
    auto transform_iter = thrust::make_transform_iterator(counting_iter, lambda);

    match_count_host = thrust::reduce(
      policy, transform_iter, transform_iter + total_count, size_t(0), thrust::plus<size_t>());
  }

  // Compute recall
  float recall = static_cast<float>(match_count_host) / static_cast<float>(total_count);

  return recall;
}

}  // namespace detail

/**
 * @brief Device implementation wrapper for recall computation.
 */
template <typename IdxT, typename ExtentsT>
float recall_impl(raft::resources const& handle,
                  raft::device_matrix_view<const IdxT, ExtentsT, raft::row_major> expected_indices,
                  raft::device_matrix_view<const IdxT, ExtentsT, raft::row_major> actual_indices)
{
  return detail::recall_device(handle, expected_indices, actual_indices);
}

}  // namespace cuvs::neighbors
