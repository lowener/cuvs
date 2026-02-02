/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../../core/nvtx.hpp"
#include "../../core/omp_wrapper.hpp"

#include <raft/core/host_mdspan.hpp>

#include <algorithm>

namespace cuvs::neighbors {

namespace detail {

/**
 * @brief Compute recall on host using CPU implementation.
 *
 * This follows the algorithm from cpp/tests/neighbors/ann_utils.cuh:calc_recall
 * but with OpenMP parallelization for better performance.
 *
 * @tparam IdxT Index type
 * @tparam ExtentsT Matrix extents type
 * @param expected_indices Ground truth neighbor indices [n_queries, k]
 * @param actual_indices Actual neighbor indices from ANN search [n_queries, k]
 * @return recall value as float
 */
template <typename IdxT, typename ExtentsT>
[[gnu::optimize(3), gnu::optimize("tree-vectorize")]] float recall_host(
  raft::host_matrix_view<const IdxT, ExtentsT, raft::row_major> expected_indices,
  raft::host_matrix_view<const IdxT, ExtentsT, raft::row_major> actual_indices)
{
  size_t n_queries   = expected_indices.extent(0);
  size_t k           = expected_indices.extent(1);
  size_t total_count = n_queries * k;

  if (total_count == 0) { return 0.0f; }

  cuvs::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> fun_scope(
    "neighbors::recall_host(%zu, %zu)", n_queries, k);

  size_t match_count = 0;

  // Parallelize over queries
#pragma omp parallel reduction(+ : match_count)
  {
#pragma omp for schedule(static)
    for (size_t i = 0; i < n_queries; ++i) {
      // For each actual neighbor in this query
      for (size_t j = 0; j < k; ++j) {
        IdxT act_idx = actual_indices(i, j);

        // Search for this actual index in the expected neighbors
        bool found = false;
        for (size_t l = 0; l < k; ++l) {
          IdxT exp_idx = expected_indices(i, l);
          if (act_idx == exp_idx) {
            found = true;
            break;
          }
        }

        if (found) { match_count++; }
      }
    }
  }

  float recall = static_cast<float>(match_count) / static_cast<float>(total_count);
  return recall;
}

}  // namespace detail

/**
 * @brief Host implementation wrapper for recall computation.
 */
template <typename IdxT, typename ExtentsT>
float recall_impl(raft::resources const& handle,
                  raft::host_matrix_view<const IdxT, ExtentsT, raft::row_major> expected_indices,
                  raft::host_matrix_view<const IdxT, ExtentsT, raft::row_major> actual_indices)
{
  // Validate input dimensions
  RAFT_EXPECTS(expected_indices.extent(0) == actual_indices.extent(0),
               "Number of queries must match between expected and actual indices");
  RAFT_EXPECTS(expected_indices.extent(1) == actual_indices.extent(1),
               "Number of neighbors (k) must match between expected and actual indices");

  return detail::recall_host(expected_indices, actual_indices);
}

}  // namespace cuvs::neighbors
