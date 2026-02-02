/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/neighbors/common.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/error.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/resources.hpp>

namespace cuvs::neighbors {
/**
 * @defgroup ann_recall Approximate Nearest Neighbors Recall Computation
 * @{
 */

/**
 * @brief Compute recall for approximate nearest neighbor search results.
 *
 * Recall is computed by comparing the actual (approximate) neighbor indices against
 * the expected (ground truth) neighbor indices. For each query, we count how many
 * of the actual neighbors appear anywhere in the expected neighbors for that query.
 *
 * The recall is defined as: match_count / total_count
 * where match_count is the number of matches found and total_count is n_queries * k.
 *
 * Example usage
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // Perform ANN search
 *   ivf_pq::search(handle, search_params, index, queries, neighbors, distances);
 *   // Compute recall against ground truth
 *   float recall = compute_recall(handle, ground_truth_neighbors, neighbors);
 *   std::cout << "Recall: " << recall << std::endl;
 * @endcode
 *
 * @tparam IdxT Index type (int32_t, int64_t, or uint32_t)
 * @tparam ExtentsT Matrix extents type
 *
 * @param[in] handle the raft handle
 * @param[in] expected_indices device matrix of ground truth neighbor indices [n_queries, k]
 * @param[in] actual_indices device matrix of actual neighbor indices [n_queries, k]
 *
 * @return recall value as a float (range: 0.0 to 1.0)
 */
template <typename IdxT, typename ExtentsT = int64_t>
float compute_recall(
  raft::resources const& handle,
  raft::device_matrix_view<const IdxT, ExtentsT, raft::row_major> expected_indices,
  raft::device_matrix_view<const IdxT, ExtentsT, raft::row_major> actual_indices);

/**
 * @brief Compute recall for approximate nearest neighbor search results (host variant).
 *
 * Recall is computed by comparing the actual (approximate) neighbor indices against
 * the expected (ground truth) neighbor indices. For each query, we count how many
 * of the actual neighbors appear anywhere in the expected neighbors for that query.
 *
 * The recall is defined as: match_count / total_count
 * where match_count is the number of matches found and total_count is n_queries * k.
 *
 * This variant accepts host matrices. The computation is performed on the CPU using
 * multi-threaded OpenMP parallelization.
 *
 * Example usage
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // Load ground truth and results from host memory
 *   float recall = compute_recall(handle, ground_truth_neighbors, neighbors);
 *   std::cout << "Recall: " << recall << std::endl;
 * @endcode
 *
 * @tparam IdxT Index type (int32_t, int64_t, or uint32_t)
 * @tparam ExtentsT Matrix extents type
 *
 * @param[in] handle the raft handle
 * @param[in] expected_indices host matrix of ground truth neighbor indices [n_queries, k]
 * @param[in] actual_indices host matrix of actual neighbor indices [n_queries, k]
 *
 * @return recall value as a float (range: 0.0 to 1.0)
 */
template <typename IdxT, typename ExtentsT = int64_t>
float compute_recall(raft::resources const& handle,
                     raft::host_matrix_view<const IdxT, ExtentsT, raft::row_major> expected_indices,
                     raft::host_matrix_view<const IdxT, ExtentsT, raft::row_major> actual_indices);

// Explicit instantiation declarations for common types
extern template float compute_recall(
  raft::resources const& handle,
  raft::device_matrix_view<const int32_t, int64_t, raft::row_major> expected_indices,
  raft::device_matrix_view<const int32_t, int64_t, raft::row_major> actual_indices);

extern template float compute_recall(
  raft::resources const& handle,
  raft::device_matrix_view<const int64_t, int64_t, raft::row_major> expected_indices,
  raft::device_matrix_view<const int64_t, int64_t, raft::row_major> actual_indices);

extern template float compute_recall(
  raft::resources const& handle,
  raft::device_matrix_view<const uint32_t, int64_t, raft::row_major> expected_indices,
  raft::device_matrix_view<const uint32_t, int64_t, raft::row_major> actual_indices);

extern template float compute_recall(
  raft::resources const& handle,
  raft::host_matrix_view<const int32_t, int64_t, raft::row_major> expected_indices,
  raft::host_matrix_view<const int32_t, int64_t, raft::row_major> actual_indices);

extern template float compute_recall(
  raft::resources const& handle,
  raft::host_matrix_view<const int64_t, int64_t, raft::row_major> expected_indices,
  raft::host_matrix_view<const int64_t, int64_t, raft::row_major> actual_indices);

extern template float compute_recall(
  raft::resources const& handle,
  raft::host_matrix_view<const uint32_t, int64_t, raft::row_major> expected_indices,
  raft::host_matrix_view<const uint32_t, int64_t, raft::row_major> actual_indices);

/**
 * @}
 */

}  // namespace cuvs::neighbors
