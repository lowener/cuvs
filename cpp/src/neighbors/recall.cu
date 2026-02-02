/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "recall/recall_device.cuh"
#include "recall/recall_host.hpp"

#include <cuvs/neighbors/recall.hpp>

namespace cuvs::neighbors {

// Device implementations
template <typename IdxT, typename ExtentsT>
float compute_recall(
  raft::resources const& handle,
  raft::device_matrix_view<const IdxT, ExtentsT, raft::row_major> expected_indices,
  raft::device_matrix_view<const IdxT, ExtentsT, raft::row_major> actual_indices)
{
  return recall_impl(handle, expected_indices, actual_indices);
}

// Host implementations
template <typename IdxT, typename ExtentsT>
float compute_recall(raft::resources const& handle,
                     raft::host_matrix_view<const IdxT, ExtentsT, raft::row_major> expected_indices,
                     raft::host_matrix_view<const IdxT, ExtentsT, raft::row_major> actual_indices)
{
  return recall_impl(handle, expected_indices, actual_indices);
}

// Explicit template instantiations for device matrices
template float compute_recall(
  raft::resources const& handle,
  raft::device_matrix_view<const int32_t, int64_t, raft::row_major> expected_indices,
  raft::device_matrix_view<const int32_t, int64_t, raft::row_major> actual_indices);

template float compute_recall(
  raft::resources const& handle,
  raft::device_matrix_view<const int64_t, int64_t, raft::row_major> expected_indices,
  raft::device_matrix_view<const int64_t, int64_t, raft::row_major> actual_indices);

template float compute_recall(
  raft::resources const& handle,
  raft::device_matrix_view<const uint32_t, int64_t, raft::row_major> expected_indices,
  raft::device_matrix_view<const uint32_t, int64_t, raft::row_major> actual_indices);

// Explicit template instantiations for host matrices
template float compute_recall(
  raft::resources const& handle,
  raft::host_matrix_view<const int32_t, int64_t, raft::row_major> expected_indices,
  raft::host_matrix_view<const int32_t, int64_t, raft::row_major> actual_indices);

template float compute_recall(
  raft::resources const& handle,
  raft::host_matrix_view<const int64_t, int64_t, raft::row_major> expected_indices,
  raft::host_matrix_view<const int64_t, int64_t, raft::row_major> actual_indices);

template float compute_recall(
  raft::resources const& handle,
  raft::host_matrix_view<const uint32_t, int64_t, raft::row_major> expected_indices,
  raft::host_matrix_view<const uint32_t, int64_t, raft::row_major> actual_indices);

}  // namespace cuvs::neighbors
