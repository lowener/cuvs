/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cuvs/distance/distance.hpp>
#include <raft/core/device_mdarray.hpp>  // raft::make_device_matrix
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/matrix/copy.cuh>
#include <raft/matrix/detail/select_k.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device_memory_resource.hpp>

#include <atomic>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <limits>

namespace cuvs::neighbors {

struct print_dtype {
  cudaDataType_t value;
};

inline auto operator<<(std::ostream& os, const print_dtype& p) -> std::ostream&
{
  switch (p.value) {
    case CUDA_R_16F: os << "CUDA_R_16F"; break;
    case CUDA_C_16F: os << "CUDA_C_16F"; break;
    case CUDA_R_16BF: os << "CUDA_R_16BF"; break;
    case CUDA_C_16BF: os << "CUDA_C_16BF"; break;
    case CUDA_R_32F: os << "CUDA_R_32F"; break;
    case CUDA_C_32F: os << "CUDA_C_32F"; break;
    case CUDA_R_64F: os << "CUDA_R_64F"; break;
    case CUDA_C_64F: os << "CUDA_C_64F"; break;
    case CUDA_R_4I: os << "CUDA_R_4I"; break;
    case CUDA_C_4I: os << "CUDA_C_4I"; break;
    case CUDA_R_4U: os << "CUDA_R_4U"; break;
    case CUDA_C_4U: os << "CUDA_C_4U"; break;
    case CUDA_R_8I: os << "CUDA_R_8I"; break;
    case CUDA_C_8I: os << "CUDA_C_8I"; break;
    case CUDA_R_8U: os << "CUDA_R_8U"; break;
    case CUDA_C_8U: os << "CUDA_C_8U"; break;
    case CUDA_R_16I: os << "CUDA_R_16I"; break;
    case CUDA_C_16I: os << "CUDA_C_16I"; break;
    case CUDA_R_16U: os << "CUDA_R_16U"; break;
    case CUDA_C_16U: os << "CUDA_C_16U"; break;
    case CUDA_R_32I: os << "CUDA_R_32I"; break;
    case CUDA_C_32I: os << "CUDA_C_32I"; break;
    case CUDA_R_32U: os << "CUDA_R_32U"; break;
    case CUDA_C_32U: os << "CUDA_C_32U"; break;
    case CUDA_R_64I: os << "CUDA_R_64I"; break;
    case CUDA_C_64I: os << "CUDA_C_64I"; break;
    case CUDA_R_64U: os << "CUDA_R_64U"; break;
    case CUDA_C_64U: os << "CUDA_C_64U"; break;
    default: RAFT_FAIL("unreachable code");
  }
  return os;
}

struct print_metric {
  cuvs::distance::DistanceType value;
};

inline auto operator<<(std::ostream& os, const print_metric& p) -> std::ostream&
{
  switch (p.value) {
    case cuvs::distance::DistanceType::L2Expanded: os << "distance::L2Expanded"; break;
    case cuvs::distance::DistanceType::L2SqrtExpanded: os << "distance::L2SqrtExpanded"; break;
    case cuvs::distance::DistanceType::CosineExpanded: os << "distance::CosineExpanded"; break;
    case cuvs::distance::DistanceType::L1: os << "distance::L1"; break;
    case cuvs::distance::DistanceType::L2Unexpanded: os << "distance::L2Unexpanded"; break;
    case cuvs::distance::DistanceType::L2SqrtUnexpanded: os << "distance::L2SqrtUnexpanded"; break;
    case cuvs::distance::DistanceType::InnerProduct: os << "distance::InnerProduct"; break;
    case cuvs::distance::DistanceType::Linf: os << "distance::Linf"; break;
    case cuvs::distance::DistanceType::Canberra: os << "distance::Canberra"; break;
    case cuvs::distance::DistanceType::LpUnexpanded: os << "distance::LpUnexpanded"; break;
    case cuvs::distance::DistanceType::CorrelationExpanded:
      os << "distance::CorrelationExpanded";
      break;
    case cuvs::distance::DistanceType::JaccardExpanded: os << "distance::JaccardExpanded"; break;
    case cuvs::distance::DistanceType::HellingerExpanded:
      os << "distance::HellingerExpanded";
      break;
    case cuvs::distance::DistanceType::Haversine: os << "distance::Haversine"; break;
    case cuvs::distance::DistanceType::BrayCurtis: os << "distance::BrayCurtis"; break;
    case cuvs::distance::DistanceType::JensenShannon: os << "distance::JensenShannon"; break;
    case cuvs::distance::DistanceType::HammingUnexpanded:
      os << "distance::HammingUnexpanded";
      break;
    case cuvs::distance::DistanceType::KLDivergence: os << "distance::KLDivergence"; break;
    case cuvs::distance::DistanceType::RusselRaoExpanded:
      os << "distance::RusselRaoExpanded";
      break;
    case cuvs::distance::DistanceType::DiceExpanded: os << "distance::DiceExpanded"; break;
    case cuvs::distance::DistanceType::Precomputed: os << "distance::Precomputed"; break;
    default: RAFT_FAIL("unreachable code");
  }
  return os;
}

template <typename IdxT, typename DistT, typename CompareDist>
struct idx_dist_pair {
  IdxT idx;
  DistT dist;
  CompareDist eq_compare;
  auto operator==(const idx_dist_pair<IdxT, DistT, CompareDist>& a) const -> bool
  {
    if (idx == a.idx) return true;
    if (eq_compare(dist, a.dist)) return true;
    return false;
  }
  idx_dist_pair(IdxT x, DistT y, CompareDist op) : idx(x), dist(y), eq_compare(op) {}
};

/** Calculate recall value using only neighbor indices
 */
template <typename T>
auto calc_recall(const std::vector<T>& expected_idx,
                 const std::vector<T>& actual_idx,
                 size_t rows,
                 size_t cols)
{
  size_t match_count = 0;
  size_t total_count = static_cast<size_t>(rows) * static_cast<size_t>(cols);
  for (size_t i = 0; i < rows; ++i) {
    for (size_t k = 0; k < cols; ++k) {
      size_t idx_k = i * cols + k;  // row major assumption!
      auto act_idx = actual_idx[idx_k];
      for (size_t j = 0; j < cols; ++j) {
        size_t idx   = i * cols + j;  // row major assumption!
        auto exp_idx = expected_idx[idx];
        if (act_idx == exp_idx) {
          match_count++;
          break;
        }
      }
    }
  }
  return std::make_tuple(
    static_cast<double>(match_count) / static_cast<double>(total_count), match_count, total_count);
}

}  // namespace cuvs::neighbors
