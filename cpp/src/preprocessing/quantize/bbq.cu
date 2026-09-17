/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <optional>
#include <cuvs/preprocessing/quantize/bbq.hpp>
#include <raft/linalg/map.cuh>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>

namespace CUVS_EXPORT cuvs {

namespace preprocessing::quantize::bbq {

namespace helpers {
auto resolve_dequant_delta(
    raft::resources& res,
    const raft::device_mdarray<float, raft::vector_extent<int64_t>>& lower_intervals,
    const raft::device_mdarray<float, raft::vector_extent<int64_t>>& upper_intervals,
    uint32_t bits) -> raft::device_mdarray<float, raft::vector_extent<int64_t>>
{
  const auto n_rows = lower_intervals.extent(0);
  auto out = raft::make_device_vector<float, int64_t>(res, n_rows);
  const float scale = 1.0f / static_cast<float>((uint32_t{1} << bits) - 1);
  raft::linalg::map(res,
                    lower_intervals.view(),
                    upper_intervals.view(),
                    out.view(),
                    [scale] __device__(float lo, float up) { return (up - lo) * scale; });
  return out;
}

auto resolve_dequant_sum_delta(
  raft::resources& res,
  const raft::device_mdarray<float, raft::vector_extent<int64_t>>& dequant_delta,
  const raft::device_mdarray<int32_t, raft::vector_extent<int64_t>>& quantized_component_sums)
  -> raft::device_mdarray<float, raft::vector_extent<int64_t>>
{
  const auto n_rows = dequant_delta.extent(0);
  auto out = raft::make_device_vector<float, int64_t>(res, n_rows);
  raft::linalg::map(
    res,
    dequant_delta.view(),
    quantized_component_sums.view(),
    out.view(),
    [] __device__(float delta, int32_t sum) { return delta * static_cast<float>(sum); });
  return out;
}
} // namespace helpers
} // namespace cuvs::preprocessing::quantize::bbq
} // namespace cuvs