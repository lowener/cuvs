/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/preprocessing/quantize/bbq.hpp>

#include <raft/core/copy.cuh>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/random/rng.cuh>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <optional>
#include <sstream>
#include <utility>
#include <vector>

#include <raft/core/logger.hpp>

namespace cuvs::preprocessing::cpu_bbq {

// Host-side Lucene OptimizedScalarQuantizer
using cuvs::preprocessing::quantize::bbq::bbq_code_layout;

constexpr float kMinimumMseGrid[8][2] = {{-0.798f, 0.798f},
                                         {-1.493f, 1.493f},
                                         {-2.051f, 2.051f},
                                         {-2.514f, 2.514f},
                                         {-2.916f, 2.916f},
                                         {-3.278f, 3.278f},
                                         {-3.611f, 3.611f},
                                         {-3.922f, 3.922f}};

constexpr float kDefaultLambda = 0.1f;
constexpr int kDefaultIters    = 5;

inline long round(double x) { return static_cast<long>(std::floor(x + 0.5)); }

inline double clamp(double x, double a, double b) { return std::min(std::max(x, a), b); }

inline double loss(
  const std::vector<float>& vector, const float interval[2], int points, float norm2, float lambda)
{
  const double a        = interval[0];
  const double b        = interval[1];
  const double step     = (b - a) / (points - 1.0);
  const double step_inv = 1.0 / step;
  double xe             = 0.0;
  double e              = 0.0;
  for (double xi : vector) {
    const double xiq = a + step * static_cast<double>(round((clamp(xi, a, b) - a) * step_inv));
    xe += xi * (xi - xiq);
    e += (xi - xiq) * (xi - xiq);
  }
  return (1.0 - lambda) * xe * xe / norm2 + lambda * e;
}

inline void optimize_intervals(float interval[2],
                               const std::vector<float>& vector,
                               float norm2,
                               int points,
                               float lambda = kDefaultLambda,
                               int iters    = kDefaultIters)
{
  double initial_loss = loss(vector, interval, points, norm2, lambda);
  const float scale   = (1.0f - lambda) / norm2;
  if (!std::isfinite(scale)) { return; }
  for (int i = 0; i < iters; ++i) {
    const float a        = interval[0];
    const float b        = interval[1];
    const float step_inv = (points - 1.0f) / (b - a);
    double daa = 0.0, dab = 0.0, dbb = 0.0, dax = 0.0, dbx = 0.0;
    for (float xi : vector) {
      const float k =
        static_cast<float>(round(static_cast<float>((clamp(xi, a, b) - a) * step_inv)));
      const float s = k / (points - 1);
      daa += (1.0 - s) * (1.0 - s);
      dab += (1.0 - s) * s;
      dbb += s * s;
      dax += xi * (1.0 - s);
      dbx += xi * s;
    }
    const double m0  = scale * dax * dax + lambda * daa;
    const double m1  = scale * dax * dbx + lambda * dab;
    const double m2  = scale * dbx * dbx + lambda * dbb;
    const double det = m0 * m2 - m1 * m1;
    if (det == 0) { return; }
    const float a_opt = static_cast<float>((m2 * dax - m1 * dbx) / det);
    const float b_opt = static_cast<float>((m0 * dbx - m1 * dax) / det);
    if (std::abs(interval[0] - a_opt) < 1e-8 && std::abs(interval[1] - b_opt) < 1e-8) { return; }
    float new_interval[2] = {a_opt, b_opt};
    const double new_loss = loss(vector, new_interval, points, norm2, lambda);
    if (new_loss > initial_loss) { return; }
    interval[0]  = a_opt;
    interval[1]  = b_opt;
    initial_loss = new_loss;
  }
}

struct row_result {
  float lower_interval;
  float upper_interval;
  float additional_correction;
  int32_t quantized_component_sum;
};

template <typename DataT>
inline row_result scalar_quantize(std::vector<DataT>& vector,
                                  std::vector<uint8_t>& destination,
                                  uint8_t bits,
                                  const DataT* centroid,
                                  bool euclidean)
{
  const int n        = static_cast<int>(vector.size());
  const int points   = 1 << bits;
  double vec_mean    = 0.0;
  double vec_var     = 0.0;
  float norm2        = 0.0f;
  DataT centroid_dot = DataT(0.0);
  DataT min          = std::numeric_limits<DataT>::max();
  DataT max          = std::numeric_limits<DataT>::min();
  for (int i = 0; i < n; ++i) {
    if (!euclidean) { centroid_dot += vector[i] * centroid[i]; }
    vector[i] = vector[i] - centroid[i];
    min       = std::min(min, vector[i]);
    max       = std::max(max, vector[i]);
    norm2 += vector[i] * vector[i];
    const double delta = vector[i] - vec_mean;
    vec_mean += delta / (i + 1);
    vec_var += delta * (vector[i] - vec_mean);
  }
  vec_var /= n;
  const double vec_std = std::sqrt(vec_var);

  float interval[2];
  interval[0] =
    static_cast<float>(clamp(kMinimumMseGrid[bits - 1][0] * vec_std + vec_mean, min, max));
  interval[1] =
    static_cast<float>(clamp(kMinimumMseGrid[bits - 1][1] * vec_std + vec_mean, min, max));
  optimize_intervals(interval, vector, norm2, points);

  const float n_steps = static_cast<float>((1 << bits) - 1);
  const float a       = interval[0];
  const float b       = interval[1];
  const float step    = (b - a) / n_steps;
  int sum_query       = 0;
  for (int h = 0; h < n; ++h) {
    const float xi       = static_cast<float>(clamp(vector[h], a, b));
    const int assignment = static_cast<int>(round((xi - a) / step));
    sum_query += assignment;
    destination[h] = static_cast<uint8_t>(assignment);
  }
  return row_result{interval[0], interval[1], euclidean ? norm2 : centroid_dot, sum_query};
}

inline size_t encoded_row_length(size_t dim, uint32_t bits, bbq_code_layout layout)
{
  switch (layout) {
    case bbq_code_layout::single_bit: return (dim * bits + 7) / 8;
    case bbq_code_layout::dibit: return bits * ((dim + 7) / 8);
    case bbq_code_layout::packed_nibble: return (dim + 1) / 2;
    case bbq_code_layout::seven_bit: return dim;
    case bbq_code_layout::unsigned_byte: return dim;
    case bbq_code_layout::transpose_half_byte: return 4 * ((dim + 7) / 8);
  }
  return 0;
}

// Packs one-byte-per-component codes into single_bit / dibit / packed_nibble /
// transpose_half_byte (or leaves unpacked). Matches Lucene packAsBinary,
// packNibbles, transposeDibit, transposeHalfByte.
inline std::vector<uint8_t> pack_codes(const std::vector<uint8_t>& unpacked,
                                       size_t n_rows,
                                       size_t dim,
                                       uint32_t bits,
                                       bbq_code_layout layout)
{
  const size_t row_length = encoded_row_length(dim, bits, layout);
  if (layout == bbq_code_layout::unsigned_byte || layout == bbq_code_layout::seven_bit) {
    return unpacked;
  }

  std::vector<uint8_t> packed(n_rows * row_length, 0);
  for (size_t row = 0; row < n_rows; ++row) {
    auto* output      = packed.data() + row * row_length;
    const auto* input = unpacked.data() + row * dim;
    if (layout == bbq_code_layout::packed_nibble) {
      // Lucene OffHeapScalarQuantizedVectorValues.packNibbles
      const size_t half = dim / 2;
      for (size_t i = 0; i < half; ++i) {
        output[i] = static_cast<uint8_t>((input[i] << 4) | (input[half + i] & 0x0f));
      }
      continue;
    }
    for (size_t d = 0; d < dim; ++d) {
      const uint8_t code = input[d];
      if (layout == bbq_code_layout::single_bit) {
        for (uint32_t bit = 0; bit < bits; ++bit) {
          const size_t position = d * bits + bit;
          output[position / 8] |=
            static_cast<uint8_t>(((code >> (bits - 1 - bit)) & 1u) << (7 - position % 8));
        }
      } else {
        // dibit / transpose_half_byte bit-planes (LSB plane first)
        const size_t stripe = (dim + 7) / 8;
        for (uint32_t bit = 0; bit < bits; ++bit) {
          output[bit * stripe + d / 8] |= static_cast<uint8_t>(((code >> bit) & 1u) << (7 - d % 8));
        }
      }
    }
  }
  return packed;
}

template <typename DataT>
inline cuvs::preprocessing::quantize::bbq::bbq_quantizer<DataT, int64_t> quantize_on_cpu(
  raft::resources const& res,
  const std::vector<DataT>& data,
  int64_t n_rows,
  int64_t dim,
  uint8_t bits,
  cuvs::distance::DistanceType metric,
  bbq_code_layout layout = bbq_code_layout::unsigned_byte)
{
  const bool euclidean = metric == cuvs::distance::DistanceType::L2Expanded ||
                         metric == cuvs::distance::DistanceType::L2SqrtExpanded;

  auto centroid          = raft::make_host_vector<DataT, int64_t>(dim);
  auto centroid_d        = raft::make_device_vector<DataT, int64_t>(res, dim);
  float centroid_norm_sq = 0.0f;
  std::fill_n(centroid.data_handle(), static_cast<size_t>(dim), 0.0f);
  for (int64_t i = 0; i < n_rows; ++i) {
    for (int64_t d = 0; d < dim; ++d) {
      centroid(d) += DataT(data[i * dim + d]);
    }
  }
  for (int64_t d = 0; d < dim; ++d) {
    centroid(d) /= static_cast<DataT>(n_rows);
    centroid_norm_sq += centroid(d) * centroid(d);
  }

  std::vector<uint8_t> unpacked(static_cast<size_t>(n_rows * dim));
  auto lower_intervals            = raft::make_host_vector<float, int64_t>(n_rows);
  auto upper_intervals            = raft::make_host_vector<float, int64_t>(n_rows);
  auto additional_corrections     = raft::make_host_vector<float, int64_t>(n_rows);
  auto quantized_component_sums   = raft::make_host_vector<int32_t, int64_t>(n_rows);
  auto lower_intervals_d          = raft::make_device_vector<float, int64_t>(res, n_rows);
  auto upper_intervals_d          = raft::make_device_vector<float, int64_t>(res, n_rows);
  auto additional_corrections_d   = raft::make_device_vector<float, int64_t>(res, n_rows);
  auto quantized_component_sums_d = raft::make_device_vector<int32_t, int64_t>(res, n_rows);

  for (int64_t i = 0; i < n_rows; ++i) {
    std::vector<DataT> row(data.begin() + i * dim, data.begin() + (i + 1) * dim);
    std::vector<uint8_t> codes(dim);
    const auto result = scalar_quantize(row, codes, bits, centroid.data_handle(), euclidean);
    std::copy(codes.begin(), codes.end(), unpacked.begin() + i * dim);
    lower_intervals(i)          = result.lower_interval;
    upper_intervals(i)          = result.upper_interval;
    additional_corrections(i)   = result.additional_correction;
    quantized_component_sums(i) = result.quantized_component_sum;
  }

  auto packed =
    pack_codes(unpacked, static_cast<size_t>(n_rows), static_cast<size_t>(dim), bits, layout);
  auto codes = raft::make_device_matrix<uint8_t, int64_t, raft::layout_c_contiguous>(
    res, n_rows, static_cast<int64_t>(encoded_row_length(dim, bits, layout)));
  raft::copy(codes.data_handle(),
             packed.data(),
             codes.extent(0) * codes.extent(1),
             raft::resource::get_cuda_stream(res));
  raft::copy(res, lower_intervals_d.view(), lower_intervals.view());
  raft::copy(res, upper_intervals_d.view(), upper_intervals.view());
  raft::copy(res, additional_corrections_d.view(), additional_corrections.view());
  raft::copy(res, quantized_component_sums_d.view(), quantized_component_sums.view());
  raft::copy(res, centroid_d.view(), centroid.view());
  raft::resource::sync_stream(res);
  return cuvs::preprocessing::quantize::bbq::bbq_quantizer<DataT, int64_t>{
    std::move(codes),
    std::move(lower_intervals_d),
    std::move(upper_intervals_d),
    std::move(additional_corrections_d),
    std::move(quantized_component_sums_d),
    std::move(centroid_d),
    static_cast<uint32_t>(bits),
    layout,
    metric,
    centroid_norm_sq};
}

}  // namespace cuvs::preprocessing::cpu_bbq
