/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/distance/distance.hpp>
#include <cuvs/preprocessing/quantize/bbq.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/error.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <optional>
#include <string>
#include <utility>
#include <vector>

/**
 * Host-side reference implementation of Lucene's OptimizedScalarQuantizer, used to feed the
 * BBQ NN-Descent build path. This is prototype scaffolding: it is a slow CPU quantizer that
 * exists so the compressed build can be exercised end to end, and it is expected to be replaced
 * by a GPU quantizer. It is shared between the unit tests and the ann-bench CAGRA wrapper so the
 * two can never disagree about the code format; libcuvs itself only ever sees the uploaded codes.
 */
namespace cuvs_internal::bbq {

using cuvs::preprocessing::quantize::bbq::bbq_code_layout;

using host_storage = cuvs::neighbors::host_bbq_dataset<int64_t>::owning_storage_type;

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

inline row_result scalar_quantize(std::vector<float>& vector,
                                  std::vector<uint8_t>& destination,
                                  uint8_t bits,
                                  const float* centroid,
                                  bool euclidean)
{
  const int n        = static_cast<int>(vector.size());
  const int points   = 1 << bits;
  double vec_mean    = 0.0;
  double vec_var     = 0.0;
  float norm2        = 0.0f;
  float centroid_dot = 0.0f;
  float min          = FLT_MAX;
  float max          = -FLT_MAX;
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
    case bbq_code_layout::packed_1b: return (dim * bits + 7) / 8;
    case bbq_code_layout::transposed_2b: return bits * ((dim + 7) / 8);
    case bbq_code_layout::packed_2b: return (dim + 3) / 4;
    case bbq_code_layout::packed_4b: return (dim + 1) / 2;
    case bbq_code_layout::packed_7b: return dim;
    case bbq_code_layout::packed_8b: return dim;
    case bbq_code_layout::transposed_4b: return 4 * ((dim + 7) / 8);
  }
  return 0;
}

// Packs one-byte-per-component codes into packed_1b / packed_2b / transposed_2b /
// packed_4b / transposed_4b (or leaves unpacked). Matches Lucene packAsBinary,
// packNibbles, transposeDibit, transposeHalfByte.
inline std::vector<uint8_t> pack_codes(const std::vector<uint8_t>& unpacked,
                                       size_t n_rows,
                                       size_t dim,
                                       uint32_t bits,
                                       bbq_code_layout layout)
{
  const size_t row_length = encoded_row_length(dim, bits, layout);
  if (layout == bbq_code_layout::packed_8b || layout == bbq_code_layout::packed_7b) {
    return unpacked;
  }

  std::vector<uint8_t> packed(n_rows * row_length, 0);
#pragma omp parallel for
  for (int64_t row = 0; row < static_cast<int64_t>(n_rows); ++row) {
    auto* output      = packed.data() + static_cast<size_t>(row) * row_length;
    const auto* input = unpacked.data() + static_cast<size_t>(row) * dim;
    if (layout == bbq_code_layout::packed_4b) {
      // Lucene OffHeapScalarQuantizedVectorValues.packNibbles
      const size_t half = dim / 2;
      for (size_t i = 0; i < half; ++i) {
        output[i] = static_cast<uint8_t>((input[i] << 4) | (input[half + i] & 0x0f));
      }
      continue;
    }
    if (layout == bbq_code_layout::packed_2b) {
      // Dense 2-bit: four consecutive dimensions per byte, most significant
      // first. Not a bit-plane layout, so it must not fall through below.
      const size_t quads = dim / 4;
      for (size_t i = 0; i < quads; ++i) {
        output[i] = static_cast<uint8_t>((input[4 * i] << 6) | (input[4 * i + 1] << 4) |
                                         (input[4 * i + 2] << 2) | input[4 * i + 3]);
      }
      continue;
    }
    for (size_t d = 0; d < dim; ++d) {
      const uint8_t code = input[d];
      if (layout == bbq_code_layout::packed_1b) {
        for (uint32_t bit = 0; bit < bits; ++bit) {
          const size_t position = d * bits + bit;
          output[position / 8] |=
            static_cast<uint8_t>(((code >> (bits - 1 - bit)) & 1u) << (7 - position % 8));
        }
      } else {
        // transposed_2b / transposed_4b bit-planes (LSB plane first)
        const size_t stripe = (dim + 7) / 8;
        for (uint32_t bit = 0; bit < bits; ++bit) {
          output[bit * stripe + d / 8] |= static_cast<uint8_t>(((code >> bit) & 1u) << (7 - d % 8));
        }
      }
    }
  }
  return packed;
}

inline host_storage quantize(const float* data,
                             int64_t n_rows,
                             int64_t dim,
                             uint8_t bits,
                             cuvs::distance::DistanceType metric,
                             bbq_code_layout layout = bbq_code_layout::packed_8b)
{
  const bool euclidean = metric == cuvs::distance::DistanceType::L2Expanded ||
                         metric == cuvs::distance::DistanceType::L2SqrtExpanded;

  auto centroid          = raft::make_host_vector<float, int64_t>(dim);
  float centroid_norm_sq = 0.0f;
  std::fill_n(centroid.data_handle(), static_cast<size_t>(dim), 0.0f);
  for (int64_t i = 0; i < n_rows; ++i) {
    for (int64_t d = 0; d < dim; ++d) {
      centroid(d) += data[i * dim + d];
    }
  }
  for (int64_t d = 0; d < dim; ++d) {
    centroid(d) /= static_cast<float>(n_rows);
    centroid_norm_sq += centroid(d) * centroid(d);
  }

  std::vector<uint8_t> unpacked(static_cast<size_t>(n_rows * dim));
  auto lower_intervals          = raft::make_host_vector<float, int64_t>(n_rows);
  auto upper_intervals          = raft::make_host_vector<float, int64_t>(n_rows);
  auto additional_corrections   = raft::make_host_vector<float, int64_t>(n_rows);
  auto quantized_component_sums = raft::make_host_vector<int32_t, int64_t>(n_rows);

#pragma omp parallel for
  for (int64_t i = 0; i < n_rows; ++i) {
    std::vector<float> row(data + i * dim, data + (i + 1) * dim);
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
  auto codes = raft::make_host_matrix<uint8_t, int64_t>(
    n_rows, static_cast<int64_t>(encoded_row_length(dim, bits, layout)));
  std::copy(packed.begin(), packed.end(), codes.data_handle());

  return host_storage{std::move(codes),
                      std::move(lower_intervals),
                      std::move(upper_intervals),
                      std::move(additional_corrections),
                      std::move(quantized_component_sums),
                      std::move(centroid),
                      static_cast<uint32_t>(bits),
                      layout,
                      metric,
                      centroid_norm_sq};
}

/** Overload for callers that already hold the rows in a vector. */
inline host_storage quantize(const std::vector<float>& data,
                             int64_t n_rows,
                             int64_t dim,
                             uint8_t bits,
                             cuvs::distance::DistanceType metric,
                             bbq_code_layout layout = bbq_code_layout::packed_8b)
{
  return quantize(data.data(), n_rows, dim, bits, metric, layout);
}

template <typename IdxT>
auto copy_bbq_owning_storage_host_to_device(
  raft::resources const& res,
  typename cuvs::neighbors::host_bbq_dataset<IdxT>::owning_storage_type const& host_storage) ->
  typename cuvs::neighbors::device_bbq_dataset<IdxT>::owning_storage_type
{
  auto stream = raft::resource::get_cuda_stream(res);
  auto codes  = raft::make_device_matrix<uint8_t, IdxT>(
    res, host_storage.codes.extent(0), host_storage.codes.extent(1));
  auto lower_intervals =
    raft::make_device_vector<float, IdxT>(res, host_storage.lower_intervals.extent(0));
  auto upper_intervals =
    raft::make_device_vector<float, IdxT>(res, host_storage.upper_intervals.extent(0));
  auto additional_corrections =
    raft::make_device_vector<float, IdxT>(res, host_storage.additional_corrections.extent(0));
  auto quantized_component_sums =
    raft::make_device_vector<int32_t, IdxT>(res, host_storage.quantized_component_sums.extent(0));
  auto centroid = raft::make_device_vector<float, IdxT>(res, host_storage.centroid.extent(0));

  raft::copy(codes.data_handle(), host_storage.codes.data_handle(), codes.size(), stream);
  raft::copy(lower_intervals.data_handle(),
             host_storage.lower_intervals.data_handle(),
             lower_intervals.size(),
             stream);
  raft::copy(upper_intervals.data_handle(),
             host_storage.upper_intervals.data_handle(),
             upper_intervals.size(),
             stream);
  raft::copy(additional_corrections.data_handle(),
             host_storage.additional_corrections.data_handle(),
             additional_corrections.size(),
             stream);
  raft::copy(quantized_component_sums.data_handle(),
             host_storage.quantized_component_sums.data_handle(),
             quantized_component_sums.size(),
             stream);
  raft::copy(centroid.data_handle(), host_storage.centroid.data_handle(), centroid.size(), stream);

  return {std::move(codes),
          std::move(lower_intervals),
          std::move(upper_intervals),
          std::move(additional_corrections),
          std::move(quantized_component_sums),
          std::move(centroid),
          host_storage.bits,
          host_storage.layout,
          host_storage.metric,
          host_storage.centroid_norm_sq};
}

template <typename IdxT>
auto make_device_bbq_dataset(raft::resources const& res,
                             cuvs::neighbors::host_bbq_dataset<IdxT> const& host)
  -> cuvs::neighbors::device_bbq_dataset<IdxT>
{
  RAFT_EXPECTS(host.quantizers.size() != 0, "host BBQ dataset has no storage");
  cuvs::neighbors::device_bbq_dataset<IdxT> device{
    copy_bbq_owning_storage_host_to_device<IdxT>(res, host.quantizers[0])};
  for (std::size_t i = 1; i < host.quantizers.size(); ++i) {
    device.add_quantizer(copy_bbq_owning_storage_host_to_device<IdxT>(res, host.quantizers[i]));
  }
  return device;
}

/** The one packing each supported code width is stored in. */
inline auto layout_for_bits(uint32_t bits) -> bbq_code_layout
{
  switch (bits) {
    case 1: return bbq_code_layout::packed_1b;
    case 2: return bbq_code_layout::transposed_2b;
    case 4: return bbq_code_layout::transposed_4b;
    case 7: return bbq_code_layout::packed_7b;
    case 8: return bbq_code_layout::packed_8b;
    default: RAFT_FAIL("BBQ bits must be one of 1, 2, 4, 7 or 8; got %u.", bits);
  }
}

/**
 * Reject pairs the kernels cannot serve. Without this the library silently falls back to a
 * symmetric join on the first quantizer, which looks like a working asymmetric run.
 */
inline void validate_bits(uint32_t query_bits, uint32_t doc_bits)
{
  (void)layout_for_bits(query_bits);
  (void)layout_for_bits(doc_bits);
  if (query_bits == doc_bits) { return; }
  const bool supported = (query_bits == 2 && doc_bits == 1) || (query_bits == 4 && doc_bits == 1) ||
                         (query_bits == 4 && doc_bits == 2);
  RAFT_EXPECTS(supported,
               "Asymmetric BBQ NN-Descent supports only (query_bits, doc_bits) of (2, 1), (4, 1) "
               "or (4, 2); got (%u, %u).",
               query_bits,
               doc_bits);
}

/** Every parameter that affects the codes is in the name, so the cache self-invalidates. */
inline auto cache_path(int64_t n_rows,
                       int64_t dim,
                       uint32_t bits,
                       bbq_code_layout layout,
                       cuvs::distance::DistanceType metric) -> std::string
{
  const char* dir = std::getenv("CUVS_BBQ_CACHE_DIR");
  return std::string{dir != nullptr && dir[0] != '\0' ? dir : "/tmp"} + "/bbq-n" +
         std::to_string(n_rows) + "-d" + std::to_string(dim) + "-b" + std::to_string(bits) + "-l" +
         std::to_string(static_cast<int>(layout)) + "-m" +
         std::to_string(static_cast<int>(metric)) + ".bin";
}

/** Visits every raw buffer of @p q in a fixed order; this is the on-disk layout. */
template <typename OpT>
void for_each_buffer(host_storage& q, OpT op)
{
  op(q.codes.data_handle(), q.codes.size() * sizeof(uint8_t));
  op(q.lower_intervals.data_handle(), q.lower_intervals.size() * sizeof(float));
  op(q.upper_intervals.data_handle(), q.upper_intervals.size() * sizeof(float));
  op(q.additional_corrections.data_handle(), q.additional_corrections.size() * sizeof(float));
  op(q.quantized_component_sums.data_handle(), q.quantized_component_sums.size() * sizeof(int32_t));
  op(q.centroid.data_handle(), q.centroid.size() * sizeof(float));
}

/** Allocates the arrays of the given shape, leaving their contents undefined. */
inline auto make_host_storage(int64_t n_rows,
                              int64_t dim,
                              uint32_t bits,
                              bbq_code_layout layout,
                              cuvs::distance::DistanceType metric) -> host_storage
{
  return host_storage{raft::make_host_matrix<uint8_t, int64_t>(
                        n_rows, static_cast<int64_t>(encoded_row_length(dim, bits, layout))),
                      raft::make_host_vector<float, int64_t>(n_rows),
                      raft::make_host_vector<float, int64_t>(n_rows),
                      raft::make_host_vector<float, int64_t>(n_rows),
                      raft::make_host_vector<int32_t, int64_t>(n_rows),
                      raft::make_host_vector<float, int64_t>(dim),
                      bits,
                      layout,
                      metric,
                      0.0f};
}

/** Reads the codes back if the file is present and has exactly the expected length. */
inline auto cache_load(const std::string& path,
                       int64_t n_rows,
                       int64_t dim,
                       uint32_t bits,
                       bbq_code_layout layout,
                       cuvs::distance::DistanceType metric) -> std::optional<host_storage>
{
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) { return std::nullopt; }
  auto q               = make_host_storage(n_rows, dim, bits, layout, metric);
  std::streamoff bytes = 0;
  for_each_buffer(q, [&bytes](void*, size_t n) { bytes += static_cast<std::streamoff>(n); });
  if (f.tellg() != bytes) {
    RAFT_LOG_WARN("Ignoring BBQ cache of unexpected size: %s", path.c_str());
    return std::nullopt;
  }
  f.seekg(0);
  for_each_buffer(
    q, [&f](void* p, size_t n) { f.read(static_cast<char*>(p), static_cast<std::streamsize>(n)); });
  if (!f) {
    RAFT_LOG_WARN("Failed to read BBQ cache, re-quantizing: %s", path.c_str());
    return std::nullopt;
  }
  // Cheaper to recompute than to store and validate.
  for (int64_t d = 0; d < dim; ++d) {
    q.centroid_norm_sq += q.centroid(d) * q.centroid(d);
  }
  RAFT_LOG_INFO("Loaded BBQ codes from %s", path.c_str());
  return q;
}

/** Writes via a temporary so an interrupted run cannot leave a truncated cache behind. */
inline void cache_store(const std::string& path, host_storage& q)
{
  const std::string tmp = path + ".tmp";
  {
    std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
    if (!f) {
      RAFT_LOG_WARN("Cannot write BBQ cache: %s", tmp.c_str());
      return;
    }
    for_each_buffer(q, [&f](void* p, size_t n) {
      f.write(static_cast<const char*>(p), static_cast<std::streamsize>(n));
    });
    if (!f) {
      RAFT_LOG_WARN("Failed to write BBQ cache: %s", tmp.c_str());
      f.close();
      std::remove(tmp.c_str());
      return;
    }
  }
  if (std::rename(tmp.c_str(), path.c_str()) != 0) {
    RAFT_LOG_WARN("Cannot move BBQ cache into place: %s", path.c_str());
    std::remove(tmp.c_str());
    return;
  }
  RAFT_LOG_INFO("Wrote BBQ codes to %s", path.c_str());
}

inline auto quantize_cached(const float* rows,
                            int64_t n_rows,
                            int64_t dim,
                            uint32_t bits,
                            cuvs::distance::DistanceType metric) -> host_storage
{
  const auto layout = layout_for_bits(bits);
  const auto path   = cache_path(n_rows, dim, bits, layout, metric);
  if (auto cached = cache_load(path, n_rows, dim, bits, layout, metric)) {
    return std::move(*cached);
  }
  auto quantized = quantize(rows, n_rows, dim, static_cast<uint8_t>(bits), metric, layout);
  cache_store(path, quantized);
  return quantized;
}

/**
 * Quantize host-resident @p rows and upload the codes, ready for the BBQ NN-Descent build.
 *
 * `query_bits == doc_bits` produces a single quantizer and a symmetric join; differing widths
 * produce two quantizers and an asymmetric join, where the wider codes serve the query side.
 */
inline auto quantize_to_device(raft::resources const& res,
                               const float* rows,
                               int64_t n_rows,
                               int64_t dim,
                               cuvs::distance::DistanceType metric,
                               uint32_t query_bits,
                               uint32_t doc_bits) -> cuvs::neighbors::device_bbq_dataset<int64_t>
{
  validate_bits(query_bits, doc_bits);
  cuvs::neighbors::host_bbq_dataset<int64_t> host{
    quantize_cached(rows, n_rows, dim, query_bits, metric)};
  if (doc_bits != query_bits) {
    host.add_quantizer(quantize_cached(rows, n_rows, dim, doc_bits, metric));
  }
  auto device = make_device_bbq_dataset<int64_t>(res, host);
  // The uploads are stream-ordered against `host`, which dies with this frame.
  raft::resource::sync_stream(res);
  return device;
}

}  // namespace cuvs_internal::bbq
