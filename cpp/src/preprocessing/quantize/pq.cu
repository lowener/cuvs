/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "./detail/pq.cuh"

#include <cuvs/preprocessing/quantize/pq.hpp>

namespace cuvs::preprocessing::quantize::pq {

#define CUVS_INST_QUANTIZATION(T, QuantI)                                         \
  auto train(raft::resources const& res,                                          \
             const params params,                                                 \
             raft::device_matrix_view<const T, int64_t> dataset) -> quantizer<T>  \
  {                                                                               \
    return detail::train<T, T>(res, params, dataset);                             \
  }                                                                               \
  auto train(raft::resources const& res,                                          \
             const params params,                                                 \
             raft::host_matrix_view<const T, int64_t> dataset) -> quantizer<T>    \
  {                                                                               \
    return detail::train<T, T>(res, params, dataset);                             \
  }                                                                               \
  void transform(raft::resources const& res,                                      \
                 const quantizer<T>& quantizer,                                   \
                 raft::device_matrix_view<const T, int64_t> dataset,              \
                 raft::device_matrix_view<QuantI, int64_t> out)                   \
  {                                                                               \
    detail::transform(res, quantizer, dataset, out);                              \
  }                                                                               \
  void transform(raft::resources const& res,                                      \
                 const quantizer<T>& quantizer,                                   \
                 raft::host_matrix_view<const T, int64_t> dataset,                \
                 raft::device_matrix_view<QuantI, int64_t> out)                   \
  {                                                                               \
    detail::transform(res, quantizer, dataset, out);                              \
  }                                                                               \
  void inverse_transform(raft::resources const& res,                              \
                         const quantizer<T>& quantizer,                           \
                         raft::device_matrix_view<const QuantI, int64_t> dataset, \
                         raft::device_matrix_view<T, int64_t> out)                \
  {                                                                               \
    detail::inverse_transform(res, quantizer, dataset, out);                      \
  }

CUVS_INST_QUANTIZATION(float, uint8_t);

#undef CUVS_INST_QUANTIZATION

}  // namespace cuvs::preprocessing::quantize::pq

namespace cuvs::preprocessing::quantize::pq {

#define CUVS_INST_VPQ_BUILD(T)                                                               \
  auto vpq_build(const raft::resources& res,                                                 \
                 const cuvs::neighbors::vpq_params& params,                                  \
                 const raft::host_matrix_view<const T, int64_t, raft::row_major>& dataset)   \
  {                                                                                          \
    return cuvs::neighbors::detail::vpq_convert_math_type<half, float, int64_t>(             \
      res,                                                                                   \
      cuvs::neighbors::detail::vpq_build<decltype(dataset), float, int64_t>(                 \
        res, params, dataset));                                                              \
  }                                                                                          \
  auto vpq_build(const raft::resources& res,                                                 \
                 const cuvs::neighbors::vpq_params& params,                                  \
                 const raft::device_matrix_view<const T, int64_t, raft::row_major>& dataset) \
  {                                                                                          \
    return cuvs::neighbors::detail::vpq_convert_math_type<half, float, int64_t>(             \
      res,                                                                                   \
      cuvs::neighbors::detail::vpq_build<decltype(dataset), float, int64_t>(                 \
        res, params, dataset));                                                              \
  }

CUVS_INST_VPQ_BUILD(float);
CUVS_INST_VPQ_BUILD(half);
CUVS_INST_VPQ_BUILD(int8_t);
CUVS_INST_VPQ_BUILD(uint8_t);

#undef CUVS_INST_VPQ_BUILD
}  // namespace cuvs::preprocessing::quantize::pq
