/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/core/export.hpp>
#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/common.hpp>
#include <cuvs/preprocessing/quantize/bbq.hpp>

#include <raft/core/device_mdspan.hpp>

#include <cassert>
#include <cstddef>
#include <cstdint>

namespace CUVS_EXPORT cuvs {
namespace preprocessing::quantize::bbq {

#ifdef __CUDACC__

// --------------------------------------------------------------------------
// Layout geometry
// Encoded row length implied by a code layout, bit width and dimensionality.
// --------------------------------------------------------------------------

_RAFT_HOST_DEVICE constexpr uint32_t get_encoded_row_length(const bbq_code_layout layout,
                                                            const uint32_t bits,
                                                            const uint32_t dim)
{
  switch (layout) {
    case bbq_code_layout::packed_1b: return (dim * bits + 7) / 8;
    case bbq_code_layout::transposed_2b: return bits * ((dim + 7) / 8);
    case bbq_code_layout::packed_4b: return (dim + 1) / 2;
    case bbq_code_layout::packed_2b: return (dim + 3) / 4;
    case bbq_code_layout::transposed_4b: return 4 * ((dim + 7) / 8);
    case bbq_code_layout::packed_7b: return dim;
    case bbq_code_layout::packed_8b: return dim;
  }
  return 0;
}

template <typename DataT, typename IdxT, typename Accessor>
_RAFT_HOST_DEVICE constexpr uint32_t get_encoded_row_length(
  const bbq_quantizer_view<DataT, IdxT, Accessor>& dataset)
{
  return get_encoded_row_length(dataset.layout, dataset.bits, dataset.dim());
}

// --------------------------------------------------------------------------
// Code inner products (1x1)
// Raw byte-level dot products over one row pair, per code layout.
// --------------------------------------------------------------------------

/**
 * Cross-plane binary inner product over `Planes` bit planes, shifting each (i, j) plane pair by
 * i + j. `Planes == 1` is a plain binary product, so this covers packed_1b as well as the
 * transposed layouts. The 4-byte body needs both rows 4-byte aligned; the byte tail handles a
 * stripe whose length is not a multiple of 4.
 */
template <int Planes>
__device__ __forceinline__ uint32_t code_inner_product_transposed(const uint8_t* row_a,
                                                                  const uint8_t* row_b,
                                                                  size_t n_bytes,
                                                                  uint32_t result = 0)
{
  const size_t stripe = n_bytes / Planes;
  // Plane p starts at row + p * stripe and is read as uint32_t words, so a stripe that is not a
  // multiple of 4 misaligns every plane past the first -- the byte tail below only covers a ragged
  // stripe *length*, not a ragged stripe *offset*. Planes == 1 has no offset and so is exempt.
  assert(Planes == 1 || stripe % sizeof(uint32_t) == 0);
#pragma unroll
  for (int i = 0; i < Planes; ++i) {
#pragma unroll
    for (int j = 0; j < Planes; ++j) {
      const uint8_t* a = row_a + i * stripe;
      const uint8_t* b = row_b + j * stripe;
      uint32_t partial = 0;
      size_t k         = 0;
#pragma unroll 4
      for (; k + sizeof(uint32_t) <= stripe; k += sizeof(uint32_t)) {
        partial += __popc(*reinterpret_cast<const uint32_t*>(a + k) &
                          *reinterpret_cast<const uint32_t*>(b + k));
      }
      for (; k < stripe; ++k) {
        partial += __popc(static_cast<unsigned>(a[k] & b[k]));
      }
      result += partial << (i + j);
    }
  }
  return result;
}

/** Symmetric for packNibbles (Lucene int4DotProductBothPacked). */
__device__ __forceinline__ uint32_t code_inner_product_packed_4b(const uint8_t* row_a,
                                                                 const uint8_t* row_b,
                                                                 size_t n_bytes,
                                                                 uint32_t total = 0)
{
  constexpr uint32_t nibble_mask = 0x0F0F0F0Fu;
  size_t i                       = 0;
#pragma unroll 4
  for (; i + 4 <= n_bytes; i += 4) {
    const auto a = *reinterpret_cast<const uint32_t*>(row_a + i);
    const auto b = *reinterpret_cast<const uint32_t*>(row_b + i);
    total        = __dp4a(a & nibble_mask, b & nibble_mask, total);
    total        = __dp4a((a >> 4) & nibble_mask, (b >> 4) & nibble_mask, total);
  }
  for (; i < n_bytes; ++i) {
    const unsigned a = row_a[i];
    const unsigned b = row_b[i];
    total += (a & 0x0Fu) * (b & 0x0Fu);
    total += ((a >> 4) & 0x0Fu) * ((b >> 4) & 0x0Fu);
  }
  return total;
}

/** One-byte-per-code dot product, optionally masking unused high bits. */
__device__ __forceinline__ uint32_t code_inner_product_packed_8b(const uint8_t* row_a,
                                                                 const uint8_t* row_b,
                                                                 size_t n_bytes,
                                                                 uint32_t result   = 0,
                                                                 uint8_t code_mask = 0xFFu)
{
  const uint32_t word_mask = uint32_t{code_mask} * 0x01010101u;
  size_t i       = 0;
#pragma unroll 4
  for (; i + 4 <= n_bytes; i += 4) {
    const auto a = *reinterpret_cast<const uint32_t*>(row_a + i) & word_mask;
    const auto b = *reinterpret_cast<const uint32_t*>(row_b + i) & word_mask;
    result       = __dp4a(a, b, result);
  }
  for (; i < n_bytes; ++i) {
    result +=
      static_cast<uint32_t>(row_a[i] & code_mask) * static_cast<uint32_t>(row_b[i] & code_mask);
  }
  return result;
}

/**
 * Integer inner product between two encoded rows.
 *
 * The uint32_t result bounds every BBQ layout to 66,050 dimensions: the worst case is
 * `packed_8b`, where `dim * 255 * 255` must not exceed UINT32_MAX.
 */
__device__ __forceinline__ uint32_t code_inner_product(const uint8_t* row_a,
                                                       const uint8_t* row_b,
                                                       const bbq_code_layout layout,
                                                       const uint32_t bits,
                                                       const size_t n_bytes,
                                                       uint32_t result = 0)
{
  switch (layout) {
    case bbq_code_layout::packed_1b:
      return code_inner_product_transposed<1>(row_a, row_b, n_bytes, result);
    case bbq_code_layout::transposed_2b:
      return code_inner_product_transposed<2>(row_a, row_b, n_bytes, result);
    case bbq_code_layout::packed_4b:
      return code_inner_product_packed_4b(row_a, row_b, n_bytes, result);
    case bbq_code_layout::transposed_4b:
      return code_inner_product_transposed<4>(row_a, row_b, n_bytes, result);
    case bbq_code_layout::packed_8b:
      return code_inner_product_packed_8b(row_a, row_b, n_bytes, result);
    case bbq_code_layout::packed_7b:
    default:
      return code_inner_product_packed_8b(
        row_a, row_b, n_bytes, result, static_cast<uint8_t>((uint32_t{1} << bits) - 1));
  }
}

template <typename DataT, typename IdxT, typename Accessor>
__device__ __forceinline__ uint32_t
code_inner_product(const uint8_t* row_a,
                   const uint8_t* row_b,
                   const bbq_quantizer_view<DataT, IdxT, Accessor>& dataset)
{
  return code_inner_product(
    row_a, row_b, dataset.layout, dataset.bits, get_encoded_row_length(dataset));
}

// --------------------------------------------------------------------------
// Metrics: (document x query)
// Two quantizer views; the row lives in the document view, the column in the query view. A
// self-join (symmetric case) is just this with dataset_document == dataset_query -- passing the
// same view twice reproduces the old single-dataset formulas bit-for-bit, so there is no separate
// symmetric code path.
// --------------------------------------------------------------------------

template <typename DataT, typename IdxT, typename Accessor>
__device__ __forceinline__ float centered_dot(
  const bbq_quantizer_view<DataT, IdxT, Accessor>& dataset_document,
  const bbq_quantizer_view<DataT, IdxT, Accessor>& dataset_query,
  float code_ip,
  int64_t row_document,
  int64_t row_query)
{
  const float lower_doc     = dataset_document.lower_intervals(row_document);
  const float lower_q       = dataset_query.lower_intervals(row_query);
  const float delta_doc     = dataset_document.dequant_delta(row_document);
  const float delta_q       = dataset_query.dequant_delta(row_query);
  const float sum_delta_doc = dataset_document.dequant_sum_delta(row_document);
  const float sum_delta_q   = dataset_query.dequant_sum_delta(row_query);

  auto dim = static_cast<float>(dataset_document.dim());
  return dim * lower_doc * lower_q + lower_q * sum_delta_doc + lower_doc * sum_delta_q +
         delta_doc * delta_q * code_ip;
}

// Dot product overload when the centered dot product is already computed
template <typename DataT, typename IdxT, typename Accessor>
__device__ __forceinline__ float dot_product(
  const bbq_quantizer_view<DataT, IdxT, Accessor>& dataset_document,
  const bbq_quantizer_view<DataT, IdxT, Accessor>& dataset_query,
  float centered_dot_value,
  int64_t row_document,
  int64_t row_query)
{
  return centered_dot_value + dataset_document.additional_corrections(row_document) +
         dataset_query.additional_corrections(row_query) - dataset_document.centroid_norm_sq;
}

/** Squared L2 distance overload when the centered dot product is already computed */
template <typename DataT, typename IdxT, typename Accessor>
__device__ __forceinline__ float l2_distance(
  const bbq_quantizer_view<DataT, IdxT, Accessor>& dataset_doc,
  const bbq_quantizer_view<DataT, IdxT, Accessor>& dataset_query,
  float centered_dot_value,
  int64_t row_document,
  int64_t row_query)
{
  const float distance = dataset_doc.additional_corrections(row_document) +
                         dataset_query.additional_corrections(row_query) -
                         2.0f * centered_dot_value;
  return distance < 0.0f ? 0.0f : distance;
}

/** Squared norm of one original-space row -- a self-join (row against itself, same dataset). */
template <typename DataT, typename IdxT, typename Accessor>
__device__ __forceinline__ float row_norm(const bbq_quantizer_view<DataT, IdxT, Accessor>& dataset,
                                          int64_t row)
{
  const uint8_t* codes_row = dataset.codes.data_handle() + row * get_encoded_row_length(dataset);
  const float code_ip      = static_cast<float>(code_inner_product(codes_row, codes_row, dataset));
  const float centered     = centered_dot(dataset, dataset, code_ip, row, row);
  const float norm         = dot_product(dataset, dataset, centered, row, row);
  return norm < 0.0f ? 0.0f : norm;
}

template <typename DataT, typename IdxT, typename Accessor>
struct bbq_row_norm_op {
  const bbq_quantizer_view<DataT, IdxT, Accessor> quantizer;

  __device__ auto operator()(size_t row) const -> float
  {
    return row_norm(quantizer, static_cast<int64_t>(row));
  }
};

// --------------------------------------------------------------------------
// Fused inner products (2x1)
// Two left rows against a shared right operand, for the local-join inner loop.
// --------------------------------------------------------------------------

/**
 * Two cross-plane inner products over statically known document and query plane counts.
 * Not asymmetric-specific: a symmetric pair is just document_planes == query_planes, which is how
 * the 1x1 and 2t x 2t self-joins are computed.
 */
template <int document_planes, int query_planes, size_t document_row_bytes, size_t query_row_bytes>
__device__ inline void code_inner_product_planes_2x1(const uint8_t* row_a0,
                                                     const uint8_t* row_a1,
                                                     const uint8_t* row_b,
                                                     uint32_t& total0,
                                                     uint32_t& total1)
{
  constexpr size_t document_plane_stride = document_row_bytes / document_planes;
  constexpr size_t query_plane_stride    = query_row_bytes / query_planes;
  // Both operands are stepped by their own plane stride and then read as uint32_t, so both
  // strides -- not just the query's -- must be 4-byte aligned.
  static_assert(query_plane_stride % sizeof(uint32_t) == 0);
  static_assert(document_plane_stride % sizeof(uint32_t) == 0);
#pragma unroll
  for (int p_query = 0; p_query < query_planes; ++p_query) {
#pragma unroll
    for (int p_document = 0; p_document < document_planes; ++p_document) {
      const uint8_t* a0 = row_a0 + p_document * document_plane_stride;
      const uint8_t* a1 = row_a1 + p_document * document_plane_stride;
      const uint8_t* b  = row_b + p_query * query_plane_stride;
      uint32_t partial0 = 0;
      uint32_t partial1 = 0;
#pragma unroll 4
      for (size_t i = 0; i < query_plane_stride; i += sizeof(uint32_t)) {
        const auto wa0 = *reinterpret_cast<const uint32_t*>(a0 + i);
        const auto wa1 = *reinterpret_cast<const uint32_t*>(a1 + i);
        const auto wb  = *reinterpret_cast<const uint32_t*>(b + i);
        partial0 += __popc(wa0 & wb);
        partial1 += __popc(wa1 & wb);
      }
      total0 += partial0 << (p_document + p_query);
      total1 += partial1 << (p_document + p_query);
    }
  }
}

template <size_t n_bytes>
__device__ inline void code_inner_product_packed_4b_symmetric_2x1(const uint8_t* row_a0,
                                                                  const uint8_t* row_a1,
                                                                  const uint8_t* row_b,
                                                                  uint32_t& total0,
                                                                  uint32_t& total1)
{
  static_assert(n_bytes % sizeof(uint32_t) == 0);
  constexpr uint32_t nibble_mask = 0x0F0F0F0Fu;
#pragma unroll 4
  for (size_t i = 0; i < n_bytes; i += sizeof(uint32_t)) {
    const auto a0     = *reinterpret_cast<const uint32_t*>(row_a0 + i);
    const auto a1     = *reinterpret_cast<const uint32_t*>(row_a1 + i);
    const auto b      = *reinterpret_cast<const uint32_t*>(row_b + i);
    const auto b_low  = b & nibble_mask;
    const auto b_high = (b >> 4) & nibble_mask;
    total0            = __dp4a(a0 & nibble_mask, b_low, total0);
    total0            = __dp4a((a0 >> 4) & nibble_mask, b_high, total0);
    total1            = __dp4a(a1 & nibble_mask, b_low, total1);
    total1            = __dp4a((a1 >> 4) & nibble_mask, b_high, total1);
  }
}

template <size_t n_bytes>
__device__ inline void code_inner_product_packed_8b_2x1(const uint8_t* row_a0,
                                                        const uint8_t* row_a1,
                                                        const uint8_t* row_b,
                                                        uint32_t& total0,
                                                        uint32_t& total1,
                                                        uint8_t code_mask = 0xFFu)
{
  static_assert(n_bytes % sizeof(uint32_t) == 0);
  const uint32_t word_mask = uint32_t{code_mask} * 0x01010101u;
#pragma unroll 4
  for (size_t i = 0; i < n_bytes; i += sizeof(uint32_t)) {
    const auto a0 = *reinterpret_cast<const uint32_t*>(row_a0 + i) & word_mask;
    const auto a1 = *reinterpret_cast<const uint32_t*>(row_a1 + i) & word_mask;
    const auto b  = *reinterpret_cast<const uint32_t*>(row_b + i) & word_mask;
    total0        = __dp4a(a0, b, total0);
    total1        = __dp4a(a1, b, total1);
  }
}

// Selects the SIMT inner product for a (document, query) layout pair: bit-sliced layouts go to the
// cross-plane popc, densely-packed ones to dp4a. The dp4a forms are self-join only, since dp4a
// needs both operands in the same packing. packed_7b and packed_8b reach this from
// GNND::local_join; packed_4b does not (symmetric packed_4b goes to the wmma kernel), but is kept
// as a SIMT reference point.
template <bbq_code_layout DocumentLayout,
          bbq_code_layout QueryLayout,
          bool SelfJoin,
          int DocumentPlanes,
          int QueryPlanes,
          size_t DocumentRowBytes,
          size_t QueryRowBytes>
__device__ __forceinline__ void bbq_code_inner_product_2x1(const uint8_t* row_a0,
                                                           const uint8_t* row_a1,
                                                           const uint8_t* row_b,
                                                           uint32_t& total0,
                                                           uint32_t& total1)
{
  namespace bbq = cuvs::preprocessing::quantize::bbq;
  if constexpr (SelfJoin && DocumentLayout == bbq_code_layout::packed_4b) {
    bbq::code_inner_product_packed_4b_symmetric_2x1<DocumentRowBytes>(
      row_a0, row_a1, row_b, total0, total1);
  } else if constexpr (SelfJoin && (DocumentLayout == bbq_code_layout::packed_8b ||
                                    DocumentLayout == bbq_code_layout::packed_7b)) {
    // packed_7b is packed_8b with the top bit masked off, matching code_inner_product's
    // (1 << bits) - 1 mask for the same two layouts.
    constexpr uint8_t code_mask = DocumentLayout == bbq_code_layout::packed_7b ? 0x7Fu : 0xFFu;
    bbq::code_inner_product_packed_8b_2x1<DocumentRowBytes>(
      row_a0, row_a1, row_b, total0, total1, code_mask);
  } else {
    bbq::
      code_inner_product_planes_2x1<DocumentPlanes, QueryPlanes, DocumentRowBytes, QueryRowBytes>(
        row_a0, row_a1, row_b, total0, total1);
  }
}

#endif  // __CUDACC__

}  // namespace preprocessing::quantize::bbq
}  // namespace CUVS_EXPORT cuvs
