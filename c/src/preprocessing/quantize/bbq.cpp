/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/core/dataset.h>
#include <cuvs/preprocessing/quantize/bbq.h>
#include <cuvs/preprocessing/quantize/bbq.hpp>

#include "../../core/exceptions.hpp"
#include "../../core/interop.hpp"
#include "bbq_helpers.hpp"

#include <raft/core/error.hpp>
#include <raft/core/mdspan_types.hpp>

#include <cstdint>
#include <limits>
#include <memory>

namespace {

namespace bbq = cuvs::preprocessing::quantize::bbq;

/** State behind an owning `cuvsBbqQuantizer`, i.e. one built by `cuvsBbqQuantizerBuild`. */
template <typename T>
struct built_quantizer {
  bbq::quantizer<T> quantizer;
  bbq::params params;
};

template <typename T>
void destroy_typed_addr(void* ptr)
{
  delete reinterpret_cast<T*>(ptr);
}

auto to_cpp_params(cuvsBbqQuantizerParams const& params) -> bbq::params
{
  constexpr auto caller = "cuvsBbqQuantizerBuild";
  bbq::params out;
  out.doc_layout   = bbq::c_api::to_cpp_layout(params.doc_layout, caller);
  out.query_layout = bbq::c_api::to_cpp_layout(params.query_layout, caller);
  out.metric       = static_cast<cuvs::distance::DistanceType>(params.metric);
  RAFT_EXPECTS(params.lambda >= 0.0f && params.lambda <= 1.0f,
               "%s: lambda must be within [0, 1], got %f",
               caller,
               static_cast<double>(params.lambda));
  RAFT_EXPECTS(params.iters <= static_cast<uint32_t>(std::numeric_limits<int>::max()),
               "%s: iters must not exceed %d, got %u",
               caller,
               std::numeric_limits<int>::max(),
               params.iters);
  out.lambda = params.lambda;
  out.iters  = static_cast<int>(params.iters);

  const bbq::bbq_code_layout layouts[] = {out.doc_layout, out.query_layout};
  bbq::c_api::expect_supported_layout_set(
    layouts, out.doc_layout == out.query_layout ? 1 : 2, caller);
  return out;
}

auto is_float32(DLDataType dtype) -> bool
{
  return dtype.code == kDLFloat && dtype.bits == 32 && dtype.lanes == 1;
}

/** Calls `fn(quantizer_view)` on the centroid of a built or view quantizer. */
template <typename T, typename Fn>
void with_typed_quantizer_view(cuvsBbqQuantizer_t quantizer, Fn&& fn)
{
  if (quantizer->is_owning) {
    fn(reinterpret_cast<built_quantizer<T>*>(quantizer->addr)->quantizer.view());
  } else {
    fn(reinterpret_cast<bbq::bbq_dataset_view_storage<T, int64_t>*>(quantizer->addr)->quantizer);
  }
}

template <typename Fn>
void with_quantizer_view(cuvsBbqQuantizer_t quantizer, char const* caller, Fn&& fn)
{
  RAFT_EXPECTS(quantizer != nullptr, "%s: null quantizer", caller);
  RAFT_EXPECTS(quantizer->addr != 0, "%s: quantizer has not been built", caller);
  auto dtype = quantizer->dtype;
  if (is_float32(dtype)) {
    with_typed_quantizer_view<float>(quantizer, fn);
  } else if (dtype.code == kDLFloat && dtype.bits == 16) {
    with_typed_quantizer_view<half>(quantizer, fn);
  } else if (dtype.code == kDLInt && dtype.bits == 8) {
    with_typed_quantizer_view<int8_t>(quantizer, fn);
  } else if (dtype.code == kDLUInt && dtype.bits == 8) {
    with_typed_quantizer_view<uint8_t>(quantizer, fn);
  } else {
    RAFT_FAIL("%s: unsupported quantizer dtype: code=%d, bits=%d", caller, dtype.code, dtype.bits);
  }
}

auto dataset_matrix_view(DLManagedTensor* dataset, char const* caller)
  -> raft::device_matrix_view<const float, int64_t, raft::row_major>
{
  RAFT_EXPECTS(dataset != nullptr, "%s: null dataset", caller);
  auto tensor = dataset->dl_tensor;
  RAFT_EXPECTS(cuvs::core::is_dlpack_device_compatible(tensor),
               "%s: dataset must be in device-accessible memory",
               caller);
  RAFT_EXPECTS(is_float32(tensor.dtype),
               "%s: unsupported dataset dtype: code=%d, bits=%d; only float32 is supported",
               caller,
               tensor.dtype.code,
               tensor.dtype.bits);
  return cuvs::core::from_dlpack<raft::device_matrix_view<const float, int64_t, raft::row_major>>(
    dataset);
}

template <typename T>
auto make_cpp_quantizer_view(
  DLManagedTensor* codes_tensor,
  DLManagedTensor* lower_intervals_tensor,
  DLManagedTensor* upper_intervals_tensor,
  DLManagedTensor* additional_corrections_tensor,
  DLManagedTensor* quantized_component_sums_tensor,
  DLManagedTensor* centroid_tensor,
  DLManagedTensor* dequant_delta_tensor,
  DLManagedTensor* dequant_sum_delta_tensor,
  DLManagedTensor* row_norm_tensor,
  cuvsBbqCodeLayout_t c_layout,
  cuvsDistanceType metric,
  float centroid_norm_sq)
  -> cuvs::preprocessing::quantize::bbq::bbq_dataset_view_storage<T, int64_t>
{
  using quantizer_view_t = cuvs::preprocessing::quantize::bbq::quantizer_view<T>;
  using storage_view_t =
    cuvs::preprocessing::quantize::bbq::bbq_dataset_view_storage<T, int64_t>;
  using codes_view_t    = raft::device_matrix_view<const uint8_t, int64_t>;
  using float_view_t    = raft::device_vector_view<const float, int64_t>;
  using int_view_t      = raft::device_vector_view<const int32_t, int64_t>;
  using centroid_view_t = raft::device_vector_view<const T, int64_t>;

  auto codes       = cuvs::core::from_dlpack<codes_view_t>(codes_tensor);
  auto lower       = cuvs::core::from_dlpack<float_view_t>(lower_intervals_tensor);
  auto upper       = cuvs::core::from_dlpack<float_view_t>(upper_intervals_tensor);
  auto corrections = cuvs::core::from_dlpack<float_view_t>(additional_corrections_tensor);
  auto sums        = cuvs::core::from_dlpack<int_view_t>(quantized_component_sums_tensor);
  auto centroid    = cuvs::core::from_dlpack<centroid_view_t>(centroid_tensor);
  auto delta       = cuvs::core::from_dlpack<float_view_t>(dequant_delta_tensor);
  auto sum_delta   = cuvs::core::from_dlpack<float_view_t>(dequant_sum_delta_tensor);
  auto row_norm    = cuvs::core::from_dlpack<float_view_t>(row_norm_tensor);
  auto layout      = bbq::c_api::to_cpp_layout(c_layout, "cuvsBbqQuantizerCreateView");

  const auto n_rows = codes.extent(0);
  const auto dim    = static_cast<uint32_t>(centroid.extent(0));
  RAFT_EXPECTS(n_rows > 0 && dim > 0, "cuvsBbqQuantizerCreateView: quantizer must not be empty");
  RAFT_EXPECTS(
    codes.extent(1) ==
      static_cast<int64_t>(
        cuvs::preprocessing::quantize::bbq::get_encoded_row_length(dim, layout)),
    "cuvsBbqQuantizerCreateView: code row length does not match the dimension and layout");
  RAFT_EXPECTS(lower.extent(0) == n_rows && upper.extent(0) == n_rows &&
                 corrections.extent(0) == n_rows && sums.extent(0) == n_rows &&
                 delta.extent(0) == n_rows && sum_delta.extent(0) == n_rows &&
                 row_norm.extent(0) == n_rows,
               "cuvsBbqQuantizerCreateView: every correction vector must contain one value per row");

  return storage_view_t{quantizer_view_t{centroid, centroid_norm_sq},
                        codes,
                        lower,
                        upper,
                        corrections,
                        sums,
                        delta,
                        sum_delta,
                        row_norm,
                        layout,
                        static_cast<cuvs::distance::DistanceType>(metric)};
}

template <typename T>
void make_and_bind_quantizer_view(
  cuvsBbqQuantizer_t* output,
  DLDataType dtype,
  DLManagedTensor* codes,
  DLManagedTensor* lower_intervals,
  DLManagedTensor* upper_intervals,
  DLManagedTensor* additional_corrections,
  DLManagedTensor* quantized_component_sums,
  DLManagedTensor* centroid,
  DLManagedTensor* dequant_delta,
  DLManagedTensor* dequant_sum_delta,
  DLManagedTensor* row_norm,
  cuvsBbqCodeLayout_t layout,
  cuvsDistanceType metric,
  float centroid_norm_sq)
{
  using view_t = cuvs::preprocessing::quantize::bbq::bbq_dataset_view_storage<T, int64_t>;
  auto view    = std::make_unique<view_t>(make_cpp_quantizer_view<T>(codes,
                                                                  lower_intervals,
                                                                  upper_intervals,
                                                                  additional_corrections,
                                                                  quantized_component_sums,
                                                                  centroid,
                                                                  dequant_delta,
                                                                  dequant_sum_delta,
                                                                  row_norm,
                                                                  layout,
                                                                  metric,
                                                                  centroid_norm_sq));
  auto handle          = std::make_unique<cuvsBbqQuantizer>();
  handle->addr         = reinterpret_cast<uintptr_t>(view.release());
  handle->destroy_addr = &destroy_typed_addr<view_t>;
  handle->dtype        = dtype;
  handle->is_owning    = false;
  *output              = handle.release();
}

}  // namespace

extern "C" cuvsError_t cuvsBbqQuantizerParamsCreate(cuvsBbqQuantizerParams_t* params)
{
  return cuvs::core::translate_exceptions([=] {
    RAFT_EXPECTS(params != nullptr, "cuvsBbqQuantizerParamsCreate: null output");
    const bbq::params defaults;
    *params =
      new cuvsBbqQuantizerParams{.doc_layout   = bbq::c_api::to_c_layout(defaults.doc_layout),
                                 .query_layout = bbq::c_api::to_c_layout(defaults.query_layout),
                                 .metric       = static_cast<cuvsDistanceType>(defaults.metric),
                                 .lambda       = defaults.lambda,
                                 .iters        = static_cast<uint32_t>(defaults.iters)};
  });
}

extern "C" cuvsError_t cuvsBbqQuantizerParamsDestroy(cuvsBbqQuantizerParams_t params)
{
  return cuvs::core::translate_exceptions([=] { delete params; });
}

extern "C" cuvsError_t cuvsBbqQuantizerCreate(cuvsBbqQuantizer_t* quantizer)
{
  return cuvs::core::translate_exceptions([=] {
    RAFT_EXPECTS(quantizer != nullptr, "cuvsBbqQuantizerCreate: null output");
    *quantizer = new cuvsBbqQuantizer{0, nullptr, DLDataType{}, true};
  });
}

extern "C" cuvsError_t cuvsBbqQuantizerBuild(cuvsResources_t res,
                                             cuvsBbqQuantizerParams_t params,
                                             DLManagedTensor* dataset,
                                             cuvsBbqQuantizer_t quantizer)
{
  return cuvs::core::translate_exceptions([=] {
    constexpr auto caller = "cuvsBbqQuantizerBuild";
    RAFT_EXPECTS(params != nullptr, "%s: null params", caller);
    RAFT_EXPECTS(quantizer != nullptr, "%s: null quantizer", caller);
    RAFT_EXPECTS(quantizer->is_owning,
                 "%s: quantizer was created with cuvsBbqQuantizerCreateView; create it with "
                 "cuvsBbqQuantizerCreate",
                 caller);
    RAFT_EXPECTS(quantizer->addr == 0, "%s: quantizer has already been built", caller);
    auto cpp_params = to_cpp_params(*params);
    auto view       = dataset_matrix_view(dataset, caller);

    auto* res_ptr  = reinterpret_cast<raft::resources*>(res);
    using holder_t = built_quantizer<float>;
    auto holder = std::unique_ptr<holder_t>(new holder_t{bbq::build(*res_ptr, view), cpp_params});

    quantizer->addr         = reinterpret_cast<uintptr_t>(holder.release());
    quantizer->destroy_addr = &destroy_typed_addr<holder_t>;
    quantizer->dtype        = DLDataType{kDLFloat, 32, 1};
  });
}

extern "C" cuvsError_t cuvsBbqQuantizerTransform(cuvsResources_t res,
                                                 cuvsBbqQuantizer_t quantizer,
                                                 DLManagedTensor* dataset,
                                                 cuvsDataset_t bbq_dataset)
{
  return cuvs::core::translate_exceptions([=] {
    constexpr auto caller = "cuvsBbqQuantizerTransform";
    RAFT_EXPECTS(quantizer != nullptr, "%s: null quantizer", caller);
    RAFT_EXPECTS(quantizer->is_owning,
                 "%s: quantizer was created with cuvsBbqQuantizerCreateView; only a quantizer "
                 "built with cuvsBbqQuantizerBuild can encode",
                 caller);
    RAFT_EXPECTS(quantizer->addr != 0, "%s: quantizer has not been built", caller);
    RAFT_EXPECTS(is_float32(quantizer->dtype),
                 "%s: unsupported quantizer dtype: code=%d, bits=%d",
                 caller,
                 quantizer->dtype.code,
                 quantizer->dtype.bits);
    RAFT_EXPECTS(bbq_dataset != nullptr, "%s: null output dataset", caller);
    RAFT_EXPECTS(bbq_dataset->addr == 0,
                 "%s: output dataset handle must be empty; create it with cuvsDatasetCreate",
                 caller);
    auto view = dataset_matrix_view(dataset, caller);

    auto* res_ptr = reinterpret_cast<raft::resources*>(res);
    auto* holder  = reinterpret_cast<built_quantizer<float>*>(quantizer->addr);
    using owner_t = cuvs::neighbors::device_bbq_dataset<float, int64_t>;
    auto encoded  = std::make_unique<owner_t>(
      bbq::transform(*res_ptr, holder->params, holder->quantizer.view(), view));

    bbq_dataset->addr         = reinterpret_cast<uintptr_t>(encoded.release());
    bbq_dataset->destroy_addr = &destroy_typed_addr<owner_t>;
    bbq_dataset->dtype        = quantizer->dtype;
    bbq_dataset->mem_type     = CUVS_DATASET_MEM_TYPE_DEVICE;
    bbq_dataset->layout       = CUVS_DATASET_LAYOUT_BBQ;
    bbq_dataset->is_owning    = true;
  });
}

extern "C" cuvsError_t cuvsBbqQuantizerGetDim(cuvsBbqQuantizer_t quantizer, uint32_t* dim)
{
  return cuvs::core::translate_exceptions([=] {
    RAFT_EXPECTS(dim != nullptr, "cuvsBbqQuantizerGetDim: null output");
    with_quantizer_view(
      quantizer, "cuvsBbqQuantizerGetDim", [&](auto const& view) { *dim = view.dim(); });
  });
}

extern "C" cuvsError_t cuvsBbqQuantizerGetCentroid(cuvsBbqQuantizer_t quantizer,
                                                   DLManagedTensor* centroid)
{
  return cuvs::core::translate_exceptions([=] {
    RAFT_EXPECTS(centroid != nullptr, "cuvsBbqQuantizerGetCentroid: null output");
    with_quantizer_view(quantizer, "cuvsBbqQuantizerGetCentroid", [&](auto const& view) {
      cuvs::core::to_dlpack(view.centroid, centroid);
    });
  });
}

extern "C" cuvsError_t cuvsBbqQuantizerGetCentroidNormSq(cuvsBbqQuantizer_t quantizer,
                                                         float* centroid_norm_sq)
{
  return cuvs::core::translate_exceptions([=] {
    RAFT_EXPECTS(centroid_norm_sq != nullptr, "cuvsBbqQuantizerGetCentroidNormSq: null output");
    with_quantizer_view(quantizer, "cuvsBbqQuantizerGetCentroidNormSq", [&](auto const& view) {
      *centroid_norm_sq = view.centroid_norm_sq;
    });
  });
}

extern "C" cuvsError_t cuvsBbqGetEncodedRowLength(uint32_t dim,
                                                  cuvsBbqCodeLayout_t layout,
                                                  uint32_t* row_length)
{
  return cuvs::core::translate_exceptions([=] {
    RAFT_EXPECTS(row_length != nullptr, "cuvsBbqGetEncodedRowLength: null output");
    *row_length = bbq::get_encoded_row_length(
      dim, bbq::c_api::to_cpp_layout(layout, "cuvsBbqGetEncodedRowLength"));
  });
}

extern "C" cuvsError_t cuvsBbqQuantizerCreateView(
  DLManagedTensor* codes,
  DLManagedTensor* lower_intervals,
  DLManagedTensor* upper_intervals,
  DLManagedTensor* additional_corrections,
  DLManagedTensor* quantized_component_sums,
  DLManagedTensor* centroid,
  DLManagedTensor* dequant_delta,
  DLManagedTensor* dequant_sum_delta,
  DLManagedTensor* row_norm,
  cuvsBbqCodeLayout_t layout,
  cuvsDistanceType metric,
  float centroid_norm_sq,
  cuvsBbqQuantizer_t* quantizer)
{
  return cuvs::core::translate_exceptions([=] {
    RAFT_EXPECTS(quantizer != nullptr, "cuvsBbqQuantizerCreateView: null output");
    *quantizer = nullptr;
    RAFT_EXPECTS(codes != nullptr && lower_intervals != nullptr && upper_intervals != nullptr &&
                   additional_corrections != nullptr && quantized_component_sums != nullptr &&
                   centroid != nullptr && dequant_delta != nullptr &&
                   dequant_sum_delta != nullptr && row_norm != nullptr,
                 "cuvsBbqQuantizerCreateView: null tensor");
    auto dtype = centroid->dl_tensor.dtype;
    if (dtype.code == kDLFloat && dtype.bits == 32) {
      make_and_bind_quantizer_view<float>(quantizer,
                                          dtype,
                                          codes,
                                          lower_intervals,
                                          upper_intervals,
                                          additional_corrections,
                                          quantized_component_sums,
                                          centroid,
                                          dequant_delta,
                                          dequant_sum_delta,
                                          row_norm,
                                          layout,
                                          metric,
                                          centroid_norm_sq);
    } else if (dtype.code == kDLFloat && dtype.bits == 16) {
      make_and_bind_quantizer_view<half>(quantizer,
                                         dtype,
                                         codes,
                                         lower_intervals,
                                         upper_intervals,
                                         additional_corrections,
                                         quantized_component_sums,
                                         centroid,
                                         dequant_delta,
                                         dequant_sum_delta,
                                         row_norm,
                                         layout,
                                         metric,
                                         centroid_norm_sq);
    } else if (dtype.code == kDLInt && dtype.bits == 8) {
      make_and_bind_quantizer_view<int8_t>(quantizer,
                                           dtype,
                                           codes,
                                           lower_intervals,
                                           upper_intervals,
                                           additional_corrections,
                                           quantized_component_sums,
                                           centroid,
                                           dequant_delta,
                                           dequant_sum_delta,
                                           row_norm,
                                           layout,
                                           metric,
                                           centroid_norm_sq);
    } else if (dtype.code == kDLUInt && dtype.bits == 8) {
      make_and_bind_quantizer_view<uint8_t>(quantizer,
                                            dtype,
                                            codes,
                                            lower_intervals,
                                            upper_intervals,
                                            additional_corrections,
                                            quantized_component_sums,
                                            centroid,
                                            dequant_delta,
                                            dequant_sum_delta,
                                            row_norm,
                                            layout,
                                            metric,
                                            centroid_norm_sq);
    } else {
      RAFT_FAIL("cuvsBbqQuantizerCreateView: unsupported centroid dtype: code=%d, bits=%d",
                dtype.code,
                dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsBbqQuantizerDestroy(cuvsBbqQuantizer_t quantizer)
{
  return cuvs::core::translate_exceptions([=] {
    if (quantizer == nullptr) { return; }
    if (quantizer->destroy_addr != nullptr && quantizer->addr != 0) {
      quantizer->destroy_addr(reinterpret_cast<void*>(quantizer->addr));
    }
    delete quantizer;
  });
}
