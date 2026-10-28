/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../../src/core/interop.hpp"
#include "../../src/preprocessing/quantize/bbq_helpers.hpp"

#include <cuvs/core/c_api.h>
#include <cuvs/core/dataset.h>
#include <cuvs/neighbors/cagra.h>
#include <cuvs/preprocessing/quantize/bbq.h>
#include <cuvs/preprocessing/quantize/bbq.hpp>
#include <dlpack/dlpack.h>

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>
#include <rmm/device_uvector.hpp>

#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace {

namespace bbq = cuvs::preprocessing::quantize::bbq;

static_assert(CUVS_BBQ_CODE_LAYOUT_PACKED_1B == 0);
static_assert(CUVS_BBQ_CODE_LAYOUT_TRANSPOSED_2B == 1);
static_assert(CUVS_BBQ_CODE_LAYOUT_TRANSPOSED_4B == 2);
static_assert(CUVS_BBQ_CODE_LAYOUT_PACKED_4B == 3);
static_assert(CUVS_BBQ_CODE_LAYOUT_PACKED_7B == 4);
static_assert(CUVS_BBQ_CODE_LAYOUT_PACKED_8B == 5);

template <typename T>
auto make_device_matrix_tensor(T* data, int64_t rows, int64_t columns) -> DLManagedTensor
{
  DLManagedTensor tensor{};
  cuvs::core::to_dlpack(raft::make_device_matrix_view<T, int64_t>(data, rows, columns), &tensor);
  return tensor;
}

template <typename T>
auto make_device_vector_tensor(T* data, int64_t size) -> DLManagedTensor
{
  DLManagedTensor tensor{};
  cuvs::core::to_dlpack(raft::make_device_vector_view<T, int64_t>(data, size), &tensor);
  return tensor;
}

void free_tensor(DLManagedTensor& tensor)
{
  if (tensor.deleter != nullptr) { tensor.deleter(&tensor); }
}

template <typename T>
auto to_host(const T* data, std::size_t size, cudaStream_t stream) -> std::vector<T>
{
  std::vector<T> host(size);
  raft::copy(host.data(), data, size, stream);
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  return host;
}

auto random_host_matrix(int64_t n_rows, int64_t dim, uint32_t seed) -> std::vector<float>
{
  std::mt19937 gen(seed);
  std::normal_distribution<float> dist(0.5f, 2.0f);
  std::vector<float> host(n_rows * dim);
  for (auto& v : host) {
    v = dist(gen);
  }
  return host;
}

auto last_error() -> std::string { return std::string(cuvsGetLastErrorText()); }

/** Device-resident float32 matrix and its DLPack tensor. */
struct DeviceMatrix {
  rmm::device_uvector<float> data;
  DLManagedTensor tensor;

  DeviceMatrix(std::vector<float> const& host, int64_t n_rows, int64_t dim, cudaStream_t stream)
    : data(host.size(), stream), tensor{}
  {
    raft::copy(data.data(), host.data(), host.size(), stream);
    tensor = make_device_matrix_tensor(data.data(), n_rows, dim);
  }
  ~DeviceMatrix() { free_tensor(tensor); }
};

/** Tensors backing a view quantizer of `n_rows` packed_1b rows of dimension `dim`. */
struct ViewTensors {
  rmm::device_uvector<uint8_t> codes;
  rmm::device_uvector<float> lower, upper, corrections;
  rmm::device_uvector<int32_t> sums;
  rmm::device_uvector<float> centroid, delta, sum_delta, row_norm;
  DLManagedTensor codes_t, lower_t, upper_t, corrections_t, sums_t, centroid_t, delta_t,
    sum_delta_t, row_norm_t;

  ViewTensors(int64_t n_rows, int64_t dim, cudaStream_t stream)
    : codes(n_rows * ((dim + 7) / 8), stream),
      lower(n_rows, stream),
      upper(n_rows, stream),
      corrections(n_rows, stream),
      sums(n_rows, stream),
      centroid(dim, stream),
      delta(n_rows, stream),
      sum_delta(n_rows, stream),
      row_norm(n_rows, stream)
  {
    codes_t       = make_device_matrix_tensor(codes.data(), n_rows, (dim + 7) / 8);
    lower_t       = make_device_vector_tensor(lower.data(), n_rows);
    upper_t       = make_device_vector_tensor(upper.data(), n_rows);
    corrections_t = make_device_vector_tensor(corrections.data(), n_rows);
    sums_t        = make_device_vector_tensor(sums.data(), n_rows);
    centroid_t    = make_device_vector_tensor(centroid.data(), dim);
    delta_t       = make_device_vector_tensor(delta.data(), n_rows);
    sum_delta_t   = make_device_vector_tensor(sum_delta.data(), n_rows);
    row_norm_t    = make_device_vector_tensor(row_norm.data(), n_rows);
  }
  ~ViewTensors()
  {
    for (auto* t : {&codes_t,
                    &lower_t,
                    &upper_t,
                    &corrections_t,
                    &sums_t,
                    &centroid_t,
                    &delta_t,
                    &sum_delta_t,
                    &row_norm_t}) {
      free_tensor(*t);
    }
  }

  auto create_view(float centroid_norm_sq, cuvsBbqQuantizer_t* quantizer) -> cuvsError_t
  {
    return cuvsBbqQuantizerCreateView(&codes_t,
                                      &lower_t,
                                      &upper_t,
                                      &corrections_t,
                                      &sums_t,
                                      &centroid_t,
                                      &delta_t,
                                      &sum_delta_t,
                                      &row_norm_t,
                                      CUVS_BBQ_CODE_LAYOUT_PACKED_1B,
                                      L2Expanded,
                                      centroid_norm_sq,
                                      quantizer);
  }
};

template <typename T, typename IdxT>
void expect_equal_device(raft::device_vector_view<const T, IdxT> actual,
                         raft::device_vector_view<const T, IdxT> expected,
                         cudaStream_t stream,
                         char const* name)
{
  ASSERT_EQ(actual.extent(0), expected.extent(0)) << name;
  auto a = to_host(actual.data_handle(), actual.size(), stream);
  auto e = to_host(expected.data_handle(), expected.size(), stream);
  for (std::size_t i = 0; i < a.size(); ++i) {
    ASSERT_EQ(a[i], e[i]) << name << " differs at " << i;
  }
}

class BbqC : public ::testing::Test {
 protected:
  void SetUp() override
  {
    ASSERT_EQ(cuvsResourcesCreate(&res), CUVS_SUCCESS);
    ASSERT_EQ(cuvsStreamGet(res, &stream), CUVS_SUCCESS);
  }
  void TearDown() override { ASSERT_EQ(cuvsResourcesDestroy(res), CUVS_SUCCESS); }

  cuvsResources_t res{};
  cudaStream_t stream{};
};

}  // namespace

TEST(BbqCParams, Defaults)
{
  cuvsBbqQuantizerParams_t params = nullptr;
  ASSERT_EQ(cuvsBbqQuantizerParamsCreate(&params), CUVS_SUCCESS);
  ASSERT_NE(params, nullptr);
  const bbq::params defaults;
  EXPECT_EQ(params->doc_layout, CUVS_BBQ_CODE_LAYOUT_PACKED_1B);
  EXPECT_EQ(params->query_layout, CUVS_BBQ_CODE_LAYOUT_TRANSPOSED_4B);
  EXPECT_EQ(params->metric, L2Expanded);
  EXPECT_EQ(params->lambda, defaults.lambda);
  EXPECT_EQ(params->iters, static_cast<uint32_t>(defaults.iters));
  ASSERT_EQ(cuvsBbqQuantizerParamsDestroy(params), CUVS_SUCCESS);
  EXPECT_EQ(cuvsBbqQuantizerParamsDestroy(nullptr), CUVS_SUCCESS);
  EXPECT_EQ(cuvsBbqQuantizerParamsCreate(nullptr), CUVS_ERROR);
}

TEST(BbqCParams, EncodedRowLength)
{
  struct expectation {
    cuvsBbqCodeLayout_t layout;
    uint32_t row_length;
  };
  constexpr uint32_t dim = 41;
  for (auto [layout, expected] : {expectation{CUVS_BBQ_CODE_LAYOUT_PACKED_1B, 6},
                                  expectation{CUVS_BBQ_CODE_LAYOUT_TRANSPOSED_2B, 12},
                                  expectation{CUVS_BBQ_CODE_LAYOUT_TRANSPOSED_4B, 24},
                                  expectation{CUVS_BBQ_CODE_LAYOUT_PACKED_4B, 21},
                                  expectation{CUVS_BBQ_CODE_LAYOUT_PACKED_7B, 41},
                                  expectation{CUVS_BBQ_CODE_LAYOUT_PACKED_8B, 41}}) {
    uint32_t row_length = 0;
    ASSERT_EQ(cuvsBbqGetEncodedRowLength(dim, layout, &row_length), CUVS_SUCCESS);
    EXPECT_EQ(row_length, expected) << "layout " << layout;
  }
  uint32_t row_length = 0;
  EXPECT_EQ(cuvsBbqGetEncodedRowLength(dim, static_cast<cuvsBbqCodeLayout_t>(42), &row_length),
            CUVS_ERROR);
  EXPECT_EQ(cuvsBbqGetEncodedRowLength(dim, CUVS_BBQ_CODE_LAYOUT_PACKED_1B, nullptr), CUVS_ERROR);
}

struct BbqCTransformInputs {
  cuvsBbqCodeLayout_t doc_layout;
  cuvsBbqCodeLayout_t query_layout;
  cuvsDistanceType metric;
};

class BbqCTransform : public BbqC, public ::testing::WithParamInterface<BbqCTransformInputs> {};

TEST_P(BbqCTransform, MatchesCppApi)
{
  constexpr int64_t n_rows = 500;
  constexpr int64_t dim    = 40;
  const auto inputs        = GetParam();
  auto host                = random_host_matrix(n_rows, dim, 42);
  DeviceMatrix dataset(host, n_rows, dim, stream);

  cuvsBbqQuantizerParams_t params;
  ASSERT_EQ(cuvsBbqQuantizerParamsCreate(&params), CUVS_SUCCESS);
  params->doc_layout   = inputs.doc_layout;
  params->query_layout = inputs.query_layout;
  params->metric       = inputs.metric;
  params->lambda       = 0.2f;
  params->iters        = 3;

  cuvsBbqQuantizer_t quantizer;
  ASSERT_EQ(cuvsBbqQuantizerCreate(&quantizer), CUVS_SUCCESS);
  ASSERT_EQ(cuvsBbqQuantizerBuild(res, params, &dataset.tensor, quantizer), CUVS_SUCCESS)
    << last_error();
  EXPECT_TRUE(quantizer->is_owning);
  EXPECT_EQ(quantizer->dtype.code, kDLFloat);
  EXPECT_EQ(quantizer->dtype.bits, 32);

  auto const& handle = *reinterpret_cast<raft::resources*>(res);
  auto cpp_dataset =
    raft::make_device_matrix_view<const float, int64_t>(dataset.data.data(), n_rows, dim);
  auto cpp_quantizer = bbq::build(handle, cpp_dataset);
  bbq::params cpp_params;
  cpp_params.doc_layout   = bbq::c_api::to_cpp_layout(inputs.doc_layout, "test");
  cpp_params.query_layout = bbq::c_api::to_cpp_layout(inputs.query_layout, "test");
  cpp_params.metric       = static_cast<cuvs::distance::DistanceType>(inputs.metric);
  cpp_params.lambda       = 0.2f;
  cpp_params.iters        = 3;

  uint32_t c_dim = 0;
  ASSERT_EQ(cuvsBbqQuantizerGetDim(quantizer, &c_dim), CUVS_SUCCESS);
  EXPECT_EQ(c_dim, static_cast<uint32_t>(dim));

  float norm_sq = 0;
  ASSERT_EQ(cuvsBbqQuantizerGetCentroidNormSq(quantizer, &norm_sq), CUVS_SUCCESS);
  EXPECT_EQ(norm_sq, cpp_quantizer.centroid_norm_sq);

  DLManagedTensor centroid{};
  ASSERT_EQ(cuvsBbqQuantizerGetCentroid(quantizer, &centroid), CUVS_SUCCESS);
  ASSERT_EQ(centroid.dl_tensor.ndim, 1);
  ASSERT_EQ(centroid.dl_tensor.shape[0], dim);
  EXPECT_EQ(centroid.dl_tensor.dtype.code, kDLFloat);
  EXPECT_EQ(centroid.dl_tensor.dtype.bits, 32);
  EXPECT_EQ(centroid.dl_tensor.device.device_type, kDLCUDA);
  auto c_centroid = to_host(static_cast<const float*>(centroid.dl_tensor.data), dim, stream);
  free_tensor(centroid);
  auto cpp_centroid       = to_host(cpp_quantizer.centroid.data_handle(), dim, stream);
  double expected_norm_sq = 0;
  for (int64_t j = 0; j < dim; ++j) {
    EXPECT_EQ(c_centroid[j], cpp_centroid[j]) << "centroid differs at " << j;
    double mean = 0;
    for (int64_t i = 0; i < n_rows; ++i) {
      mean += host[i * dim + j];
    }
    mean /= n_rows;
    EXPECT_NEAR(c_centroid[j], mean, 1e-4);
    expected_norm_sq += mean * mean;
  }
  EXPECT_NEAR(norm_sq, expected_norm_sq, 1e-3 * expected_norm_sq);

  cuvsDataset_t encoded;
  ASSERT_EQ(cuvsDatasetCreate(&encoded), CUVS_SUCCESS);
  ASSERT_EQ(cuvsBbqQuantizerTransform(res, quantizer, &dataset.tensor, encoded), CUVS_SUCCESS)
    << last_error();

  cuvsDatasetLayout_t layout;
  cuvsDatasetMemType_t mem_type;
  bool is_owning = false;
  DLDataType dtype;
  ASSERT_EQ(cuvsDatasetGetLayout(encoded, &layout), CUVS_SUCCESS);
  ASSERT_EQ(cuvsDatasetGetMemType(encoded, &mem_type), CUVS_SUCCESS);
  ASSERT_EQ(cuvsDatasetGetIsOwning(encoded, &is_owning), CUVS_SUCCESS);
  ASSERT_EQ(cuvsDatasetGetDtype(encoded, &dtype), CUVS_SUCCESS);
  EXPECT_EQ(layout, CUVS_DATASET_LAYOUT_BBQ);
  EXPECT_EQ(mem_type, CUVS_DATASET_MEM_TYPE_DEVICE);
  EXPECT_TRUE(is_owning);
  EXPECT_EQ(dtype.code, kDLFloat);
  EXPECT_EQ(dtype.bits, 32);

  // The encoded dataset must not depend on the quantizer that produced it.
  ASSERT_EQ(cuvsBbqQuantizerDestroy(quantizer), CUVS_SUCCESS);
  ASSERT_EQ(cuvsBbqQuantizerParamsDestroy(params), CUVS_SUCCESS);

  auto expected = bbq::transform(handle, cpp_params, cpp_quantizer.view(), cpp_dataset);
  auto const& actual =
    *reinterpret_cast<cuvs::neighbors::device_bbq_dataset<float, int64_t>*>(encoded->addr);
  ASSERT_EQ(actual.storages.size(), expected.storages.size());
  ASSERT_EQ(actual.n_rows(), n_rows);
  ASSERT_EQ(actual.dim(), static_cast<uint32_t>(dim));
  for (std::size_t s = 0; s < expected.storages.size(); ++s) {
    auto a = actual.storages[s].view();
    auto e = expected.storages[s].view();
    ASSERT_EQ(a.layout, e.layout);
    ASSERT_EQ(a.metric, e.metric);
    ASSERT_EQ(a.quantizer.centroid_norm_sq, e.quantizer.centroid_norm_sq);
    ASSERT_EQ(a.codes.extent(0), e.codes.extent(0));
    ASSERT_EQ(a.codes.extent(1), e.codes.extent(1));
    expect_equal_device(
      raft::make_device_vector_view<const uint8_t, int64_t>(a.codes.data_handle(), a.codes.size()),
      raft::make_device_vector_view<const uint8_t, int64_t>(e.codes.data_handle(), e.codes.size()),
      stream,
      "codes");
    expect_equal_device(a.lower_intervals, e.lower_intervals, stream, "lower_intervals");
    expect_equal_device(a.upper_intervals, e.upper_intervals, stream, "upper_intervals");
    expect_equal_device(
      a.additional_corrections, e.additional_corrections, stream, "additional_corrections");
    expect_equal_device(
      a.quantized_component_sums, e.quantized_component_sums, stream, "quantized_component_sums");
    expect_equal_device(a.dequant_delta, e.dequant_delta, stream, "dequant_delta");
    expect_equal_device(a.dequant_sum_delta, e.dequant_sum_delta, stream, "dequant_sum_delta");
    expect_equal_device(a.row_norm, e.row_norm, stream, "row_norm");
    expect_equal_device(a.quantizer.centroid, e.quantizer.centroid, stream, "storage centroid");
  }

  ASSERT_EQ(cuvsDatasetDestroy(encoded), CUVS_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(
  BbqC,
  BbqCTransform,
  ::testing::Values(
    BbqCTransformInputs{
      CUVS_BBQ_CODE_LAYOUT_PACKED_1B, CUVS_BBQ_CODE_LAYOUT_TRANSPOSED_4B, L2Expanded},
    BbqCTransformInputs{
      CUVS_BBQ_CODE_LAYOUT_PACKED_1B, CUVS_BBQ_CODE_LAYOUT_PACKED_4B, InnerProduct},
    BbqCTransformInputs{
      CUVS_BBQ_CODE_LAYOUT_TRANSPOSED_2B, CUVS_BBQ_CODE_LAYOUT_TRANSPOSED_4B, L2Expanded},
    BbqCTransformInputs{CUVS_BBQ_CODE_LAYOUT_PACKED_8B, CUVS_BBQ_CODE_LAYOUT_PACKED_8B, L2Expanded},
    BbqCTransformInputs{
      CUVS_BBQ_CODE_LAYOUT_PACKED_7B, CUVS_BBQ_CODE_LAYOUT_PACKED_7B, CosineExpanded}));

TEST_F(BbqC, TransformedDatasetBuildsCagraGraph)
{
  constexpr int64_t n_rows = 256;
  constexpr int64_t dim    = 32;
  DeviceMatrix dataset(random_host_matrix(n_rows, dim, 7), n_rows, dim, stream);

  cuvsBbqQuantizerParams_t params;
  ASSERT_EQ(cuvsBbqQuantizerParamsCreate(&params), CUVS_SUCCESS);
  cuvsBbqQuantizer_t quantizer;
  ASSERT_EQ(cuvsBbqQuantizerCreate(&quantizer), CUVS_SUCCESS);
  ASSERT_EQ(cuvsBbqQuantizerBuild(res, params, &dataset.tensor, quantizer), CUVS_SUCCESS)
    << last_error();
  cuvsDataset_t encoded;
  ASSERT_EQ(cuvsDatasetCreate(&encoded), CUVS_SUCCESS);
  ASSERT_EQ(cuvsBbqQuantizerTransform(res, quantizer, &dataset.tensor, encoded), CUVS_SUCCESS)
    << last_error();
  ASSERT_EQ(cuvsBbqQuantizerDestroy(quantizer), CUVS_SUCCESS);
  ASSERT_EQ(cuvsBbqQuantizerParamsDestroy(params), CUVS_SUCCESS);

  cuvsCagraIndexParams_t index_params;
  ASSERT_EQ(cuvsCagraIndexParamsCreate(&index_params), CUVS_SUCCESS);
  index_params->metric                    = L2Expanded;
  index_params->build_algo                = NN_DESCENT;
  index_params->graph_degree              = 16;
  index_params->intermediate_graph_degree = 32;
  index_params->nn_descent_niter          = 10;

  cuvsCagraIndex_t index;
  ASSERT_EQ(cuvsCagraIndexCreate(&index), CUVS_SUCCESS);
  ASSERT_EQ(cuvsCagraBuild(res, index_params, encoded, index), CUVS_SUCCESS) << last_error();

  DLManagedTensor graph_tensor{};
  ASSERT_EQ(cuvsCagraIndexGetGraph(index, &graph_tensor), CUVS_SUCCESS);
  ASSERT_EQ(graph_tensor.dl_tensor.shape[0], n_rows);
  ASSERT_EQ(graph_tensor.dl_tensor.shape[1], static_cast<int64_t>(index_params->graph_degree));
  auto graph = to_host(static_cast<const uint32_t*>(graph_tensor.dl_tensor.data),
                       n_rows * index_params->graph_degree,
                       stream);
  for (auto neighbor : graph) {
    EXPECT_LT(neighbor, static_cast<uint32_t>(n_rows));
  }
  free_tensor(graph_tensor);

  ASSERT_EQ(cuvsCagraIndexDestroy(index), CUVS_SUCCESS);
  ASSERT_EQ(cuvsCagraIndexParamsDestroy(index_params), CUVS_SUCCESS);
  ASSERT_EQ(cuvsDatasetDestroy(encoded), CUVS_SUCCESS);
}

TEST_F(BbqC, ViewQuantizerGetters)
{
  constexpr int64_t n_rows = 4;
  constexpr int64_t dim    = 8;
  ViewTensors tensors(n_rows, dim, stream);

  cuvsBbqQuantizer_t quantizer;
  ASSERT_EQ(tensors.create_view(3.5f, &quantizer), CUVS_SUCCESS);
  EXPECT_FALSE(quantizer->is_owning);

  uint32_t c_dim = 0;
  ASSERT_EQ(cuvsBbqQuantizerGetDim(quantizer, &c_dim), CUVS_SUCCESS);
  EXPECT_EQ(c_dim, static_cast<uint32_t>(dim));
  float norm_sq = 0;
  ASSERT_EQ(cuvsBbqQuantizerGetCentroidNormSq(quantizer, &norm_sq), CUVS_SUCCESS);
  EXPECT_EQ(norm_sq, 3.5f);
  DLManagedTensor centroid{};
  ASSERT_EQ(cuvsBbqQuantizerGetCentroid(quantizer, &centroid), CUVS_SUCCESS);
  EXPECT_EQ(centroid.dl_tensor.data, tensors.centroid.data());
  ASSERT_EQ(centroid.dl_tensor.ndim, 1);
  EXPECT_EQ(centroid.dl_tensor.shape[0], dim);
  free_tensor(centroid);

  ASSERT_EQ(cuvsBbqQuantizerDestroy(quantizer), CUVS_SUCCESS);
}

TEST_F(BbqC, BuildErrors)
{
  constexpr int64_t n_rows = 64;
  constexpr int64_t dim    = 16;
  auto host                = random_host_matrix(n_rows, dim, 3);
  DeviceMatrix dataset(host, n_rows, dim, stream);

  cuvsBbqQuantizerParams_t params;
  ASSERT_EQ(cuvsBbqQuantizerParamsCreate(&params), CUVS_SUCCESS);

  auto expect_build_error = [&](DLManagedTensor* tensor, char const* message) {
    cuvsBbqQuantizer_t quantizer;
    ASSERT_EQ(cuvsBbqQuantizerCreate(&quantizer), CUVS_SUCCESS);
    EXPECT_EQ(cuvsBbqQuantizerBuild(res, params, tensor, quantizer), CUVS_ERROR) << message;
    EXPECT_NE(last_error().find(message), std::string::npos) << last_error();
    EXPECT_EQ(quantizer->addr, 0u);
    ASSERT_EQ(cuvsBbqQuantizerDestroy(quantizer), CUVS_SUCCESS);
  };

  expect_build_error(nullptr, "null dataset");

  DLManagedTensor host_tensor{};
  cuvs::core::to_dlpack(raft::make_host_matrix_view<float, int64_t>(host.data(), n_rows, dim),
                        &host_tensor);
  expect_build_error(&host_tensor, "device-accessible");
  free_tensor(host_tensor);

  rmm::device_uvector<half> half_data(n_rows * dim, stream);
  auto half_tensor = make_device_matrix_tensor(half_data.data(), n_rows, dim);
  expect_build_error(&half_tensor, "only float32");
  free_tensor(half_tensor);

  params->lambda = 1.5f;
  expect_build_error(&dataset.tensor, "lambda");
  params->lambda = 0.1f;

  params->doc_layout = static_cast<cuvsBbqCodeLayout_t>(42);
  expect_build_error(&dataset.tensor, "invalid BBQ code layout");
  params->doc_layout = CUVS_BBQ_CODE_LAYOUT_PACKED_1B;

  params->query_layout = CUVS_BBQ_CODE_LAYOUT_PACKED_8B;
  expect_build_error(&dataset.tensor, "unsupported asymmetric BBQ layout pair");

  params->doc_layout   = CUVS_BBQ_CODE_LAYOUT_TRANSPOSED_4B;
  params->query_layout = CUVS_BBQ_CODE_LAYOUT_TRANSPOSED_4B;
  expect_build_error(&dataset.tensor, "transposed_4b requires an asymmetric layout pair");
  params->doc_layout   = CUVS_BBQ_CODE_LAYOUT_PACKED_1B;
  params->query_layout = CUVS_BBQ_CODE_LAYOUT_TRANSPOSED_4B;

  cuvsBbqQuantizer_t quantizer;
  ASSERT_EQ(cuvsBbqQuantizerCreate(&quantizer), CUVS_SUCCESS);
  EXPECT_EQ(cuvsBbqQuantizerBuild(res, nullptr, &dataset.tensor, quantizer), CUVS_ERROR);
  EXPECT_EQ(cuvsBbqQuantizerBuild(res, params, &dataset.tensor, nullptr), CUVS_ERROR);
  ASSERT_EQ(cuvsBbqQuantizerBuild(res, params, &dataset.tensor, quantizer), CUVS_SUCCESS);
  EXPECT_EQ(cuvsBbqQuantizerBuild(res, params, &dataset.tensor, quantizer), CUVS_ERROR);
  EXPECT_NE(last_error().find("already been built"), std::string::npos) << last_error();
  ASSERT_EQ(cuvsBbqQuantizerDestroy(quantizer), CUVS_SUCCESS);

  ViewTensors view_tensors(4, dim, stream);
  cuvsBbqQuantizer_t view;
  ASSERT_EQ(view_tensors.create_view(0.0f, &view), CUVS_SUCCESS);
  EXPECT_EQ(cuvsBbqQuantizerBuild(res, params, &dataset.tensor, view), CUVS_ERROR);
  EXPECT_NE(last_error().find("cuvsBbqQuantizerCreateView"), std::string::npos) << last_error();
  ASSERT_EQ(cuvsBbqQuantizerDestroy(view), CUVS_SUCCESS);

  ASSERT_EQ(cuvsBbqQuantizerParamsDestroy(params), CUVS_SUCCESS);
}

TEST_F(BbqC, TransformAndGetterErrors)
{
  constexpr int64_t n_rows = 64;
  constexpr int64_t dim    = 16;
  DeviceMatrix dataset(random_host_matrix(n_rows, dim, 5), n_rows, dim, stream);
  DeviceMatrix narrow(random_host_matrix(n_rows, dim / 2, 6), n_rows, dim / 2, stream);

  cuvsBbqQuantizerParams_t params;
  ASSERT_EQ(cuvsBbqQuantizerParamsCreate(&params), CUVS_SUCCESS);
  cuvsBbqQuantizer_t quantizer;
  ASSERT_EQ(cuvsBbqQuantizerCreate(&quantizer), CUVS_SUCCESS);
  cuvsDataset_t encoded;
  ASSERT_EQ(cuvsDatasetCreate(&encoded), CUVS_SUCCESS);

  uint32_t c_dim;
  float norm_sq;
  DLManagedTensor centroid{};
  EXPECT_EQ(cuvsBbqQuantizerGetDim(quantizer, &c_dim), CUVS_ERROR);
  EXPECT_NE(last_error().find("has not been built"), std::string::npos) << last_error();
  EXPECT_EQ(cuvsBbqQuantizerGetCentroidNormSq(quantizer, &norm_sq), CUVS_ERROR);
  EXPECT_EQ(cuvsBbqQuantizerGetCentroid(quantizer, &centroid), CUVS_ERROR);
  EXPECT_EQ(cuvsBbqQuantizerTransform(res, quantizer, &dataset.tensor, encoded), CUVS_ERROR);
  EXPECT_NE(last_error().find("has not been built"), std::string::npos) << last_error();

  ASSERT_EQ(cuvsBbqQuantizerBuild(res, params, &dataset.tensor, quantizer), CUVS_SUCCESS);
  EXPECT_EQ(cuvsBbqQuantizerGetDim(quantizer, nullptr), CUVS_ERROR);
  EXPECT_EQ(cuvsBbqQuantizerGetCentroidNormSq(quantizer, nullptr), CUVS_ERROR);
  EXPECT_EQ(cuvsBbqQuantizerGetCentroid(quantizer, nullptr), CUVS_ERROR);
  EXPECT_EQ(cuvsBbqQuantizerGetDim(nullptr, &c_dim), CUVS_ERROR);

  EXPECT_EQ(cuvsBbqQuantizerTransform(res, nullptr, &dataset.tensor, encoded), CUVS_ERROR);
  EXPECT_EQ(cuvsBbqQuantizerTransform(res, quantizer, nullptr, encoded), CUVS_ERROR);
  EXPECT_EQ(cuvsBbqQuantizerTransform(res, quantizer, &dataset.tensor, nullptr), CUVS_ERROR);
  EXPECT_EQ(cuvsBbqQuantizerTransform(res, quantizer, &narrow.tensor, encoded), CUVS_ERROR);
  EXPECT_EQ(encoded->addr, 0u);

  ASSERT_EQ(cuvsBbqQuantizerTransform(res, quantizer, &dataset.tensor, encoded), CUVS_SUCCESS);
  EXPECT_EQ(cuvsBbqQuantizerTransform(res, quantizer, &dataset.tensor, encoded), CUVS_ERROR);
  EXPECT_NE(last_error().find("must be empty"), std::string::npos) << last_error();

  cuvsDataset_t view_dataset = nullptr;
  EXPECT_EQ(cuvsDatasetMakeBbqView(res, &quantizer, 1, &view_dataset), CUVS_ERROR);
  EXPECT_NE(last_error().find("cuvsBbqQuantizerTransform"), std::string::npos) << last_error();
  EXPECT_EQ(view_dataset, nullptr);

  ViewTensors view_tensors(4, dim, stream);
  cuvsBbqQuantizer_t view;
  ASSERT_EQ(view_tensors.create_view(0.0f, &view), CUVS_SUCCESS);
  cuvsDataset_t other;
  ASSERT_EQ(cuvsDatasetCreate(&other), CUVS_SUCCESS);
  EXPECT_EQ(cuvsBbqQuantizerTransform(res, view, &dataset.tensor, other), CUVS_ERROR);
  EXPECT_NE(last_error().find("cuvsBbqQuantizerBuild"), std::string::npos) << last_error();
  ASSERT_EQ(cuvsDatasetDestroy(other), CUVS_SUCCESS);
  ASSERT_EQ(cuvsBbqQuantizerDestroy(view), CUVS_SUCCESS);

  ASSERT_EQ(cuvsDatasetDestroy(encoded), CUVS_SUCCESS);
  ASSERT_EQ(cuvsBbqQuantizerDestroy(quantizer), CUVS_SUCCESS);
  ASSERT_EQ(cuvsBbqQuantizerParamsDestroy(params), CUVS_SUCCESS);
  EXPECT_EQ(cuvsBbqQuantizerDestroy(nullptr), CUVS_SUCCESS);
}
