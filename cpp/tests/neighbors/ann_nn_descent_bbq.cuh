/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ann_nn_descent.cuh"

#include <cuvs/preprocessing/quantize/bbq.hpp>
#include <cuvs_internal/preprocessing/bbq_cpu_quantize.hpp>

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

namespace cuvs::neighbors::nn_descent {
// The host reference quantizer is shared with the ann-bench CAGRA wrapper, so these tests and
// the benchmark can never disagree about the code format.
namespace cpu_bbq = cuvs_internal::bbq;
using cuvs_internal::bbq::make_device_bbq_dataset;

// CUDA-event elapsed time around @p fn on @p stream. Because the stop event is
// recorded only after @p fn returns, stream-idle gaps from host work inside NN-Descent
// are included — so this is effectively a wall-clock bracket of the build.
template <typename Fn>
float time_cuda_ms(rmm::cuda_stream_view stream, Fn&& fn)
{
  cudaEvent_t start{};
  cudaEvent_t stop{};
  RAFT_CUDA_TRY(cudaEventCreate(&start));
  RAFT_CUDA_TRY(cudaEventCreate(&stop));
  RAFT_CUDA_TRY(cudaEventRecord(start, stream));
  std::forward<Fn>(fn)();
  RAFT_CUDA_TRY(cudaEventRecord(stop, stream));
  RAFT_CUDA_TRY(cudaEventSynchronize(stop));
  float ms = 0.0f;
  RAFT_CUDA_TRY(cudaEventElapsedTime(&ms, start, stop));
  RAFT_CUDA_TRY(cudaEventDestroy(start));
  RAFT_CUDA_TRY(cudaEventDestroy(stop));
  return ms;
}

// Each layout encodes exactly one code width -- that is what the packed_Nb / transposed_Nb
// naming means -- so the width is derived rather than carried alongside, which would allow the
// two to disagree.
constexpr uint8_t bits_of(cuvs::preprocessing::quantize::bbq::bbq_code_layout layout)
{
  switch (layout) {
    case cuvs::preprocessing::quantize::bbq::bbq_code_layout::packed_1b: return 1;
    case cuvs::preprocessing::quantize::bbq::bbq_code_layout::packed_2b:
    case cuvs::preprocessing::quantize::bbq::bbq_code_layout::transposed_2b: return 2;
    case cuvs::preprocessing::quantize::bbq::bbq_code_layout::packed_4b:
    case cuvs::preprocessing::quantize::bbq::bbq_code_layout::transposed_4b: return 4;
    case cuvs::preprocessing::quantize::bbq::bbq_code_layout::packed_7b: return 7;
    case cuvs::preprocessing::quantize::bbq::bbq_code_layout::packed_8b: return 8;
  }
  return 0;
}

struct AnnNNDescentBbqInputs : AnnNNDescentInputs {
  uint8_t bits;
  cuvs::preprocessing::quantize::bbq::bbq_code_layout layout;
  // Document layout for the second (asymmetric) dataset; its width follows via bits_of().
  std::optional<cuvs::preprocessing::quantize::bbq::bbq_code_layout> second_dataset_layout;
};

inline ::std::ostream& operator<<(::std::ostream& os, const AnnNNDescentBbqInputs& p)
{
  os << "dataset shape=" << p.n_rows << "x" << p.dim << ", graph_degree=" << p.graph_degree
     << ", metric="
     << cuvs::neighbors::print_metric{static_cast<cuvs::distance::DistanceType>((int)p.metric)}
     << (p.host_dataset ? ", host" : ", device") << ", bits=" << static_cast<int>(p.bits)
     << ", layout=" << static_cast<int>(p.layout) << ", second_dataset_layout="
     << (p.second_dataset_layout.has_value() ? static_cast<int>(p.second_dataset_layout.value())
                                             : -1)
     << std::endl;
  return os;
}

class AnnNNDescentBbqTest : public ::testing::TestWithParam<AnnNNDescentBbqInputs> {
 public:
  AnnNNDescentBbqTest()
    : stream_(raft::resource::get_cuda_stream(handle_)),
      ps(::testing::TestWithParam<AnnNNDescentBbqInputs>::GetParam()),
      database(raft::make_device_matrix<float, int64_t>(handle_, ps.n_rows, ps.dim))
  {
  }

 protected:
  void testNNDescent()
  {
    if (ps.second_dataset_layout.has_value()) {
      // The document must be strictly coarser than the query; the pair's layouts are stated in
      // the spec, so validity of the layout combination is the spec's business, not inferred here.
      const uint8_t second_bits = bits_of(ps.second_dataset_layout.value());
      if (ps.bits > 4 || ps.bits == second_bits || ps.bits == 1) {
        GTEST_SKIP() << "Second dataset is N/A: bits=" << static_cast<int>(ps.bits)
                     << ", layout=" << static_cast<int>(ps.layout)
                     << " and second bits=" << static_cast<int>(second_bits);
      }
    }
    size_t queries_size = ps.n_rows * ps.graph_degree;
    std::vector<uint32_t> indices_NNDescent(queries_size);
    std::vector<float> distances_NNDescent(queries_size);
    std::vector<uint32_t> indices_naive(queries_size);
    std::vector<float> distances_naive(queries_size);

    {
      rmm::device_uvector<float> distances_naive_dev(queries_size, stream_);
      rmm::device_uvector<uint32_t> indices_naive_dev(queries_size, stream_);
      naive_knn<float, float, uint32_t>(handle_,
                                        distances_naive_dev.data(),
                                        indices_naive_dev.data(),
                                        database.data_handle(),
                                        database.data_handle(),
                                        ps.n_rows,
                                        ps.n_rows,
                                        ps.dim,
                                        ps.graph_degree,
                                        ps.metric);
      raft::update_host(indices_naive.data(), indices_naive_dev.data(), queries_size, stream_);
      raft::update_host(distances_naive.data(), distances_naive_dev.data(), queries_size, stream_);
      raft::resource::sync_stream(handle_);
    }

    {
      std::vector<float> host_data(static_cast<size_t>(ps.n_rows) * ps.dim);
      raft::update_host(host_data.data(), database.data_handle(), host_data.size(), stream_);
      raft::resource::sync_stream(handle_);

      auto bbq_host_storage =
        cpu_bbq::quantize(host_data, ps.n_rows, ps.dim, ps.bits, ps.metric, ps.layout);
      auto bbq_host = cuvs::neighbors::host_bbq_dataset<int64_t>{std::move(bbq_host_storage)};

      if (ps.second_dataset_layout.has_value()) {
        auto bbq_host_second_storage = cpu_bbq::quantize(host_data,
                                                         ps.n_rows,
                                                         ps.dim,
                                                         bits_of(ps.second_dataset_layout.value()),
                                                         ps.metric,
                                                         ps.second_dataset_layout.value());
        bbq_host.add_quantizer(std::move(bbq_host_second_storage));
      }
      auto owning_dataset = make_device_bbq_dataset(handle_, bbq_host);
      auto dataset        = owning_dataset.as_dataset_view();
      nn_descent::index_params index_params;
      index_params.metric                    = ps.metric;
      index_params.graph_degree              = ps.graph_degree;
      index_params.intermediate_graph_degree = 2 * ps.graph_degree;
      index_params.max_iterations            = 100;
      index_params.return_distances          = true;

      // Dense float baseline on the same data / params, timed with the same CUDA events.
      const float dense_ms = time_cuda_ms(stream_, [&] {
        auto database_view = raft::make_const_mdspan(database.view());
        auto dense_index   = nn_descent::build(handle_, index_params, database_view);
        (void)dense_index;
      });

      std::optional<index<uint32_t>> index;
      const float bbq_ms = time_cuda_ms(
        stream_, [&] { index.emplace(nn_descent::build(handle_, index_params, dataset)); });

      std::ostringstream metric_name;
      metric_name << print_metric{ps.metric};
      RAFT_LOG_INFO(
        "NN-Descent build timing: bbq(%u-bit,layout=%d, second_bits=%d) dense=%.3f ms, bbq=%.3f "
        "ms, speedup=%.2fx "
        "(n_rows=%d, dim=%d, graph_degree=%d, metric=%s)",
        static_cast<unsigned>(ps.bits),
        static_cast<int>(ps.layout),
        static_cast<int>(
          ps.second_dataset_layout.has_value() ? bits_of(ps.second_dataset_layout.value()) : 0),
        dense_ms,
        bbq_ms,
        dense_ms / std::max(bbq_ms, 1e-3f),
        ps.n_rows,
        ps.dim,
        ps.graph_degree,
        metric_name.str().c_str());

      raft::copy(indices_NNDescent.data(), index->graph().data_handle(), queries_size, stream_);
      ASSERT_TRUE(index->distances().has_value());
      raft::copy(distances_NNDescent.data(),
                 index->distances().value().data_handle(),
                 queries_size,
                 stream_);
      raft::resource::sync_stream(handle_);
    }

    EXPECT_TRUE(eval_neighbours(indices_naive,
                                indices_NNDescent,
                                distances_naive,
                                distances_NNDescent,
                                ps.n_rows,
                                ps.graph_degree,
                                0.001,
                                ps.min_recall));
  }

  void SetUp() override
  {
    raft::random::RngState r(1234ULL);
    raft::random::normal(handle_, r, database.data_handle(), ps.n_rows * ps.dim, 0.1f, 2.0f);
    raft::resource::sync_stream(handle_);
  }

  void TearDown() override { raft::resource::sync_stream(handle_); }

 private:
  raft::resources handle_;
  rmm::cuda_stream_view stream_;
  AnnNNDescentBbqInputs ps;
  raft::device_matrix<float, int64_t> database;
};

// Estimated recall based on bruteforce (InnerProduct): 1: 0.23, 2: 0.52, 4: 0.85, 7: 0.98, 8: 0.99.
const std::vector<AnnNNDescentBbqInputs> bbq_inputs = [] {
  using cuvs::preprocessing::quantize::bbq::bbq_code_layout;
  const std::vector<std::tuple<uint8_t, double, bbq_code_layout, std::optional<bbq_code_layout>>>
    bits_specifications{
      // query bits, min_recall, query layout, document layout (width follows from it)
      {1, 0.15, bbq_code_layout::packed_1b, std::optional<bbq_code_layout>{}},
      {2, 0.50, bbq_code_layout::transposed_2b, std::optional<bbq_code_layout>{}},
      {2,
       0.27,
       bbq_code_layout::transposed_2b,
       std::optional<bbq_code_layout>{bbq_code_layout::packed_1b}},
      {4, 0.80, bbq_code_layout::packed_4b, std::optional<bbq_code_layout>{}},
      // Asymmetric packed_4b queries take the int4 wmma path (SelfJoin = false). At dim=256 these
      // are also the only coverage of the phase-2 staging skip (n_tiles == 1) outside a self-join.
      {4,
       0.35,
       bbq_code_layout::packed_4b,
       std::optional<bbq_code_layout>{bbq_code_layout::packed_1b}},
      {4,
       0.65,
       bbq_code_layout::packed_4b,
       std::optional<bbq_code_layout>{bbq_code_layout::packed_2b}},
      // Asymmetric transposed_4b queries (1 + 4t, 2t + 4t) take the SIMT path.
      {4,
       0.35,
       bbq_code_layout::transposed_4b,
       std::optional<bbq_code_layout>{bbq_code_layout::packed_1b}},
      {4,
       0.65,
       bbq_code_layout::transposed_4b,
       std::optional<bbq_code_layout>{bbq_code_layout::transposed_2b}},
      {7, 0.93, bbq_code_layout::packed_7b, std::optional<bbq_code_layout>{}},
      {8, 0.95, bbq_code_layout::packed_8b, std::optional<bbq_code_layout>{}}};
  std::vector<AnnNNDescentBbqInputs> out;
  for (const auto& [bits, min_recall, layout, second_layout] : bits_specifications) {
    const auto batch = raft::util::itertools::product<AnnNNDescentBbqInputs>(
      {20000},
      {256, 1024},  // dim
      {128},        // graph_degree
      {cuvs::distance::DistanceType::L2Expanded,
       cuvs::distance::DistanceType::L2SqrtExpanded,
       cuvs::distance::DistanceType::InnerProduct,
       cuvs::distance::DistanceType::CosineExpanded},
      {false},  // host_dataset
      {min_recall},
      {bits},
      {layout},
      {second_layout});
    out.insert(out.end(), batch.begin(), batch.end());
  }
  return out;
}();

}  // namespace cuvs::neighbors::nn_descent
