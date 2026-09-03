/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ann_nn_descent.cuh"
#include "../preprocessing/bbq_util.cuh"

#include <cuvs/preprocessing/quantize/bbq.hpp>

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

struct AnnNNDescentBbqInputs : AnnNNDescentInputs {
  uint8_t bits;
  cuvs::preprocessing::quantize::bbq::bbq_code_layout layout;
  std::optional<uint8_t> second_dataset_bits;
};

inline ::std::ostream& operator<<(::std::ostream& os, const AnnNNDescentBbqInputs& p)
{
  os << "dataset shape=" << p.n_rows << "x" << p.dim << ", graph_degree=" << p.graph_degree
     << ", metric="
     << cuvs::neighbors::print_metric{static_cast<cuvs::distance::DistanceType>((int)p.metric)}
     << (p.host_dataset ? ", host" : ", device") << ", bits=" << static_cast<int>(p.bits)
     << ", layout=" << static_cast<int>(p.layout) << ", second_dataset_bits="
     << (p.second_dataset_bits.has_value() ? static_cast<int>(p.second_dataset_bits.value()) : 0)
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
    if (ps.second_dataset_bits.has_value()) {
      if (ps.bits > 4 ||
          (ps.bits == 4 &&
           ps.layout == cuvs::preprocessing::quantize::bbq::bbq_code_layout::packed_nibble) ||
          ps.bits == ps.second_dataset_bits.value() || ps.bits == 1) {
        GTEST_SKIP() << "Second dataset is N/A: bits=" << static_cast<int>(ps.bits)
                     << ", layout=" << static_cast<int>(ps.layout)
                     << " and second bits=" << static_cast<int>(ps.second_dataset_bits.value());
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

      auto bbq_quantizer =
        cuvs::preprocessing::cpu_bbq::quantize_on_cpu(handle_, host_data, ps.n_rows, ps.dim, ps.bits, ps.metric, ps.layout);
      auto bbq_dataset = cuvs::neighbors::device_bbq_dataset<float, int64_t>{std::move(bbq_quantizer)};

      if (ps.second_dataset_bits.has_value()) {
        auto second_layout           = ps.second_dataset_bits.value() == 1
                                         ? cuvs::preprocessing::quantize::bbq::bbq_code_layout::single_bit
                                         : cuvs::preprocessing::quantize::bbq::bbq_code_layout::dibit;
        auto bbq_quantizer_second = cuvs::preprocessing::cpu_bbq::quantize_on_cpu(
          handle_, host_data, ps.n_rows, ps.dim, ps.second_dataset_bits.value(), ps.metric, second_layout);
        bbq_dataset.add_quantizer(std::move(bbq_quantizer_second));
      }
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
        stream_, [&] { index.emplace(nn_descent::build(handle_, index_params, bbq_dataset.as_dataset_view())); });

      std::ostringstream metric_name;
      metric_name << print_metric{ps.metric};
      RAFT_LOG_INFO(
        "NN-Descent build timing: bbq(%u-bit,layout=%d, second_bits=%d) dense=%.3f ms, bbq=%.3f "
        "ms, speedup=%.2fx "
        "(n_rows=%d, dim=%d, graph_degree=%d, metric=%s)",
        static_cast<unsigned>(ps.bits),
        static_cast<int>(ps.layout),
        static_cast<int>(ps.second_dataset_bits.has_value() ? ps.second_dataset_bits.value() : 0),
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
  const std::vector<std::tuple<uint8_t, double, bbq_code_layout, std::optional<uint8_t>>>
    bits_specifications{// bits, min_recall, layout
                        {1, 0.15, bbq_code_layout::single_bit, std::optional<uint8_t>{}},
                        {2, 0.50, bbq_code_layout::dibit, std::optional<uint8_t>{}},
                        {2, 0.27, bbq_code_layout::dibit, std::optional<uint8_t>{1}},
                        {4, 0.80, bbq_code_layout::packed_nibble, std::optional<uint8_t>{}},
                        {4, 0.80, bbq_code_layout::transpose_half_byte, std::optional<uint8_t>{}},
                        {4, 0.35, bbq_code_layout::transpose_half_byte, std::optional<uint8_t>{1}},
                        {4, 0.65, bbq_code_layout::transpose_half_byte, std::optional<uint8_t>{2}},
                        {7, 0.93, bbq_code_layout::seven_bit, std::optional<uint8_t>{}},
                        {8, 0.95, bbq_code_layout::unsigned_byte, std::optional<uint8_t>{}}};
  std::vector<AnnNNDescentBbqInputs> out;
  for (const auto& [bits, min_recall, layout, second_bits] : bits_specifications) {
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
      {second_bits});
    out.insert(out.end(), batch.begin(), batch.end());
  }
  return out;
}();

}  // namespace cuvs::neighbors::nn_descent
