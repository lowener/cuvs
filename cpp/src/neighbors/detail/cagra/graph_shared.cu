/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "graph_core.cuh"
#include "graph_shared.cuh"
#include "utils.hpp"

// TODO: This shouldn't be invoking anything from spatial/knn
#include "../../bbq.cuh"
#include "../ann_utils.cuh"

#include <raft/core/copy.cuh>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/device_memory_resource.hpp>
#include <raft/linalg/map.cuh>
#include <raft/util/bitonic_sort.cuh>
#include <raft/util/cuda_rt_essentials.hpp>

#include <optional>
#include <type_traits>

namespace cuvs::neighbors::cagra::detail::graph {
namespace {

template <class DATA_T, int numElementsPerThread>
__global__ void kern_sort(const DATA_T* const dataset,  // [dataset_chunk_size, dataset_dim]
                          const uint32_t dataset_dim,
                          uint32_t* const knn_graph,  // [graph_chunk_size, graph_degree]
                          const uint32_t graph_size,
                          const uint32_t graph_degree,
                          const cuvs::distance::DistanceType metric)
{
  const uint32_t srcNode = (blockDim.x * blockIdx.x + threadIdx.x) / raft::WarpSize;
  if (srcNode >= graph_size) { return; }

  const uint32_t lane_id = threadIdx.x % raft::WarpSize;

  float my_keys[numElementsPerThread];
  uint32_t my_vals[numElementsPerThread];

  // Compute distance from a src node to its neighbors
  for (int k = 0; k < graph_degree; k++) {
    const uint32_t dstNode = knn_graph[k + static_cast<uint64_t>(graph_degree) * srcNode];
    float dist             = 0;
    float norm2_dst        = 0;
    if (metric == cuvs::distance::DistanceType::InnerProduct ||
        metric == cuvs::distance::DistanceType::CosineExpanded) {
      for (int d = lane_id; d < dataset_dim; d += raft::WarpSize) {
        auto elem_b = cuvs::spatial::knn::detail::utils::mapping<float>{}(
          dataset[d + static_cast<uint64_t>(dataset_dim) * dstNode]);
        dist -= cuvs::spatial::knn::detail::utils::mapping<float>{}(
                  dataset[d + static_cast<uint64_t>(dataset_dim) * srcNode]) *
                elem_b;

        if (metric == cuvs::distance::DistanceType::CosineExpanded) {
          norm2_dst += elem_b * elem_b;
        }
      }
    } else if (metric == cuvs::distance::DistanceType::L2Expanded) {
      for (int d = lane_id; d < dataset_dim; d += raft::WarpSize) {
        float diff = cuvs::spatial::knn::detail::utils::mapping<float>{}(
                       dataset[d + static_cast<uint64_t>(dataset_dim) * srcNode]) -
                     cuvs::spatial::knn::detail::utils::mapping<float>{}(
                       dataset[d + static_cast<uint64_t>(dataset_dim) * dstNode]);
        dist += diff * diff;
      }
    } else if (metric == cuvs::distance::DistanceType::L1) {
      for (int d = lane_id; d < dataset_dim; d += raft::WarpSize) {
        float diff = cuvs::spatial::knn::detail::utils::mapping<float>{}(
                       dataset[d + static_cast<uint64_t>(dataset_dim) * srcNode]) -
                     cuvs::spatial::knn::detail::utils::mapping<float>{}(
                       dataset[d + static_cast<uint64_t>(dataset_dim) * dstNode]);
        dist += raft::abs(diff);
      }
    } else if (metric == cuvs::distance::DistanceType::BitwiseHamming) {
      if constexpr (std::is_integral_v<DATA_T>) {
        for (int d = lane_id; d < dataset_dim; d += raft::WarpSize) {
          dist += __popc(
            static_cast<uint32_t>(dataset[d + static_cast<uint64_t>(dataset_dim) * srcNode] ^
                                  dataset[d + static_cast<uint64_t>(dataset_dim) * dstNode]) &
            0xffu);
        }
      }
    }
    dist += __shfl_xor_sync(0xffffffff, dist, 1);
    dist += __shfl_xor_sync(0xffffffff, dist, 2);
    dist += __shfl_xor_sync(0xffffffff, dist, 4);
    dist += __shfl_xor_sync(0xffffffff, dist, 8);
    dist += __shfl_xor_sync(0xffffffff, dist, 16);

    if (metric == cuvs::distance::DistanceType::CosineExpanded) {
      norm2_dst += __shfl_xor_sync(0xffffffff, norm2_dst, 1);
      norm2_dst += __shfl_xor_sync(0xffffffff, norm2_dst, 2);
      norm2_dst += __shfl_xor_sync(0xffffffff, norm2_dst, 4);
      norm2_dst += __shfl_xor_sync(0xffffffff, norm2_dst, 8);
      norm2_dst += __shfl_xor_sync(0xffffffff, norm2_dst, 16);
      if (lane_id == (k % raft::WarpSize)) { dist /= sqrt(norm2_dst); }
    }

    if (lane_id == (k % raft::WarpSize)) {
      my_keys[k / raft::WarpSize] = dist;
      my_vals[k / raft::WarpSize] = dstNode;
    }
  }
  for (int k = graph_degree; k < raft::WarpSize * numElementsPerThread; k++) {
    if (lane_id == k % raft::WarpSize) {
      my_keys[k / raft::WarpSize] = utils::get_max_value<float>();
      my_vals[k / raft::WarpSize] = utils::get_max_value<uint32_t>();
    }
  }

  raft::util::bitonic<numElementsPerThread>(true).sort(my_keys, my_vals);

  for (int i = 0; i < numElementsPerThread; i++) {
    const int k = i * raft::WarpSize + lane_id;
    if (k < graph_degree) {
      knn_graph[k + (static_cast<uint64_t>(graph_degree) * srcNode)] = my_vals[i];
    }
  }
}

constexpr int kMaxSortElementsPerThread = 32;

template <typename DataT>
using sort_kernel_type =
  void (*)(DataT const*, uint32_t, uint32_t*, uint32_t, uint32_t, cuvs::distance::DistanceType);

template <typename DataT>
auto select_sort_kernel(uint32_t degree) -> sort_kernel_type<DataT>
{
  if (degree <= raft::WarpSize * 1) { return kern_sort<DataT, 1>; }
  if (degree <= raft::WarpSize * 2) { return kern_sort<DataT, 2>; }
  if (degree <= raft::WarpSize * 4) { return kern_sort<DataT, 4>; }
  if (degree <= raft::WarpSize * 8) { return kern_sort<DataT, 8>; }
  if (degree <= raft::WarpSize * 16) { return kern_sort<DataT, 16>; }
  if (degree <= kMaxSortDegree) { return kern_sort<DataT, kMaxSortElementsPerThread>; }
  RAFT_FAIL(
    "The degree of input knn graph is too large (%u). It must be equal to or smaller than %lu.",
    degree,
    kMaxSortDegree);
}

template <typename DataT>
void launch_sort_knn_graph_impl(raft::resources const& res,
                                cuvs::distance::DistanceType metric,
                                DataT const* dataset,
                                uint32_t dataset_size,
                                uint32_t dataset_dim,
                                uint32_t* knn_graph,
                                uint32_t graph_degree)
{
  auto kernel = select_sort_kernel<DataT>(graph_degree);

  constexpr uint32_t block_size = 256;
  auto const warps              = block_size / raft::WarpSize;
  auto const blocks             = (dataset_size + warps - 1) / warps;
  kernel<<<blocks, block_size, 0, raft::resource::get_cuda_stream(res)>>>(
    dataset, dataset_dim, knn_graph, dataset_size, graph_degree, metric);
  RAFT_CUDA_TRY(cudaGetLastError());
}

template <typename DataT, typename IdxT>
using device_bbq_quantizer_view_t =
  cuvs::preprocessing::quantize::bbq::device_bbq_quantizer_view<DataT, IdxT>;

/**
 * Distance between two dataset rows in quantized space
 *
 * @param row_norms squared row norms, required for CosineExpanded and unused otherwise.
 */
template <typename DataT>
__device__ __forceinline__ float bbq_row_distance(
  const device_bbq_quantizer_view_t<DataT, int64_t>& quantizer,
  cuvs::distance::DistanceType metric,
  const float* row_norms,
  int64_t row_a,
  int64_t row_b)
{
  namespace bbq        = cuvs::preprocessing::quantize::bbq;
  const float centered = bbq::centered_dot(quantizer, row_a, row_b);
  switch (metric) {
    case cuvs::distance::DistanceType::InnerProduct:
      return -bbq::dot_product(quantizer, centered, row_a, row_b);
    case cuvs::distance::DistanceType::CosineExpanded:
      return bbq::cosine_distance(
        quantizer, centered, row_a, row_b, row_norms[row_a] * row_norms[row_b]);
    // L2SqrtExpanded shares this ordering: sqrt is monotonic and only the ranking is kept.
    default: return bbq::l2_distance(quantizer, centered, row_a, row_b);
  }
}

/**
 * Distance between two dataset rows scored across a document and a query quantizer, for a dataset
 * that carries a second, finer set of codes.
 *
 * @param row_norms_document squared row norms under the document quantizer, CosineExpanded only.
 * @param row_norms_query squared row norms under the query quantizer, CosineExpanded only.
 */
template <typename DataT>
__device__ __forceinline__ float bbq_row_distance(
  const device_bbq_quantizer_view_t<DataT, int64_t>& quantizer_document,
  const device_bbq_quantizer_view_t<DataT, int64_t>& quantizer_query,
  cuvs::distance::DistanceType metric,
  const float* row_norms_document,
  const float* row_norms_query,
  int64_t row_document,
  int64_t row_query)
{
  namespace bbq = cuvs::preprocessing::quantize::bbq;
  const float centered =
    bbq::centered_dot(quantizer_document, quantizer_query, row_document, row_query);
  switch (metric) {
    case cuvs::distance::DistanceType::InnerProduct:
      return -bbq::dot_product(
        quantizer_document, quantizer_query, centered, row_document, row_query);
    case cuvs::distance::DistanceType::CosineExpanded:
      return bbq::cosine_distance(quantizer_document,
                                  quantizer_query,
                                  centered,
                                  row_document,
                                  row_query,
                                  row_norms_document[row_document] * row_norms_query[row_query]);
    // L2SqrtExpanded shares this ordering: sqrt is monotonic and only the ranking is kept.
    default:
      return bbq::l2_distance(
        quantizer_document, quantizer_query, centered, row_document, row_query);
  }
}

/**
 * @param quantizer_query the finer codes to score the second endpoint with; equal to
 *        `quantizer_document` and unused when @p is_asymmetric is false.
 */
template <typename DataT, int numElementsPerThread>
__global__ void kern_sort_bbq(const device_bbq_quantizer_view_t<DataT, int64_t> quantizer_document,
                              const device_bbq_quantizer_view_t<DataT, int64_t> quantizer_query,
                              const bool is_asymmetric,
                              uint32_t* const knn_graph,  // [graph_size, graph_degree]
                              const uint32_t graph_size,
                              const uint32_t graph_degree,
                              const float* const row_norms_document,
                              const float* const row_norms_query,
                              const cuvs::distance::DistanceType metric)
{
  const uint32_t src_node = (blockDim.x * blockIdx.x + threadIdx.x) / raft::WarpSize;
  if (src_node >= graph_size) { return; }

  const uint32_t lane_id = threadIdx.x % raft::WarpSize;

  float my_keys[numElementsPerThread];
  uint32_t my_vals[numElementsPerThread];

  // One quantized distance is a whole-row popcount over packed codes, which does not decompose
  // across lanes the way a dense dot product does. So each lane computes the distances for the
  // neighbors it already owns in the bitonic register layout (element i of lane l holds neighbor
  // i * WarpSize + l) instead of the warp cooperating on one neighbor at a time.
  for (int i = 0; i < numElementsPerThread; i++) {
    const uint32_t k = i * raft::WarpSize + lane_id;
    if (k >= graph_degree) {
      my_keys[i] = utils::get_max_value<float>();
      my_vals[i] = utils::get_max_value<uint32_t>();
      continue;
    }
    const uint32_t dst_node = knn_graph[k + static_cast<uint64_t>(graph_degree) * src_node];
    // nn-descent scored the first endpoint of a pair with the document codes; keep that
    // orientation so the sort ranks by the same distance that built these lists.
    my_keys[i] =
      is_asymmetric
        ? bbq_row_distance(quantizer_document,
                           quantizer_query,
                           metric,
                           row_norms_document,
                           row_norms_query,
                           src_node,
                           dst_node)
        : bbq_row_distance(quantizer_document, metric, row_norms_document, src_node, dst_node);
    my_vals[i] = dst_node;
  }

  raft::util::bitonic<numElementsPerThread>(true).sort(my_keys, my_vals);

  for (int i = 0; i < numElementsPerThread; i++) {
    const uint32_t k = i * raft::WarpSize + lane_id;
    if (k < graph_degree) {
      knn_graph[k + (static_cast<uint64_t>(graph_degree) * src_node)] = my_vals[i];
    }
  }
}

template <typename DataT>
using sort_bbq_kernel_type = void (*)(device_bbq_quantizer_view_t<DataT, int64_t>,
                                      device_bbq_quantizer_view_t<DataT, int64_t>,
                                      bool,
                                      uint32_t*,
                                      uint32_t,
                                      uint32_t,
                                      const float*,
                                      const float*,
                                      cuvs::distance::DistanceType);

template <typename DataT>
auto select_sort_bbq_kernel(uint32_t degree) -> sort_bbq_kernel_type<DataT>
{
  if (degree <= raft::WarpSize * 1) { return kern_sort_bbq<DataT, 1>; }
  if (degree <= raft::WarpSize * 2) { return kern_sort_bbq<DataT, 2>; }
  if (degree <= raft::WarpSize * 4) { return kern_sort_bbq<DataT, 4>; }
  if (degree <= raft::WarpSize * 8) { return kern_sort_bbq<DataT, 8>; }
  if (degree <= raft::WarpSize * 16) { return kern_sort_bbq<DataT, 16>; }
  if (degree <= kMaxSortDegree) { return kern_sort_bbq<DataT, kMaxSortElementsPerThread>; }
  RAFT_FAIL(
    "The degree of input knn graph is too large (%u). It must be equal to or smaller than %lu.",
    degree,
    kMaxSortDegree);
}

template <typename DataT>
auto select_sort_quantizers(cuvs::neighbors::device_bbq_dataset_view<DataT, int64_t> const& dataset)
  -> std::tuple<device_bbq_quantizer_view_t<DataT, int64_t>,
                std::optional<device_bbq_quantizer_view_t<DataT, int64_t>>>
{
  namespace bbq = cuvs::preprocessing::quantize::bbq;
  using quant_t = device_bbq_quantizer_view_t<DataT, int64_t>;
  using tuple_t = std::tuple<quant_t, std::optional<quant_t>>;

  const bool has_single_bit = dataset.has_bit_and_layout(1, bbq::bbq_code_layout::single_bit);
  const bool has_dibit      = dataset.has_bit_and_layout(2, bbq::bbq_code_layout::dibit);
  const bool has_transpose_half_byte =
    dataset.has_bit_and_layout(4, bbq::bbq_code_layout::transpose_half_byte);

  const bool use_asymmetric =
    dataset.quantizers.size() > 1 &&
    ((has_single_bit && has_dibit) || (has_single_bit && has_transpose_half_byte) ||
     (has_dibit && has_transpose_half_byte));
  if (use_asymmetric) {
    return tuple_t{has_single_bit ? dataset.get_quantizer(1, bbq::bbq_code_layout::single_bit)
                                  : dataset.get_quantizer(2, bbq::bbq_code_layout::dibit),
                   has_transpose_half_byte
                     ? dataset.get_quantizer(4, bbq::bbq_code_layout::transpose_half_byte)
                     : dataset.get_quantizer(2, bbq::bbq_code_layout::dibit)};
  }
  return tuple_t{dataset.quantizers[0], std::nullopt};
}

template <typename DataT>
void sort_knn_graph_bbq_impl(raft::resources const& res,
                             cuvs::distance::DistanceType metric,
                             cuvs::neighbors::device_bbq_dataset_view<DataT, int64_t> dataset,
                             raft::host_matrix_view<uint32_t, int64_t, raft::row_major> knn_graph)
{
  namespace bbq = cuvs::preprocessing::quantize::bbq;

  RAFT_EXPECTS(!dataset.quantizers.empty(), "the BBQ dataset holds no quantizer");
  RAFT_EXPECTS(dataset.n_rows() == knn_graph.extent(0),
               "dataset size is expected to have the same number of graph index size");
  RAFT_EXPECTS(metric == cuvs::distance::DistanceType::L2Expanded ||
                 metric == cuvs::distance::DistanceType::L2SqrtExpanded ||
                 metric == cuvs::distance::DistanceType::InnerProduct ||
                 metric == cuvs::distance::DistanceType::CosineExpanded,
               "Unsupported metric. Only L2Expanded, L2SqrtExpanded, InnerProduct and "
               "CosineExpanded are supported for a BBQ-quantized dataset");

  auto const graph_size   = static_cast<uint32_t>(knn_graph.extent(0));
  auto const graph_degree = static_cast<uint32_t>(knn_graph.extent(1));
  auto kernel             = select_sort_bbq_kernel<DataT>(graph_degree);
  auto quantizers         = select_sort_quantizers(dataset);

  const double time_sort_start = cur_time();
  RAFT_LOG_DEBUG("# Sorting kNN Graph on GPUs ");

  auto large_tmp_mr  = raft::resource::get_large_workspace_resource_ref(res);
  auto d_input_graph = raft::make_device_mdarray<uint32_t>(
    res, large_tmp_mr, raft::make_extents<int64_t>(graph_size, graph_degree));
  raft::copy(res, d_input_graph.view(), knn_graph);

  // Cosine needs both endpoints' norms per distance, so materialize them once per row rather than
  // recomputing a whole-row popcount for every neighbor. The two endpoints are measured under
  // different quantizers when scoring asymmetrically, so each then needs its own norms.
  auto row_norms_document         = std::optional<raft::device_vector<float, size_t>>{};
  auto row_norms_query            = std::optional<raft::device_vector<float, size_t>>{};
  const float* norms_document_ptr = nullptr;
  const float* norms_query_ptr    = nullptr;
  if (metric == cuvs::distance::DistanceType::CosineExpanded) {
    row_norms_document = raft::make_device_vector<float, size_t>(res, graph_size);
    raft::linalg::map_offset(
      res, row_norms_document->view(), bbq::bbq_row_norm_op{std::get<0>(quantizers)});
    norms_document_ptr = row_norms_document->data_handle();
    norms_query_ptr    = norms_document_ptr;
    if (std::get<1>(quantizers).has_value()) {
      row_norms_query = raft::make_device_vector<float, size_t>(res, graph_size);
      raft::linalg::map_offset(
        res, row_norms_query->view(), bbq::bbq_row_norm_op{*std::get<1>(quantizers)});
      norms_query_ptr = row_norms_query->data_handle();
    }
  }

  RAFT_LOG_DEBUG(".");
  constexpr uint32_t block_size = 256;
  auto const warps              = block_size / raft::WarpSize;
  auto const blocks             = (graph_size + warps - 1) / warps;
  kernel<<<blocks, block_size, 0, raft::resource::get_cuda_stream(res)>>>(
    std::get<0>(quantizers),
    std::get<1>(quantizers).value_or(std::get<0>(quantizers)),
    std::get<1>(quantizers).has_value(),
    d_input_graph.data_handle(),
    graph_size,
    graph_degree,
    norms_document_ptr,
    norms_query_ptr,
    metric);
  RAFT_CUDA_TRY(cudaGetLastError());
  raft::resource::sync_stream(res);
  RAFT_LOG_DEBUG(".");
  raft::copy(res, knn_graph, raft::make_const_mdspan(d_input_graph.view()));
  RAFT_LOG_DEBUG("\n");

  const double time_sort_end = cur_time();
  RAFT_LOG_DEBUG("# Sorting kNN graph time: %.1lf sec\n", time_sort_end - time_sort_start);
}

}  // namespace

#define CUVS_DEFINE_CAGRA_GRAPH_SORT(DataT)                                      \
  void launch_sort_knn_graph(raft::resources const& res,                         \
                             cuvs::distance::DistanceType metric,                \
                             DataT const* dataset,                               \
                             uint32_t dataset_size,                              \
                             uint32_t dataset_dim,                               \
                             uint32_t* knn_graph,                                \
                             uint32_t graph_degree)                              \
  {                                                                              \
    launch_sort_knn_graph_impl(                                                  \
      res, metric, dataset, dataset_size, dataset_dim, knn_graph, graph_degree); \
  }

CUVS_DEFINE_CAGRA_GRAPH_SORT(float)
CUVS_DEFINE_CAGRA_GRAPH_SORT(half)
CUVS_DEFINE_CAGRA_GRAPH_SORT(int8_t)
CUVS_DEFINE_CAGRA_GRAPH_SORT(uint8_t)

#undef CUVS_DEFINE_CAGRA_GRAPH_SORT

#define CUVS_DEFINE_CAGRA_GRAPH_SORT_BBQ(DataT)                                                 \
  void sort_knn_graph_bbq(raft::resources const& res,                                           \
                          cuvs::distance::DistanceType metric,                                  \
                          cuvs::neighbors::device_bbq_dataset_view<DataT, int64_t> dataset,     \
                          raft::host_matrix_view<uint32_t, int64_t, raft::row_major> knn_graph) \
  {                                                                                             \
    sort_knn_graph_bbq_impl(res, metric, dataset, knn_graph);                                   \
  }

CUVS_DEFINE_CAGRA_GRAPH_SORT_BBQ(float)
CUVS_DEFINE_CAGRA_GRAPH_SORT_BBQ(half)
CUVS_DEFINE_CAGRA_GRAPH_SORT_BBQ(int8_t)
CUVS_DEFINE_CAGRA_GRAPH_SORT_BBQ(uint8_t)

#undef CUVS_DEFINE_CAGRA_GRAPH_SORT_BBQ

void optimize_device_graph(
  raft::resources const& res,
  raft::device_matrix_view<uint32_t, int64_t, raft::row_major> knn_graph,
  raft::device_matrix_view<uint32_t, int64_t, raft::row_major> output_graph,
  bool guarantee_connectivity)
{
  optimize(res, knn_graph, output_graph, guarantee_connectivity);
}

}  // namespace cuvs::neighbors::cagra::detail::graph
