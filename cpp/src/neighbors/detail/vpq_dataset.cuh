/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuvs/neighbors/common.hpp>
#include <cuvs/preprocessing/quantize/pq.hpp>

#include "../../cluster/kmeans_balanced.cuh"
#include "../../preprocessing/quantize/detail/pq_codepacking.cuh"  // pq_bits-bitfield
#include "ann_utils.cuh"                                           // utils::mapping etc

#include <cuvs/cluster/kmeans.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/cuda_stream_pool.hpp>
#include <raft/core/resource/device_properties.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/map.cuh>
#include <raft/stats/mean.cuh>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/integer_utils.hpp>
#include <raft/util/pow2_utils.cuh>
#include <raft/util/vectorized.cuh>
#include <raft/util/warp_primitives.cuh>

#include <cuda_pipeline_primitives.h>

// A temporary stub till https://github.com/NVIDIA/raft/pull/2077 is re-merged
namespace cuvs::util {

/**
 * Subsample the dataset to create a training set.
 *
 * @tparam DatasetT a row-major mdspan or mdarray (device or host)
 *
 * @param res raft handle
 * @param dataset input row-major mdspan or mdarray (device or host)
 * @param n_samples the size of the output mdarray
 *
 * @return a newly allocated subset of the dataset.
 */
template <typename DatasetT>
auto subsample(raft::resources const& res,
               const DatasetT& dataset,
               typename DatasetT::index_type n_samples)
  -> raft::device_matrix<typename DatasetT::value_type, typename DatasetT::index_type>
{
  using value_type = typename DatasetT::value_type;
  using index_type = typename DatasetT::index_type;
  static_assert(std::is_same_v<typename DatasetT::layout_type, raft::row_major>,
                "Only row-major layout is supported at the moment");
  RAFT_EXPECTS(n_samples <= dataset.extent(0),
               "The number of samples must be smaller than the number of input rows in the current "
               "implementation.");
  size_t dim            = dataset.extent(1);
  size_t trainset_ratio = dataset.extent(0) / n_samples;
  auto result = raft::make_device_matrix<value_type, index_type>(res, n_samples, dataset.extent(1));

  raft::copy_matrix(result.data_handle(),
                    dim,
                    dataset.data_handle(),
                    dim * trainset_ratio,
                    dim,
                    n_samples,
                    raft::resource::get_cuda_stream(res));
  return result;
}

}  // namespace cuvs::util

namespace cuvs::neighbors::detail {

template <typename MathT, typename IdxT>
void train_pq_centers(
  const raft::resources& res,
  const cuvs::preprocessing::quantize::pq::kmeans_params_variant& kmeans_params,
  const raft::device_matrix_view<const MathT, IdxT, raft::row_major> pq_trainset_view,
  const raft::device_matrix_view<MathT, uint32_t, raft::row_major> pq_centers_view,
  raft::device_vector_view<uint32_t, IdxT> sub_labels_view,
  raft::device_vector_view<uint32_t, IdxT> pq_cluster_sizes_view)
{
  std::visit(
    [&](auto const& base_kmeans_params) {
      using KP = std::decay_t<decltype(base_kmeans_params)>;
      if constexpr (std::is_same_v<KP, cuvs::cluster::kmeans::balanced_params>) {
        auto bal_params   = base_kmeans_params;
        bal_params.metric = cuvs::distance::DistanceType::L2Expanded;
        cuvs::cluster::kmeans_balanced::helpers::build_clusters<
          MathT,
          MathT,
          IdxT,
          uint32_t,
          uint32_t,
          cuvs::spatial::knn::detail::utils::mapping<MathT>>(
          res,
          bal_params,
          pq_trainset_view,
          pq_centers_view,
          sub_labels_view,
          pq_cluster_sizes_view,
          cuvs::spatial::knn::detail::utils::mapping<MathT>{});
      } else {
        auto classic_params       = base_kmeans_params;
        classic_params.n_clusters = pq_centers_view.extent(0);
        classic_params.metric     = cuvs::distance::DistanceType::L2Expanded;
        std::optional<raft::device_vector_view<const MathT, IdxT>> sample_weight = std::nullopt;
        MathT inertia;
        IdxT n_iter;
        cuvs::cluster::kmeans::fit(res,
                                   classic_params,
                                   pq_trainset_view,
                                   sample_weight,
                                   pq_centers_view,
                                   raft::make_host_scalar_view<MathT>(&inertia),
                                   raft::make_host_scalar_view<IdxT>(&n_iter));
      }
    },
    kmeans_params);
}

template <typename DatasetT>
auto fill_missing_params_heuristics(const vpq_params& params, const DatasetT& dataset) -> vpq_params
{
  vpq_params r  = params;
  double n_rows = dataset.extent(0);
  size_t dim    = dataset.extent(1);
  if (r.pq_dim == 0) { r.pq_dim = raft::div_rounding_up_safe(dim, size_t{4}); }
  if (r.pq_bits == 0) { r.pq_bits = 8; }
  if (r.vq_n_centers == 0) { r.vq_n_centers = raft::round_up_safe<uint32_t>(std::sqrt(n_rows), 8); }
  if (r.vq_kmeans_trainset_fraction == 0) {
    double vq_trainset_size       = 100.0 * r.vq_n_centers;
    r.vq_kmeans_trainset_fraction = std::min(1.0, vq_trainset_size / n_rows);
  }
  if (r.pq_kmeans_trainset_fraction == 0) {
    // NB: we'll have actually `pq_dim` times more samples than this
    //     (because the dataset is reinterpreted as `[n_rows * pq_dim, pq_len]`)
    double pq_trainset_size       = 1000.0 * (1u << r.pq_bits);
    r.pq_kmeans_trainset_fraction = std::min(1.0, pq_trainset_size / n_rows);
  }
  r.max_train_points_per_pq_code = params.max_train_points_per_pq_code;
  return r;
}

template <typename T, typename DatasetT>
auto transform_data(const raft::resources& res, DatasetT dataset)
  -> raft::device_mdarray<T, typename DatasetT::extents_type, typename DatasetT::layout_type>
{
  using index_type       = typename DatasetT::index_type;
  using extents_type     = typename DatasetT::extents_type;
  using layout_type      = typename DatasetT::layout_type;
  using out_mdarray_type = raft::device_mdarray<T, extents_type, layout_type>;
  if constexpr (std::is_same_v<out_mdarray_type, std::decay<DatasetT>>) { return dataset; }

  auto result = raft::make_device_mdarray<T, index_type, layout_type>(res, dataset.extents());

  raft::linalg::map(res,
                    result.view(),
                    cuvs::spatial::knn::detail::utils::mapping<T>{},
                    raft::make_const_mdspan(dataset.view()));

  return result;
}

/** Fix the internal indexing type to avoid integer underflows/overflows */
using ix_t = int64_t;

template <typename MathT, typename DatasetT>
auto train_vq(const raft::resources& res, const vpq_params& params, const DatasetT& dataset)
  -> raft::device_matrix<MathT, uint32_t, raft::row_major>
{
  using kmeans_in_type    = typename DatasetT::value_type;
  const ix_t n_rows       = dataset.extent(0);
  const ix_t vq_n_centers = params.vq_n_centers;
  const ix_t dim          = dataset.extent(1);
  const ix_t n_rows_train = std::min<ix_t>(n_rows * params.vq_kmeans_trainset_fraction,
                                           params.max_train_points_per_vq_cluster * vq_n_centers);

  // Subsample the dataset and transform into the required type if necessary
  auto vq_trainset = util::subsample(res, dataset, n_rows_train);
  auto vq_centers =
    raft::make_device_matrix<MathT, uint32_t, raft::row_major>(res, vq_n_centers, dim);

  auto vq_trainset_view = raft::make_device_matrix_view<const kmeans_in_type, ix_t>(
    vq_trainset.data_handle(), n_rows_train, dim);

  if (vq_n_centers == 1) {
    auto vq_centers_view =
      raft::make_device_vector_view<MathT, ix_t>(vq_centers.data_handle(), dim);
    raft::stats::mean(res, vq_trainset_view, vq_centers_view);
  } else {
    auto vq_centers_view =
      raft::make_device_matrix_view<MathT, ix_t>(vq_centers.data_handle(), vq_n_centers, dim);
    cuvs::cluster::kmeans::balanced_params kmeans_params;
    kmeans_params.n_iters = params.kmeans_n_iters;
    kmeans_params.metric  = cuvs::distance::DistanceType::L2Expanded;
    cuvs::cluster::kmeans::fit(res, kmeans_params, vq_trainset_view, vq_centers_view);
  }
  return vq_centers;
}

template <typename LabelT, typename DatasetT, typename VqCentersT>
auto predict_vq(const raft::resources& res,
                const DatasetT& dataset,
                const VqCentersT& vq_centers,
                raft::device_vector_view<LabelT, typename DatasetT::index_type> vq_labels)
{
  using kmeans_data_type = typename DatasetT::value_type;
  using kmeans_math_type = typename VqCentersT::value_type;
  using index_type       = typename DatasetT::index_type;
  using label_type       = LabelT;

  cuvs::cluster::kmeans::balanced_params kmeans_params;
  kmeans_params.metric = cuvs::distance::DistanceType::L2Expanded;

  auto vq_centers_view = raft::make_device_matrix_view<const kmeans_math_type, index_type>(
    vq_centers.data_handle(), vq_centers.extent(0), vq_centers.extent(1));

  auto vq_dataset_view = raft::make_device_matrix_view<const kmeans_data_type, index_type>(
    dataset.data_handle(), dataset.extent(0), dataset.extent(1));

  cuvs::cluster::kmeans::predict(res, kmeans_params, vq_dataset_view, vq_centers_view, vq_labels);
}

template <typename MathT, typename DatasetT>
auto train_pq(const raft::resources& res,
              const cuvs::preprocessing::quantize::pq::params& params,
              const DatasetT& dataset,
              const raft::device_matrix_view<const MathT, uint32_t, raft::row_major> vq_centers)
  -> raft::device_matrix<MathT, uint32_t, raft::row_major>
{
  const ix_t n_rows       = dataset.extent(0);
  const ix_t dim          = dataset.extent(1);
  const ix_t pq_dim       = params.pq_dim;
  const ix_t pq_bits      = params.pq_bits;
  const ix_t pq_n_centers = ix_t{1} << pq_bits;
  const ix_t pq_len       = raft::div_rounding_up_safe(dim, pq_dim);
  const ix_t n_rows_train =
    std::min<ix_t>(n_rows, params.max_train_points_per_pq_code * pq_n_centers);
  RAFT_EXPECTS(
    n_rows_train >= pq_n_centers,
    "The number of training samples must be greater than or equal to the number of PQ centers");

  // Subsample the dataset and transform into the required type if necessary
  auto pq_trainset = transform_data<MathT>(res, util::subsample(res, dataset, n_rows_train));

  // Subtract VQ centers
  if (!vq_centers.empty()) {
    auto vq_labels = raft::make_device_vector<uint32_t, ix_t>(res, pq_trainset.extent(0));
    predict_vq<uint32_t>(res, pq_trainset, vq_centers, vq_labels.view());
    using index_type = typename DatasetT::index_type;
    raft::linalg::map_offset(
      res,
      pq_trainset.view(),
      [labels = vq_labels.view(), centers = vq_centers, dim] __device__(index_type off, MathT x) {
        index_type i = off / dim;
        index_type j = off % dim;
        return x - centers(labels(i), j);
      },
      raft::make_const_mdspan(pq_trainset.view()));
  }

  auto pq_centers =
    raft::make_device_matrix<MathT, uint32_t, raft::row_major>(res, pq_n_centers, pq_len);
  auto pq_trainset_view = raft::make_device_matrix_view<const MathT, ix_t>(
    pq_trainset.data_handle(), n_rows_train * pq_dim, pq_len);
  auto sub_labels       = raft::make_device_vector<uint32_t, ix_t>(res, pq_trainset_view.extent(0));
  auto pq_cluster_sizes = raft::make_device_vector<uint32_t, ix_t>(res, pq_centers.extent(0));
  train_pq_centers<MathT, ix_t>(res,
                                params.kmeans_params,
                                pq_trainset_view,
                                pq_centers.view(),
                                sub_labels.view(),
                                pq_cluster_sizes.view());

  return pq_centers;
}

template <uint32_t SubWarpSize,
          typename CodeT,
          typename DataT,
          typename MathT,
          typename IdxT,
          typename LabelT>
__device__ auto compute_code(
  raft::device_matrix_view<const DataT, IdxT, raft::row_major> dataset,
  raft::device_matrix_view<const MathT, uint32_t, raft::row_major> vq_centers,
  raft::device_matrix_view<const MathT, uint32_t, raft::row_major> pq_centers_smem,
  raft::device_matrix_view<const MathT, uint32_t, raft::row_major> pq_centers,
  IdxT i,
  uint32_t j,
  LabelT vq_label) -> CodeT
{
  static_assert(std::is_same_v<CodeT, uint8_t> || std::is_same_v<CodeT, uint16_t>,
                "CodeT must be uint8_t or uint16_t");
  auto data_mapping      = cuvs::spatial::knn::detail::utils::mapping<MathT>{};
  const uint32_t lane_id = raft::Pow2<SubWarpSize>::mod(raft::laneId());

  const uint32_t pq_book_size = pq_centers.extent(0);
  const uint32_t pq_len       = pq_centers.extent(1);
  float min_dist              = std::numeric_limits<float>::infinity();
  CodeT code                  = 0;
  // calculate the distance for each PQ cluster, find the minimum for each thread
  uint32_t l = lane_id;
  if (!pq_centers_smem.empty()) {
    for (; l < pq_centers_smem.extent(0); l += SubWarpSize) {
      // NB: the L2 quantifiers on residuals are always trained on L2 metric.
      float d = 0.0f;
      for (uint32_t k = 0; k < pq_len; k++) {
        auto jk = j * pq_len + k;
        auto x  = data_mapping(dataset(i, jk));
        if (!vq_centers.empty()) { x -= vq_centers(vq_label, jk); }
        auto t = x - pq_centers_smem(l, k);
        d += t * t;
      }
      if (d < min_dist) {
        min_dist = d;
        code     = CodeT(l);
      }
    }
  }
  // if there are more rows than the shared memory size, compute the distance for the remaining rows
  for (; l < pq_book_size; l += SubWarpSize) {
    // NB: the L2 quantifiers on residuals are always trained on L2 metric.
    float d = 0.0f;
    for (uint32_t k = 0; k < pq_len; k++) {
      auto jk = j * pq_len + k;
      auto x  = data_mapping(dataset(i, jk));
      if (!vq_centers.empty()) { x -= vq_centers(vq_label, jk); }
      auto t = x - pq_centers(l, k);
      d += t * t;
    }
    if (d < min_dist) {
      min_dist = d;
      code     = CodeT(l);
    }
  }
  // reduce among threads
#pragma unroll
  for (uint32_t stride = SubWarpSize >> 1; stride > 0; stride >>= 1) {
    const auto other_dist = raft::shfl_xor(min_dist, stride, SubWarpSize);
    const auto other_code = raft::shfl_xor(code, stride, SubWarpSize);
    if (other_dist < min_dist) {
      min_dist = other_dist;
      code     = other_code;
    }
  }
  return code;
}

template <uint32_t BlockSize,
          uint32_t SubWarpSize,
          typename CodeT,
          typename DataT,
          typename MathT,
          typename IdxT,
          typename LabelT>
__launch_bounds__(BlockSize) RAFT_KERNEL process_and_fill_codes_kernel(
  raft::device_matrix_view<uint8_t, IdxT, raft::row_major> out_codes,
  raft::device_matrix_view<const DataT, IdxT, raft::row_major> dataset,
  raft::device_matrix_view<const MathT, uint32_t, raft::row_major> pq_centers,
  raft::device_matrix_view<const MathT, uint32_t, raft::row_major> vq_centers,
  raft::device_vector_view<const LabelT, IdxT, raft::row_major> vq_labels,
  const uint32_t rows_in_shared_memory,
  const uint32_t pq_bits,
  const bool inline_vq_labels = false)
{
  extern __shared__ __align__(256) MathT pq_centers_smem[];
  using subwarp_align = raft::Pow2<SubWarpSize>;
  const IdxT row_ix   = subwarp_align::div(IdxT{threadIdx.x} + IdxT{BlockSize} * IdxT{blockIdx.x});
  if (row_ix >= out_codes.extent(0)) { return; }

  const uint32_t pq_dim = raft::div_rounding_up_unsafe(dataset.extent(1), pq_centers.extent(1));

  // Copy the pq_centers into shared memory for faster processing
  for (uint32_t i = threadIdx.x; i < rows_in_shared_memory * pq_centers.extent(1);
       i += blockDim.x) {
    pq_centers_smem[i] = pq_centers.data_handle()[i];
  }
  __syncthreads();

  auto pq_centers_smem_view = raft::make_device_matrix_view<const MathT, uint32_t, raft::row_major>(
    pq_centers_smem, rows_in_shared_memory, pq_centers.extent(1));
  const uint32_t lane_id = subwarp_align::mod(threadIdx.x);
  const LabelT vq_label  = !vq_labels.empty() ? vq_labels(row_ix) : 0;
  auto* out_codes_ptr    = &out_codes(row_ix, 0);

  // write label
  if (!vq_centers.empty() && inline_vq_labels) {
    auto* out_label_ptr = reinterpret_cast<LabelT*>(out_codes_ptr);
    if (lane_id == 0) { *out_label_ptr = vq_label; }
    out_codes_ptr += sizeof(LabelT);
  }

  cuvs::preprocessing::quantize::pq::detail::bitfield_view_t code_view{out_codes_ptr, pq_bits};
  for (uint32_t j = 0; j < pq_dim; j++) {
    // find PQ label
    CodeT code = compute_code<SubWarpSize, CodeT>(
      dataset, vq_centers, pq_centers_smem_view, pq_centers, row_ix, j, vq_label);
    // TODO: this writes in global memory one byte per warp, which is very slow.
    //  It's better to keep the codes in the shared memory or registers and dump them at once.
    if (lane_id == 0) { code_view[j] = code; }
  }
}

/**
 * Note: `inline_vq_labels` should only be used for CAGRA-Q compatibility or internal use-cases.
 * Otherwise, vq_labels should be preferred.
 * Issue: https://github.com/NVIDIA/cuvs/issues/1722
 */
template <typename MathT, typename IdxT, typename DatasetT>
void process_and_fill_codes(
  const raft::resources& res,
  const vpq_params& params,
  const DatasetT& dataset,
  raft::device_matrix_view<const MathT, uint32_t, raft::row_major> pq_centers,
  raft::device_matrix_view<const MathT, uint32_t, raft::row_major> vq_centers,
  raft::device_vector_view<uint32_t, IdxT> vq_labels,
  raft::device_matrix_view<uint8_t, IdxT, raft::row_major> codes,
  bool inline_vq_labels = false)
{
  using data_t     = typename DatasetT::value_type;
  using cdataset_t = device_vpq_dataset<MathT, IdxT>;
  using label_t    = uint32_t;

  const ix_t n_rows       = dataset.extent(0);
  const ix_t dim          = dataset.extent(1);
  const ix_t pq_dim       = params.pq_dim;
  const ix_t pq_bits      = params.pq_bits;
  const ix_t pq_n_centers = ix_t{1} << pq_bits;
  ix_t codes_rowlen       = 0;
  if (!vq_centers.empty() && inline_vq_labels) {
    // NB: codes must be aligned at least to sizeof(label_t) to be able to read labels.
    codes_rowlen = sizeof(label_t) *
                   (1 + raft::div_rounding_up_safe<ix_t>(pq_dim * pq_bits, 8 * sizeof(label_t)));
  } else {
    codes_rowlen = raft::div_rounding_up_safe<ix_t>(pq_dim * pq_bits, 8);
  }
  RAFT_EXPECTS(codes.extent(0) == n_rows,
               "Codes matrix must have the same number of rows as the input dataset");
  RAFT_EXPECTS(codes.extent(1) == codes_rowlen,
               "Codes matrix must have the same number of columns as the input dataset");

  auto stream = raft::resource::get_cuda_stream(res);

  // TODO: with scaling workspace we could choose the batch size dynamically
  constexpr ix_t kReasonableMaxBatchSize = 65536;
  constexpr ix_t kBlockSize              = 256;
  constexpr ix_t kMaxSharedMemorySize    = 16384;
  const ix_t rows_in_shared_memory       = std::min<ix_t>(
    pq_centers.extent(0), kMaxSharedMemorySize / (sizeof(MathT) * pq_centers.extent(1)));
  const ix_t sharedMemorySize = rows_in_shared_memory * pq_centers.extent(1) * sizeof(MathT);
  const ix_t threads_per_vec  = std::min<ix_t>(raft::WarpSize, pq_n_centers);
  dim3 threads(kBlockSize, 1, 1);
  ix_t max_batch_size = std::min<ix_t>(n_rows, kReasonableMaxBatchSize);
  auto kernel         = [](uint32_t pq_bits) {
    if (pq_bits == 4) {
      return process_and_fill_codes_kernel<kBlockSize, 16, uint8_t, data_t, MathT, IdxT, label_t>;
    } else if (pq_bits <= 8) {
      return process_and_fill_codes_kernel<kBlockSize,
                                                   raft::WarpSize,
                                                   uint8_t,
                                                   data_t,
                                                   MathT,
                                                   IdxT,
                                                   label_t>;
    } else if (pq_bits <= 16) {
      return process_and_fill_codes_kernel<kBlockSize,
                                                   raft::WarpSize,
                                                   uint16_t,
                                                   data_t,
                                                   MathT,
                                                   IdxT,
                                                   label_t>;
    } else {
      RAFT_FAIL("Invalid pq_bits (%u), the value must be within [4, 16]", pq_bits);
    }
  }(pq_bits);
  bool need_copy_to_device =
    cuvs::spatial::knn::detail::utils::check_pointer_residency(dataset.data_handle()) ==
    cuvs::spatial::knn::detail::utils::pointer_residency::host_only;
  bool need_batching = n_rows > kReasonableMaxBatchSize;
  auto launch_work   = [&](auto& dataset_view, auto& labels_view, auto& codes_view) {
    if (inline_vq_labels || (!vq_labels.empty() && !vq_centers.empty())) {
      predict_vq<label_t>(res, dataset_view, vq_centers, labels_view);
    }
    dim3 blocks(
      raft::div_rounding_up_safe<ix_t>(dataset_view.extent(0), kBlockSize / threads_per_vec), 1, 1);
    kernel<<<blocks, threads, sharedMemorySize, stream.get()>>>(
      codes_view,
      dataset_view,
      pq_centers,
      vq_centers,
      raft::make_const_mdspan(labels_view),
      rows_in_shared_memory,
      pq_bits,
      inline_vq_labels);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  };
  auto batch_labels = raft::make_device_vector<label_t, IdxT>(res, 0);
  if (!need_batching && !need_copy_to_device) {
    // No batching needed, launch the kernel directly
    auto dataset_view = raft::make_device_matrix_view(dataset.data_handle(), n_rows, dim);
    auto labels_view  = raft::make_device_vector_view<label_t, IdxT>(nullptr, 0);
    if (inline_vq_labels) {
      batch_labels = raft::make_device_vector<label_t, IdxT>(res, dataset_view.extent(0));
      labels_view  = batch_labels.view();
    } else if (!vq_labels.empty() && !vq_centers.empty()) {
      labels_view = vq_labels;
    }
    launch_work(dataset_view, labels_view, codes);
    return;
  }

  auto _vpq_batches_codes = cuvs::spatial::knn::detail::utils::make_batch_load_iterator<data_t>(
    res,
    dataset.data_handle(),
    static_cast<ix_t>(n_rows),
    static_cast<ix_t>(dim),
    static_cast<size_t>(max_batch_size),
    stream,
    rmm::mr::get_current_device_resource_ref());
  for (const auto& batch : _vpq_batches_codes) {
    auto batch_view        = raft::make_device_matrix_view(batch.data(), ix_t(batch.size()), dim);
    auto batch_labels_view = raft::make_device_vector_view<label_t, IdxT>(nullptr, 0);
    if (inline_vq_labels) {
      batch_labels      = raft::make_device_vector<label_t, IdxT>(res, batch.size());
      batch_labels_view = batch_labels.view();
    } else if (!vq_labels.empty() && !vq_centers.empty()) {
      batch_labels_view = raft::make_device_vector_view<label_t, IdxT>(
        vq_labels.data_handle() + batch.offset(), batch.size());
    }
    auto batch_codes_view = raft::make_device_matrix_view<uint8_t, IdxT>(
      codes.data_handle() + batch.offset() * codes_rowlen, batch.size(), codes_rowlen);
    launch_work(batch_view, batch_labels_view, batch_codes_view);
  }
}

// Helper for operations using vectorized loads of raft::TxN_t
template <typename MathT, int VectorSize>
struct vec_op : raft::TxN_t<MathT, VectorSize> {
  DI float sum_squares() const
  {
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < VectorSize; ++i) {
      sum += this->val.data[i] * this->val.data[i];
    }
    return sum;
  }

  DI void reverse_sub(const vec_op<MathT, VectorSize>& a)
  {
#pragma unroll
    for (int i = 0; i < VectorSize; ++i) {
      this->val.data[i] = a.val.data[i] - this->val.data[i];
    }
  }
};

// Helper: compute distances to 4 centers and find minimum
template <uint32_t SubWarpSize, typename MathT, typename CodeT>
__device__ __forceinline__ void compute_4centers_and_update(
  MathT& min_dist, CodeT& code, MathT d0, MathT d1, MathT d2, MathT d3, uint32_t l)
{
  MathT best_dist_01 = fminf(d0, d1);
  MathT best_dist_23 = fminf(d2, d3);
  MathT best_dist    = fminf(best_dist_01, best_dist_23);

  const uint32_t idx_01   = (d1 < d0) ? (l + SubWarpSize) : l;
  const uint32_t idx_23   = (d3 < d2) ? (l + 3 * SubWarpSize) : (l + 2 * SubWarpSize);
  const uint32_t best_idx = (best_dist_23 < best_dist_01) ? idx_23 : idx_01;

  if (best_dist < min_dist) {
    min_dist = best_dist;
    code     = CodeT(best_idx);
  }
}

// Helper: process 4 centers
template <uint32_t SubWarpSize, typename MathT, typename GetXFunc>
__device__ __forceinline__ void process_4centers_vec(MathT& d0,
                                                     MathT& d1,
                                                     MathT& d2,
                                                     MathT& d3,
                                                     const MathT* __restrict__ centers_ptr,
                                                     uint32_t l,
                                                     const uint32_t pq_len,
                                                     GetXFunc get_x_func)
{
  uint32_t k = 0;
  // Only use vectorized loads if pq_len is a multiple of 2 or 4 (prevents misaligned accesses)
  const bool pq_len_div_4 = (pq_len & 3u) == 0u;
  const bool pq_len_div_2 = (pq_len & 1u) == 0u;
  if (pq_len_div_4) {
    for (; k + 3 < pq_len; k += 4) {
      vec_op<MathT, 4> x_vec, c0, c1, c2, c3;
      x_vec.val.data[0] = get_x_func(k);
      x_vec.val.data[1] = get_x_func(k + 1);
      x_vec.val.data[2] = get_x_func(k + 2);
      x_vec.val.data[3] = get_x_func(k + 3);
      c0.load(centers_ptr, l * pq_len + k);
      c1.load(centers_ptr, (l + SubWarpSize) * pq_len + k);
      c2.load(centers_ptr, (l + 2 * SubWarpSize) * pq_len + k);
      c3.load(centers_ptr, (l + 3 * SubWarpSize) * pq_len + k);
      c0.reverse_sub(x_vec);
      c1.reverse_sub(x_vec);
      c2.reverse_sub(x_vec);
      c3.reverse_sub(x_vec);
      d0 += c0.sum_squares();
      d1 += c1.sum_squares();
      d2 += c2.sum_squares();
      d3 += c3.sum_squares();
    }
  }
  if (pq_len_div_2) {
    for (; k + 1 < pq_len; k += 2) {
      vec_op<MathT, 2> x_vec, c0, c1, c2, c3;
      x_vec.val.data[0] = get_x_func(k);
      x_vec.val.data[1] = get_x_func(k + 1);
      c0.load(centers_ptr, l * pq_len + k);
      c1.load(centers_ptr, (l + SubWarpSize) * pq_len + k);
      c2.load(centers_ptr, (l + 2 * SubWarpSize) * pq_len + k);
      c3.load(centers_ptr, (l + 3 * SubWarpSize) * pq_len + k);
      c0.reverse_sub(x_vec);
      c1.reverse_sub(x_vec);
      c2.reverse_sub(x_vec);
      c3.reverse_sub(x_vec);
      d0 += c0.sum_squares();
      d1 += c1.sum_squares();
      d2 += c2.sum_squares();
      d3 += c3.sum_squares();
    }
  }

  // Tail loop
  for (; k < pq_len; k++) {
    MathT x  = get_x_func(k);
    MathT c0 = x - centers_ptr[l * pq_len + k];
    MathT c1 = x - centers_ptr[(l + SubWarpSize) * pq_len + k];
    MathT c2 = x - centers_ptr[(l + 2 * SubWarpSize) * pq_len + k];
    MathT c3 = x - centers_ptr[(l + 3 * SubWarpSize) * pq_len + k];

    d0 += c0 * c0;
    d1 += c1 * c1;
    d2 += c2 * c2;
    d3 += c3 * c3;
  }
}

template <uint32_t SubWarpSize, typename MathT, typename CodeT, typename GetXFunc>
__device__ __forceinline__ void process_all_centers(MathT& min_dist,
                                                    CodeT& code,
                                                    const MathT* __restrict__ centers_ptr,
                                                    uint32_t pq_len,
                                                    GetXFunc get_data_value,
                                                    uint32_t pq_book_size)
{
  // Process 4 centers at a time
  uint32_t l = raft::Pow2<SubWarpSize>::mod(raft::laneId());
  for (; l + 3 * SubWarpSize < pq_book_size; l += 4 * SubWarpSize) {
    MathT d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
    process_4centers_vec<SubWarpSize, MathT>(
      d0, d1, d2, d3, centers_ptr, l, pq_len, get_data_value);
    compute_4centers_and_update<SubWarpSize, MathT, CodeT>(min_dist, code, d0, d1, d2, d3, l);
  }

  // Tail: process 1 center at a time
  for (; l < pq_book_size; l += SubWarpSize) {
    MathT d0 = 0.0f;
    for (uint32_t k = 0; k < pq_len; k++) {
      MathT diff = get_data_value(k) - centers_ptr[l * pq_len + k];
      d0 += diff * diff;
    }
    if (d0 < min_dist) {
      min_dist = d0;
      code     = CodeT(l);
    }
  }
}

template <uint32_t SubWarpSize,
          typename CodeT,
          typename DataT,
          typename MathT,
          typename IdxT,
          typename LabelT>
__device__ auto compute_code_subspaces(
  raft::device_matrix_view<const DataT, IdxT, raft::row_major> dataset,
  raft::device_matrix_view<const MathT, uint32_t, raft::row_major> pq_centers,
  raft::device_matrix_view<const MathT, uint32_t, raft::row_major> vq_centers,
  MathT* smem_dataset,
  IdxT row_ix,
  uint32_t j,
  const bool use_shared_memory,
  LabelT vq_label) -> CodeT
{
  static_assert(std::is_same_v<CodeT, uint8_t> || std::is_same_v<CodeT, uint16_t>,
                "CodeT must be uint8_t or uint16_t");
  auto data_mapping      = cuvs::spatial::knn::detail::utils::mapping<MathT>{};
  const uint32_t lane_id = raft::Pow2<SubWarpSize>::mod(raft::laneId());

  const uint32_t pq_book_size           = pq_centers.extent(0);
  const uint32_t pq_len                 = pq_centers.extent(1);
  const MathT* __restrict__ centers_ptr = pq_centers.data_handle();
  const DataT* __restrict__ dataset_ptr = dataset.data_handle() + row_ix * dataset.extent(1);
  MathT min_dist                        = std::numeric_limits<MathT>::infinity();
  CodeT code                            = 0;

  if (pq_len <= SubWarpSize) {
    // Each thread loads one dataset element and will share to others using shuffle
    MathT my_dataset_val = 0.0f;
    if (lane_id < pq_len) {
      auto jk = j * pq_len + lane_id;
      auto x  = data_mapping(dataset_ptr[jk]);
      if (!vq_centers.empty()) { x -= vq_centers(vq_label, jk); }
      my_dataset_val = x;
    }

    const uint32_t subwarp_idx = raft::Pow2<SubWarpSize>::div(raft::laneId());
    const uint32_t subwarp_mask =
      (SubWarpSize == 32) ? 0xffffffffu : (0xffffu << (subwarp_idx * SubWarpSize));

    auto get_x_shuffle = [&](uint32_t k) {
      return raft::shfl(my_dataset_val, k, SubWarpSize, subwarp_mask);
    };

    process_all_centers<SubWarpSize, MathT, CodeT>(
      min_dist, code, centers_ptr, pq_len, get_x_shuffle, pq_book_size);
  } else if (use_shared_memory) {
    // Cooperatively load dataset into shared memory if pq_len > SubWarpSize
    for (uint32_t l = lane_id; l < pq_len; l += SubWarpSize) {
      auto jk = j * pq_len + l;
      auto x  = data_mapping(dataset_ptr[jk]);
      if (!vq_centers.empty()) { x -= vq_centers(vq_label, jk); }
      smem_dataset[l] = x;
    }
    __syncwarp();
    auto get_data_value_smem = [&](uint32_t k) { return smem_dataset[k]; };
    process_all_centers<SubWarpSize, MathT, CodeT>(
      min_dist, code, centers_ptr, pq_len, get_data_value_smem, pq_book_size);
  } else {
    auto get_data_value = [&](uint32_t k) {
      auto x = data_mapping(dataset_ptr[j * pq_len + k]);
      if (!vq_centers.empty()) { x -= vq_centers(vq_label, j * pq_len + k); }
      return x;
    };
    process_all_centers<SubWarpSize, MathT, CodeT>(
      min_dist, code, centers_ptr, pq_len, get_data_value, pq_book_size);
  }

  // Reduce among threads
#pragma unroll
  for (uint32_t stride = SubWarpSize >> 1; stride > 0; stride >>= 1) {
    const auto other_dist = raft::shfl_xor(min_dist, stride, SubWarpSize);
    const auto other_code = raft::shfl_xor(code, stride, SubWarpSize);
    if (other_dist < min_dist) {
      min_dist = other_dist;
      code     = other_code;
    }
  }
  return code;
}

template <uint32_t BlockSize,
          uint32_t SubWarpSize,
          typename CodeT,
          typename DataT,
          typename MathT,
          typename IdxT,
          typename LabelT>
__launch_bounds__(BlockSize) RAFT_KERNEL process_and_fill_codes_subspaces_kernel(
  raft::device_matrix_view<uint8_t, IdxT, raft::row_major> out_codes,
  raft::device_matrix_view<const DataT, IdxT, raft::row_major> dataset,
  raft::device_matrix_view<const MathT, uint32_t, raft::row_major> pq_centers,
  raft::device_matrix_view<const MathT, uint32_t, raft::row_major> vq_centers,
  raft::device_vector_view<const LabelT, IdxT, raft::row_major> vq_labels,
  const uint32_t pq_bits,
  const bool use_shared_memory)
{
  extern __shared__ __align__(256) uint8_t smem_buffer[];
  using subwarp_align = raft::Pow2<SubWarpSize>;
  const IdxT row_ix   = subwarp_align::div(IdxT{threadIdx.x} + IdxT{BlockSize} * IdxT{blockIdx.x});
  if (row_ix >= out_codes.extent(0)) { return; }

  const uint32_t subwarp_id = subwarp_align::div(threadIdx.x);
  MathT* smem_dataset   = reinterpret_cast<MathT*>(smem_buffer) + subwarp_id * pq_centers.extent(1);
  const uint32_t pq_dim = raft::div_rounding_up_unsafe(dataset.extent(1), pq_centers.extent(1));

  const uint32_t lane_id = subwarp_align::mod(threadIdx.x);
  const LabelT vq_label  = !vq_labels.empty() ? vq_labels(row_ix) : 0;

  cuvs::preprocessing::quantize::pq::detail::bitfield_view_t code_view{&out_codes(row_ix, 0),
                                                                       pq_bits};
  for (uint32_t j = 0; j < pq_dim; j++) {
    // find PQ label
    uint32_t subspace_offset = j * pq_centers.extent(1) * (1 << pq_bits);
    auto pq_subspace_view    = raft::make_device_matrix_view(
      pq_centers.data_handle() + subspace_offset, (uint32_t)(1 << pq_bits), pq_centers.extent(1));
    CodeT code = compute_code_subspaces<SubWarpSize, CodeT>(
      dataset, pq_subspace_view, vq_centers, smem_dataset, row_ix, j, use_shared_memory, vq_label);
    if (lane_id == 0) { code_view[j] = code; }
  }
}

/** Rows and subspaces encoded by one 256-thread block of the TF32 subspace encoder. */
constexpr uint32_t kTf32EncodeRows  = 256;
constexpr uint32_t kTf32EncodeChunk = 32;  // kTf32EncodeChunk * pq_bits is a whole byte count

__device__ __forceinline__ uint32_t to_tf32(float x)
{
  uint32_t r;
  asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x));
  return r;
}

/** The TF32 rounding of `x`, or with `lo` the TF32 rounding of what that leaves out. */
__device__ __forceinline__ uint32_t tf32_part(float x, bool lo)
{
  const uint32_t hi = to_tf32(x);
  return lo ? to_tf32(x - __uint_as_float(hi)) : hi;
}

/** Segment of GEMM K index k in the TF32 encoder: 0 for k < pq_len, 1 for k < 2 * pq_len, ... */
__device__ __forceinline__ uint32_t tf32_segment(uint32_t k, uint32_t pq_len)
{
  return uint32_t{k >= pq_len} + uint32_t{k >= 2 * pq_len};
}

/**
 * Packs every subspace codebook as the m16n8k8 B-fragments of the TF32 encoder followed by the
 * FP32 center norms: subspace j takes (k_len + 1) * n_centers words, with K index k of center c at
 * [k / 4][c / 8][(c % 8) * 4 + k % 4]. The K axis holds n_terms TF32 parts of -2c (see below),
 * then 1.
 */
template <typename MathT>
RAFT_KERNEL pack_tf32_codebooks_kernel(
  uint32_t* packed,
  raft::device_matrix_view<const MathT, uint32_t, raft::row_major> pq_centers,
  uint32_t n_centers,
  uint32_t k_len,
  uint32_t n_terms)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= pq_centers.extent(0)) { return; }
  const uint32_t pq_len = pq_centers.extent(1);
  const uint32_t c      = row % n_centers;
  uint32_t* out         = packed + size_t{row / n_centers} * (k_len + 1) * n_centers;
  float norm            = 0.0f;
  for (uint32_t k = 0; k < pq_len; k++) {
    const float v = pq_centers(row, k);
    norm += v * v;
  }
  for (uint32_t k = 0; k < k_len; k++) {
    const uint32_t s = tf32_segment(k, pq_len);
    uint32_t b       = 0;
    if (k < n_terms * pq_len) {
      b = tf32_part(-2.0f * static_cast<float>(pq_centers(row, k - s * pq_len)), s == 2);
    } else if (k == n_terms * pq_len) {
      b = __float_as_uint(1.0f);
    }
    out[(k / 4) * n_centers * 4 + (c / 8) * 32 + (c % 8) * 4 + k % 4] = b;
  }
  out[k_len * n_centers + c] = __float_as_uint(norm);
#endif
}

/**
 * Tensor-core (TF32, sm_80+) variant of `process_and_fill_codes_subspaces_kernel`.
 *
 * A block encodes kRows rows over a chunk of kChunk subspaces, copying the codebooks packed by
 * `pack_tf32_codebooks_kernel` to shared memory one subspace ahead (double-buffered dynamic shared
 * memory of 2 packed codebooks). The MMA accumulator starts from the FP32 norms ||c||^2, so that
 * the MMA yields ||c||^2 - 2<x, c> + ||x||^2. The GEMM K axis lays out
 *   A = [x | ||x||^2],  B = [-2c | 1]                                   (Split = false), or
 *   A = [x_hi | x_lo | x_hi | ||x||^2],  B = [-2c_hi | -2c_hi | -2c_lo | 1]  (Split = true),
 * where the split form recovers FP32-level accuracy of <x, c> (3xTF32) inside the K padding. The
 * ||x||^2 slot is dropped when it does not fit the padding; it is a per-row constant, so the argmin
 * does not depend on it, but it keeps the distances small and thus the error of the argmin tag,
 * which carries the center index in the low 8 mantissa bits (at most 2^-15 relative).
 */
template <uint32_t NTiles,
          uint32_t KSteps,
          bool Split,
          typename DataT,
          typename MathT,
          typename IdxT,
          typename LabelT>
// Past 2 k-steps, the shared memory of the double-buffered codebooks caps occupancy at 3 blocks.
__launch_bounds__(256, KSteps <= 2 ? 4 : 3) RAFT_KERNEL
  process_and_fill_codes_subspaces_tf32_kernel(
    raft::device_matrix_view<uint8_t, IdxT, raft::row_major> out_codes,
    raft::device_matrix_view<const DataT, IdxT, raft::row_major> dataset,
    const uint32_t* packed_centers,
    const uint32_t pq_len,
    raft::device_matrix_view<const MathT, uint32_t, raft::row_major> vq_centers,
    raft::device_vector_view<const LabelT, IdxT, raft::row_major> vq_labels,
    const uint32_t pq_bits)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  constexpr uint32_t kRows    = kTf32EncodeRows;
  constexpr uint32_t kChunk   = kTf32EncodeChunk;
  constexpr uint32_t kMTiles  = kRows / (256 / raft::WarpSize) / 16;  // 16-row MMA tiles per warp
  constexpr uint32_t kCenters = NTiles * 8;
  constexpr uint32_t kK       = KSteps * 8;
  constexpr uint32_t kTerms   = Split ? 3 : 1;
  constexpr uint32_t kBook    = (kK + 1) * kCenters;  // words of a packed subspace codebook
  extern __shared__ uint4 tf32_books[];               // two packed codebooks, double-buffered
  __shared__ uint8_t codes[kRows][kChunk];

  auto data_mapping        = cuvs::spatial::knn::detail::utils::mapping<float>{};
  const uint32_t lane      = raft::laneId();
  const uint32_t group     = lane / 4;
  const uint32_t tig       = lane % 4;
  const uint32_t pq_dim    = raft::div_rounding_up_unsafe(dataset.extent(1), pq_len);
  const uint32_t norm_slot = kTerms * pq_len;  // K index of ||x||^2; unused if >= kK
  const IdxT row0          = IdxT{blockIdx.x} * kRows;
  const uint32_t j0        = blockIdx.y * kChunk;
  const uint32_t j1        = std::min(j0 + kChunk, pq_dim);

  for (uint32_t i = threadIdx.x; i < sizeof(codes); i += blockDim.x) {
    (&codes[0][0])[i] = 0;
  }
  auto stage = [&](uint32_t j) {
    const auto* src = reinterpret_cast<const uint4*>(packed_centers + size_t{j} * kBook);
    uint4* dst      = tf32_books + (j - j0) % 2 * (kBook / 4);
    for (uint32_t i = threadIdx.x; i < kBook / 4; i += blockDim.x) {
      __pipeline_memcpy_async(dst + i, src + i, sizeof(uint4));
    }
    __pipeline_commit();
  };
  stage(j0);

  // This thread reads rows (m, h) of its warp's MMA tiles; lanes with tig == 0 own their codes.
  auto local_row = [&](uint32_t m, uint32_t h) {
    return threadIdx.x / raft::WarpSize * kMTiles * 16 + m * 16 + h * 8 + group;
  };
  LabelT labels[kMTiles][2];
#pragma unroll
  for (uint32_t m = 0; m < kMTiles; m++) {
#pragma unroll
    for (uint32_t h = 0; h < 2; h++) {
      const IdxT row = row0 + local_row(m, h);
      labels[m][h]   = (!vq_labels.empty() && row < dataset.extent(0)) ? vq_labels(row) : 0;
    }
  }

  for (uint32_t j = j0; j < j1; j++) {
    const auto* book   = reinterpret_cast<const uint32_t*>(tf32_books + (j - j0) % 2 * (kBook / 4));
    const float* norms = reinterpret_cast<const float*>(book + kK * kCenters);

    // A-fragment q of m16n8k8 holds row group + 8 * (q & 1), column tig + 4 * (q >> 1).
    float x[kMTiles][KSteps][4];
    float x_norm[kMTiles][2] = {};
#pragma unroll
    for (uint32_t m = 0; m < kMTiles; m++) {
#pragma unroll
      for (uint32_t ks = 0; ks < KSteps; ks++) {
#pragma unroll
        for (uint32_t q = 0; q < 4; q++) {
          const uint32_t h = q & 1;
          const IdxT row   = row0 + local_row(m, h);
          const uint32_t k = ks * 8 + tig + 4 * (q >> 1);
          const uint32_t s = tf32_segment(k, pq_len);
          x[m][ks][q]      = 0.0f;
          if (row < dataset.extent(0) && k < kTerms * pq_len) {
            const uint32_t col = j * pq_len + k - s * pq_len;
            x[m][ks][q]        = data_mapping(dataset(row, col));
            if (!vq_centers.empty()) { x[m][ks][q] -= vq_centers(labels[m][h], col); }
            if (s == 0) { x_norm[m][h] += x[m][ks][q] * x[m][ks][q]; }
          }
        }
      }
    }
    __pipeline_wait_prior(0);
    __syncthreads();  // codebook j is staged and nobody reads codebook j - 1 anymore
    if (j + 1 < j1) { stage(j + 1); }

    uint32_t a[kMTiles][KSteps][4];
#pragma unroll
    for (uint32_t m = 0; m < kMTiles; m++) {
#pragma unroll
      for (uint32_t h = 0; h < 2; h++) {
        x_norm[m][h] += raft::shfl_xor(x_norm[m][h], 1);
        x_norm[m][h] += raft::shfl_xor(x_norm[m][h], 2);
      }
#pragma unroll
      for (uint32_t ks = 0; ks < KSteps; ks++) {
#pragma unroll
        for (uint32_t q = 0; q < 4; q++) {
          const uint32_t k = ks * 8 + tig + 4 * (q >> 1);
          a[m][ks][q]      = k == norm_slot
                               ? to_tf32(x_norm[m][q & 1])
                               : tf32_part(x[m][ks][q], Split && tf32_segment(k, pq_len) == 1);
        }
      }
    }

    float best[kMTiles][2];
#pragma unroll
    for (uint32_t m = 0; m < kMTiles; m++) {
      best[m][0] = best[m][1] = std::numeric_limits<float>::infinity();
    }
#pragma unroll
    for (uint32_t t = 0; t < NTiles; t++) {
      // Accumulator element q holds center t * 8 + 2 * tig + (q & 1).
      const float2 norm = *reinterpret_cast<const float2*>(norms + t * 8 + 2 * tig);
      uint32_t b[KSteps][2];
#pragma unroll
      for (uint32_t ks = 0; ks < KSteps; ks++) {
        b[ks][0] = book[((2 * ks) * NTiles + t) * raft::WarpSize + lane];
        b[ks][1] = book[((2 * ks + 1) * NTiles + t) * raft::WarpSize + lane];
      }
#pragma unroll
      for (uint32_t m = 0; m < kMTiles; m++) {
        float d[4] = {norm.x, norm.y, norm.x, norm.y};
#pragma unroll
        for (uint32_t ks = 0; ks < KSteps; ks++) {
          asm(
            "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
            : "r"(a[m][ks][0]),
              "r"(a[m][ks][1]),
              "r"(a[m][ks][2]),
              "r"(a[m][ks][3]),
              "r"(b[ks][0]),
              "r"(b[ks][1]));
        }
#pragma unroll
        for (uint32_t q = 0; q < 4; q++) {
          const float tagged =
            __uint_as_float((__float_as_uint(d[q]) & ~0xFFu) | (t * 8 + (q & 1)));
          best[m][q / 2] = fminf(best[m][q / 2], tagged);
        }
      }
    }
#pragma unroll
    for (uint32_t m = 0; m < kMTiles; m++) {
#pragma unroll
      for (uint32_t h = 0; h < 2; h++) {
        const uint32_t bits = __float_as_uint(best[m][h]);
        float v             = __uint_as_float((bits & ~0xFFu) | ((bits & 0xFFu) + 2 * tig));
        v                   = fminf(v, raft::shfl_xor(v, 1));
        v                   = fminf(v, raft::shfl_xor(v, 2));
        if (tig == 0) {
          cuvs::preprocessing::quantize::pq::detail::bitfield_view_t code_view{
            &codes[local_row(m, h)][0], pq_bits};
          code_view[j - j0] = static_cast<uint8_t>(__float_as_uint(v) & 0xFFu);
        }
      }
    }
  }
  __syncthreads();

  const uint32_t n_bytes = raft::div_rounding_up_unsafe((j1 - j0) * pq_bits, 8u);
  const uint32_t byte0   = j0 * pq_bits / 8;
  for (uint32_t i = threadIdx.x; i < kRows * n_bytes; i += blockDim.x) {
    const IdxT row = row0 + i / n_bytes;
    if (row < dataset.extent(0)) {
      out_codes(row, byte0 + i % n_bytes) = codes[i / n_bytes][i % n_bytes];
    }
  }
#endif
}

template <typename MathT, typename IdxT, typename DatasetT>
void process_and_fill_codes_subspaces(
  const raft::resources& res,
  const vpq_params& params,
  const DatasetT& dataset,
  raft::device_matrix_view<const MathT, uint32_t, raft::row_major> pq_centers,
  raft::device_matrix_view<const MathT, uint32_t, raft::row_major> vq_centers,
  raft::device_vector_view<uint32_t, IdxT, raft::row_major> vq_labels,
  raft::device_matrix_view<uint8_t, IdxT, raft::row_major> codes)
{
  using data_t     = typename DatasetT::value_type;
  using cdataset_t = device_vpq_dataset<MathT, IdxT>;
  using label_t    = uint32_t;

  const ix_t n_rows       = dataset.extent(0);
  const ix_t dim          = dataset.extent(1);
  const ix_t pq_dim       = params.pq_dim;
  const ix_t pq_bits      = params.pq_bits;
  const ix_t pq_n_centers = ix_t{1} << pq_bits;
  const ix_t codes_rowlen = raft::div_rounding_up_safe<ix_t>(pq_dim * pq_bits, 8);
  RAFT_EXPECTS(codes.extent(0) == n_rows,
               "Codes matrix must have the same number of rows as the input dataset");
  RAFT_EXPECTS(codes.extent(1) == codes_rowlen,
               "Codes matrix must have the same number of columns as the input dataset");

  auto stream = raft::resource::get_cuda_stream(res);

  constexpr ix_t kBlockSize              = 512;
  constexpr ix_t kReasonableMaxBatchSize = 131072;  // 128K rows per batch
  constexpr ix_t kMaxSharedMemorySize    = 16384;
  const ix_t threads_per_vec             = std::min<ix_t>(raft::WarpSize, pq_n_centers);
  const uint32_t num_subwarps            = kBlockSize / threads_per_vec;
  const uint32_t pq_len                  = pq_centers.extent(1);

  uint32_t shared_memory_size =
    pq_len <= threads_per_vec ? 0 : num_subwarps * pq_len * sizeof(MathT);
  if (shared_memory_size > kMaxSharedMemorySize) {
    // Extreme case: pq_len is too large to fit in shared memory
    shared_memory_size = 0;
  }

  dim3 threads(kBlockSize, 1, 1);
  auto kernel = [](uint32_t pq_bits) {
    if (pq_bits == 4) {
      return process_and_fill_codes_subspaces_kernel<kBlockSize,
                                                     16,
                                                     uint8_t,
                                                     data_t,
                                                     MathT,
                                                     IdxT,
                                                     label_t>;
    } else if (pq_bits <= 8) {
      return process_and_fill_codes_subspaces_kernel<kBlockSize,
                                                     raft::WarpSize,
                                                     uint8_t,
                                                     data_t,
                                                     MathT,
                                                     IdxT,
                                                     label_t>;
    } else if (pq_bits <= 16) {
      return process_and_fill_codes_subspaces_kernel<kBlockSize,
                                                     raft::WarpSize,
                                                     uint16_t,
                                                     data_t,
                                                     MathT,
                                                     IdxT,
                                                     label_t>;
    } else {
      RAFT_FAIL("Invalid pq_bits (%u), the value must be within [4, 16]", pq_bits);
    }
  }(pq_bits);

  // ponytail: the TF32 encoder covers pq_bits <= 8 (the center index must fit the 8 tag bits) and
  // either pq_len <= 10 (3xTF32 within 4 MMA k-steps) or 16 <= pq_len <= 32 (plain TF32, which, as
  // in the k-means 1-NN, mislabels near-ties below 16 features). Other shapes and pre-sm_80 GPUs
  // keep the SIMT kernel; 11 <= pq_len < 16 would need 3xTF32 over 5-6 k-steps.
  const bool split     = 3 * pq_len <= 32;
  const uint32_t n_k   = (split ? 3 : 1) * pq_len;
  const uint32_t k_len = raft::round_up_safe<uint32_t>(n_k, 8);
  const bool use_tf32  = pq_bits <= 8 && (split || (pq_len >= 16 && pq_len <= 32)) &&
                        raft::resource::get_device_properties(res).major >= 8;
  auto tf32_kernel = [k_len, split](uint32_t pq_bits) {
    auto with_ksteps = [k_len, split](auto n_tiles) {
      constexpr uint32_t kNTiles = decltype(n_tiles)::value;

      auto pick = [](auto k_steps, auto is_split) {
        return process_and_fill_codes_subspaces_tf32_kernel<kNTiles,
                                                            decltype(k_steps)::value,
                                                            decltype(is_split)::value,
                                                            data_t,
                                                            MathT,
                                                            IdxT,
                                                            label_t>;
      };
      using std::integral_constant;
      switch (split ? k_len / 8 : k_len / 8 + 4) {
        case 1: return pick(integral_constant<uint32_t, 1>{}, std::true_type{});
        case 2: return pick(integral_constant<uint32_t, 2>{}, std::true_type{});
        case 3: return pick(integral_constant<uint32_t, 3>{}, std::true_type{});
        case 4: return pick(integral_constant<uint32_t, 4>{}, std::true_type{});
        case 6: return pick(integral_constant<uint32_t, 2>{}, std::false_type{});
        case 7: return pick(integral_constant<uint32_t, 3>{}, std::false_type{});
        case 8: return pick(integral_constant<uint32_t, 4>{}, std::false_type{});
        default: return pick(integral_constant<uint32_t, 4>{}, std::false_type{});
      }
    };
    switch (pq_bits) {
      case 4: return with_ksteps(std::integral_constant<uint32_t, 2>{});
      case 5: return with_ksteps(std::integral_constant<uint32_t, 4>{});
      case 6: return with_ksteps(std::integral_constant<uint32_t, 8>{});
      case 7: return with_ksteps(std::integral_constant<uint32_t, 16>{});
      case 8: return with_ksteps(std::integral_constant<uint32_t, 32>{});
      default: return with_ksteps(std::integral_constant<uint32_t, 32>{});
    }
  }(pq_bits);
  auto packed_centers = raft::make_device_vector<uint32_t, uint32_t>(
    res, use_tf32 ? pq_dim * (k_len + 1) * pq_n_centers : 0);
  const size_t tf32_smem = 2 * (k_len + 1) * pq_n_centers * sizeof(uint32_t);
  if (use_tf32) {
    RAFT_CUDA_TRY(cudaFuncSetAttribute(
      tf32_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(tf32_smem)));
    pack_tf32_codebooks_kernel<<<raft::div_rounding_up_safe<uint32_t>(pq_centers.extent(0), 256),
                                 256,
                                 0,
                                 stream.get()>>>(
      packed_centers.data_handle(), pq_centers, pq_n_centers, k_len, n_k / pq_len);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }

  ix_t max_batch_size = std::min<ix_t>(n_rows, kReasonableMaxBatchSize);
  auto copy_stream    = raft::resource::get_cuda_stream(res);  // Using the main stream by default
  bool enable_prefetch_stream = false;
  bool has_cuda_stream_pool_resource =
    res.has_resource_factory(raft::resource::resource_type::CUDA_STREAM_POOL) &&
    raft::resource::get_stream_pool_size(res) >= 1;
  bool need_copy_to_device =
    cuvs::spatial::knn::detail::utils::check_pointer_residency(dataset.data_handle()) ==
    cuvs::spatial::knn::detail::utils::pointer_residency::host_only;
  bool need_batching = n_rows > kReasonableMaxBatchSize;
  auto launch_work   = [&](auto& dataset_view, auto& labels_view, auto& codes_view) {
    if (!vq_labels.empty() && !vq_centers.empty()) {
      predict_vq<label_t>(res, dataset_view, vq_centers, labels_view);
    }
    if (use_tf32) {
      dim3 blocks(raft::div_rounding_up_safe<ix_t>(dataset_view.extent(0), kTf32EncodeRows),
                  raft::div_rounding_up_safe<ix_t>(pq_dim, kTf32EncodeChunk),
                  1);
      tf32_kernel<<<blocks, 256, tf32_smem, stream.get()>>>(codes_view,
                                                            dataset_view,
                                                            packed_centers.data_handle(),
                                                            pq_len,
                                                            vq_centers,
                                                            raft::make_const_mdspan(labels_view),
                                                            pq_bits);
    } else {
      dim3 blocks(
        raft::div_rounding_up_safe<ix_t>(dataset_view.extent(0), kBlockSize / threads_per_vec),
        1,
        1);
      kernel<<<blocks, threads, shared_memory_size, stream.get()>>>(
        codes_view,
        dataset_view,
        pq_centers,
        vq_centers,
        raft::make_const_mdspan(labels_view),
        pq_bits,
        shared_memory_size > 0);
    }
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  };
  if (!need_batching && !need_copy_to_device) {
    // No batching and no copy to device needed, launch the kernel directly
    auto dataset_view = raft::make_device_matrix_view(dataset.data_handle(), n_rows, dim);
    launch_work(dataset_view, vq_labels, codes);
    return;
  }
  if (has_cuda_stream_pool_resource && need_copy_to_device) {
    enable_prefetch_stream = true;
    copy_stream            = raft::resource::get_stream_from_stream_pool(res);
  }
  auto vec_batches = cuvs::spatial::knn::detail::utils::make_batch_load_iterator<data_t>(
    res,
    dataset.data_handle(),
    static_cast<ix_t>(n_rows),
    static_cast<ix_t>(dim),
    static_cast<size_t>(max_batch_size),
    copy_stream,
    raft::resource::get_workspace_resource_ref(res),
    enable_prefetch_stream);
  vec_batches.prefetch_next_batch();
  for (const auto& batch : vec_batches) {
    auto batch_view   = raft::make_device_matrix_view(batch.data(), ix_t(batch.size()), dim);
    auto batch_labels = raft::make_device_vector_view<label_t, IdxT>(nullptr, 0);
    if (!vq_labels.empty() && !vq_centers.empty()) {
      batch_labels = raft::make_device_vector_view<label_t, IdxT>(
        vq_labels.data_handle() + batch.offset(), batch.size());
    }
    auto batch_codes_view = raft::make_device_matrix_view<uint8_t, IdxT>(
      codes.data_handle() + batch.offset() * codes_rowlen, batch.size(), codes_rowlen);
    launch_work(batch_view, batch_labels, batch_codes_view);
    if (enable_prefetch_stream) {
      vec_batches.prefetch_next_batch();
      raft::resource::sync_stream(res);
    }
  }
}
}  // namespace cuvs::neighbors::detail
