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

#include <mma.h>

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

/** Threads of a block of the TF32 subspace encoder. */
constexpr uint32_t kTf32EncodeThreads = 256;
constexpr uint32_t kTf32EncodeWarps   = kTf32EncodeThreads / raft::WarpSize;
/** Subspaces encoded by a block: their codes fill whole bytes for any pq_bits. */
constexpr uint32_t kTf32EncodeChunk = 32;
/** Padding of the shared-memory rows, in floats: avoids bank conflicts, keeps wmma's alignment. */
constexpr uint32_t kTf32Pad = 4;
/** Row stride of the distance tiles, in floats. */
constexpr uint32_t kTf32TileLd = 16 + kTf32Pad;
/**
 * Row stride of the matrices of the TF32 encoder GEMM: the TF32 parts of every feature and the two
 * norm columns, padded to the k of the wmma tile, plus kTf32Pad.
 */
inline uint32_t tf32_encode_ld(uint32_t pq_len, bool split)
{
  return raft::round_up_safe((split ? 3u : 1u) * pq_len + 2u, 8u) + kTf32Pad;
}

/** Number of 8-wide k-steps of the TF32 encoder GEMM, from 1 to 5 (plain TF32 of 32 features). */
inline uint32_t tf32_encode_k_steps(uint32_t ld) { return (ld - kTf32Pad) / 8; }

/**
 * Tiles of 16 rows encoded by each warp of the TF32 encoder. Every fragment of B that a warp loads
 * serves all its row tiles, which pays off for the 128 or 256 centers of the split form; with
 * fewer centers or the plain form, the extra registers cost more than they save.
 */
inline uint32_t tf32_encode_row_tiles(uint32_t pq_bits, bool split)
{
  return split && pq_bits >= 7 ? 2 : 1;
}

/** Rows encoded by a block of the TF32 encoder. */
constexpr uint32_t tf32_encode_rows(uint32_t row_tiles)
{
  return kTf32EncodeWarps * row_tiles * 16;
}

/** Dynamic shared memory of `process_and_fill_codes_subspaces_tf32_kernel`. */
inline size_t tf32_encode_smem_bytes(uint32_t n_centers, uint32_t ld, uint32_t row_tiles)
{
  const uint32_t rows = tf32_encode_rows(row_tiles);
  return ((rows + n_centers) * ld + kTf32EncodeWarps * 16 * kTf32TileLd) * sizeof(float) +
         rows * kTf32EncodeChunk;
}

/** v_hi (lo = false) or v_lo (lo = true), the TF32 parts of v = v_hi + v_lo. */
RAFT_DEVICE_INLINE_FUNCTION float tf32_part(float v, bool lo)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  const float hi = nvcuda::wmma::__float_to_tf32(v);
  return lo ? nvcuda::wmma::__float_to_tf32(v - hi) : hi;
#else
  return lo ? 0.0f : v;  // the TF32 encoder runs on sm_80+ only
#endif
}

/** Column col of matrix B of `process_and_fill_codes_subspaces_tf32_kernel`, for center c. */
template <typename MathT>
__device__ float tf32_codebook_element(
  raft::device_matrix_view<const MathT, uint32_t, raft::row_major> pq_centers,
  uint32_t c,
  uint32_t col,
  bool split)
{
  const uint32_t pq_len   = pq_centers.extent(1);
  const uint32_t norm_col = (split ? 3 : 1) * pq_len;
  if (col < norm_col) {
    const float b = -2.0f * static_cast<float>(pq_centers(c, col % pq_len));
    return tf32_part(b, col / pq_len == 2);
  }
  if (col < norm_col + 2) {
    float norm = 0.0f;
    for (uint32_t k = 0; k < pq_len; k++) {
      const auto v = static_cast<float>(pq_centers(c, k));
      norm += v * v;
    }
    return tf32_part(norm, col == norm_col + 1);
  }
  return 0.0f;
}

/**
 * Tensor-core (TF32, sm_80+) variant of `process_and_fill_codes_subspaces_kernel`, on the model of
 * nn-descent's `local_join_kernel_wmma`.
 *
 * A block encodes tf32_encode_rows(RowTiles) rows over a chunk of kTf32EncodeChunk subspaces. For
 * every subspace, the block stages its rows of x (minus their VQ center) and the codebook in shared
 * memory, then each warp computes the distances from its RowTiles x 16 rows to all the centers with
 * wmma, 16 centers at a time, as
 *
 *   ||x - c||^2 - ||x||^2 = ||c||^2 - 2 <x, c>,
 *
 * dropping ||x||^2, which is the same for all the centers, and keeps the minimum of every row.
 *
 * TF32 keeps 10 of the 23 mantissa bits. With `split`, the encoder recovers FP32 accuracy (3xTF32)
 * by laying out the K axis of the GEMM as
 *
 *   A = [x_hi | x_lo | x_hi | 1    | 1   ],
 *   B = [b_hi | b_hi | b_lo | n_hi | n_lo],  b = -2 c,  n = ||c||^2,
 *
 * where v_hi is the TF32 rounding of v and v_lo that of v - v_hi, so that A.B is the distance up to
 * the small x_lo * b_lo. Without `split`, A = [x | 1 | 1] and B = [b | n_hi | n_lo]. K is
 * zero-padded to a multiple of 8.
 *
 * `codebooks` holds B for every subspace, row c of subspace j at row j * 2^pq_bits + c, in the
 * layout of the shared memory (`tf32_codebook_element`). KSteps = K / 8 is a template parameter so
 * that the k-step loops unroll and the fragments of A stay in registers.
 */
template <uint32_t KSteps,
          uint32_t RowTiles,
          typename DataT,
          typename MathT,
          typename IdxT,
          typename LabelT>
__launch_bounds__(kTf32EncodeThreads) RAFT_KERNEL process_and_fill_codes_subspaces_tf32_kernel(
  raft::device_matrix_view<uint8_t, IdxT, raft::row_major> out_codes,
  raft::device_matrix_view<const DataT, IdxT, raft::row_major> dataset,
  raft::device_matrix_view<const float, uint32_t, raft::row_major> codebooks,
  raft::device_matrix_view<const MathT, uint32_t, raft::row_major> vq_centers,
  raft::device_vector_view<const LabelT, IdxT, raft::row_major> vq_labels,
  const uint32_t pq_bits,
  const uint32_t pq_len,
  const bool split)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  using namespace nvcuda;
  constexpr uint32_t kRows = tf32_encode_rows(RowTiles);
  const uint32_t n_centers = 1u << pq_bits;
  const uint32_t pq_dim    = dataset.extent(1) / pq_len;
  const uint32_t n_parts   = split ? 3 : 1;
  const uint32_t norm_col  = n_parts * pq_len;
  const uint32_t ld        = codebooks.extent(1);
  // The bits that tag a distance with its tile of 16 centers (see the GEMM loop).
  const uint32_t tile_mask = n_centers / 16 - 1;

  extern __shared__ __align__(32) float tf32_smem[];
  float* s_x       = tf32_smem;                   // A: [kRows][ld], row-major
  float* s_centers = s_x + kRows * ld;            // B: [n_centers][ld], i.e. column-major
  float* s_dists   = s_centers + n_centers * ld;  // [kTf32EncodeWarps][16][kTf32TileLd]
  auto* s_codes    = reinterpret_cast<uint8_t*>(s_dists + kTf32EncodeWarps * 16 * kTf32TileLd);

  // Staging writes the first norm_col columns of A only: the ones and the K padding stay.
  for (uint32_t i = threadIdx.x; i < kRows * ld; i += blockDim.x) {
    s_x[i] = i % ld == norm_col || i % ld == norm_col + 1 ? 1.0f : 0.0f;
  }
  for (uint32_t i = threadIdx.x; i < kRows * kTf32EncodeChunk; i += blockDim.x) {
    s_codes[i] = 0;
  }

  auto data_mapping        = cuvs::spatial::knn::detail::utils::mapping<float>{};
  const IdxT row0          = IdxT{blockIdx.x} * kRows;
  const uint32_t j0        = blockIdx.y * kTf32EncodeChunk;
  const uint32_t j1        = std::min(j0 + kTf32EncodeChunk, pq_dim);
  const uint32_t warp      = threadIdx.x / raft::WarpSize;
  const uint32_t warp_row0 = warp * RowTiles * 16;
  float* warp_dists        = s_dists + warp * 16 * kTf32TileLd;
  // Lanes l and l + 16 take row l % 16 of a tile, each scanning 8 of its 16 columns.
  const uint32_t tile_row = raft::laneId() % 16;
  const uint32_t tile_col = raft::laneId() / 16 * 8;

  // Threads kThreadsPerRow * r to kThreadsPerRow * (r + 1) - 1 stage row r of x.
  constexpr uint32_t kThreadsPerRow = kTf32EncodeThreads / kRows;
  static_assert(kThreadsPerRow * kRows == kTf32EncodeThreads);
  const uint32_t x_row    = threadIdx.x / kThreadsPerRow;
  const bool x_row_valid  = row0 + x_row < dataset.extent(0);
  const LabelT x_vq_label = x_row_valid && !vq_labels.empty() ? vq_labels(row0 + x_row) : 0;

  for (uint32_t j = j0; j < j1; j++) {
    __syncthreads();  // all the warps are done with the previous subspace
    for (uint32_t k = threadIdx.x % kThreadsPerRow; k < pq_len && x_row_valid;
         k += kThreadsPerRow) {
      const uint32_t col = j * pq_len + k;
      float x            = data_mapping(dataset(row0 + x_row, col));
      if (!vq_centers.empty()) { x -= static_cast<float>(vq_centers(x_vq_label, col)); }
      for (uint32_t p = 0; p < n_parts; p++) {
        s_x[x_row * ld + p * pq_len + k] = tf32_part(x, p == 1);
      }
    }
    const auto* codebook = reinterpret_cast<const float4*>(&codebooks(j * n_centers, 0));
    for (uint32_t i = threadIdx.x; i < n_centers * ld / 4; i += blockDim.x) {
      reinterpret_cast<float4*>(s_centers)[i] = codebook[i];
    }
    __syncthreads();

    wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major>
      a_frag[RowTiles][KSteps];
#pragma unroll
    for (uint32_t t = 0; t < RowTiles; t++) {
#pragma unroll
      for (uint32_t s = 0; s < KSteps; s++) {
        wmma::load_matrix_sync(a_frag[t][s], s_x + (warp_row0 + t * 16) * ld + s * 8, ld);
      }
    }

    // Every thread keeps, for each element of its accumulator fragments, the smallest distance over
    // the tiles of centers, its low mantissa bits replaced by the tile c0 / 16 (a relative change
    // below 2^-19), so that fminf keeps both. The update is the same for all the elements, so it
    // holds whatever the (unspecified) layout of the fragments.
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> best_dist[RowTiles];
#pragma unroll
    for (uint32_t t = 0; t < RowTiles; t++) {
      wmma::fill_fragment(best_dist[t], std::numeric_limits<float>::infinity());
    }
    for (uint32_t c0 = 0; c0 < n_centers; c0 += 16) {
      wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> b_frag;
      wmma::fragment<wmma::accumulator, 16, 16, 8, float> dist[RowTiles];
#pragma unroll
      for (uint32_t t = 0; t < RowTiles; t++) {
        wmma::fill_fragment(dist[t], 0.0f);
      }
#pragma unroll
      for (uint32_t s = 0; s < KSteps; s++) {
        wmma::load_matrix_sync(b_frag, s_centers + c0 * ld + s * 8, ld);
#pragma unroll
        for (uint32_t t = 0; t < RowTiles; t++) {
          wmma::mma_sync(dist[t], a_frag[t][s], b_frag, dist[t]);
        }
      }
#pragma unroll
      for (uint32_t t = 0; t < RowTiles; t++) {
        for (int i = 0; i < dist[t].num_elements; i++) {
          const uint32_t tagged = (__float_as_uint(dist[t].x[i]) & ~tile_mask) | c0 / 16;
          best_dist[t].x[i]     = fminf(best_dist[t].x[i], __uint_as_float(tagged));
        }
      }
    }

#pragma unroll
    for (uint32_t t = 0; t < RowTiles; t++) {
      // Element (r, col) of the best tile is the distance from row r to its nearest center among
      // the centers c0 + col, tagged with c0 / 16.
      wmma::store_matrix_sync(warp_dists, best_dist[t], kTf32TileLd, wmma::mem_row_major);
      __syncwarp();
      float row_dist    = std::numeric_limits<float>::infinity();
      uint32_t row_code = 0;
      auto keep_min     = [&](float dist, uint32_t code) {
        if (dist < row_dist || (dist == row_dist && code < row_code)) {
          row_dist = dist;
          row_code = code;
        }
      };
      for (uint32_t col = tile_col; col < tile_col + 8; col++) {
        const float dist = warp_dists[tile_row * kTf32TileLd + col];
        keep_min(dist, (__float_as_uint(dist) & tile_mask) * 16 + col);
      }
      // Merge the two halves of the row.
      keep_min(raft::shfl_xor(row_dist, 16), raft::shfl_xor(row_code, 16));
      if (raft::laneId() < 16) {
        uint8_t* row_codes = s_codes + (warp_row0 + t * 16 + tile_row) * kTf32EncodeChunk;
        cuvs::preprocessing::quantize::pq::detail::bitfield_view_t code_view{row_codes, pq_bits};
        code_view[j - j0] = static_cast<uint8_t>(row_code);
      }
      __syncwarp();  // the tile is read before the next row tile overwrites it
    }
  }
  __syncthreads();

  // Copy the codes of the chunk out, row by row.
  const uint32_t n_bytes = raft::div_rounding_up_unsafe((j1 - j0) * pq_bits, 8u);
  const uint32_t byte0   = j0 * pq_bits / 8;
  for (uint32_t i = threadIdx.x; i < kRows * n_bytes; i += blockDim.x) {
    const uint32_t r = i / n_bytes;
    if (row0 + r < out_codes.extent(0)) {
      out_codes(row0 + r, byte0 + i % n_bytes) = s_codes[r * kTf32EncodeChunk + i % n_bytes];
    }
  }
#endif
}

/** `process_and_fill_codes_subspaces_tf32_kernel` for k_steps from 1 to 5. */
template <uint32_t RowTiles, typename DataT, typename MathT, typename IdxT, typename LabelT>
auto tf32_encode_kernel(uint32_t k_steps)
{
  switch (k_steps) {
    case 1:
      return process_and_fill_codes_subspaces_tf32_kernel<1, RowTiles, DataT, MathT, IdxT, LabelT>;
    case 2:
      return process_and_fill_codes_subspaces_tf32_kernel<2, RowTiles, DataT, MathT, IdxT, LabelT>;
    case 3:
      return process_and_fill_codes_subspaces_tf32_kernel<3, RowTiles, DataT, MathT, IdxT, LabelT>;
    case 4:
      return process_and_fill_codes_subspaces_tf32_kernel<4, RowTiles, DataT, MathT, IdxT, LabelT>;
    default:
      return process_and_fill_codes_subspaces_tf32_kernel<5, RowTiles, DataT, MathT, IdxT, LabelT>;
  }
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

  // ponytail: the TF32 encoder covers pq_bits <= 8 (the codebook is staged in shared memory) and
  // either pq_len <= 10 (3xTF32 within a K of 32) or 16 <= pq_len <= 32 (plain TF32, which, as in
  // the k-means 1-NN, mislabels near-ties below 16 features). Other shapes and pre-sm_80 GPUs keep
  // the SIMT kernel.
  const bool split    = 3 * pq_len <= 32;
  const bool use_tf32 = pq_bits <= 8 && (split || (pq_len >= 16 && pq_len <= 32)) &&
                        raft::resource::get_device_properties(res).major >= 8;
  const uint32_t tf32_ld        = tf32_encode_ld(pq_len, split);
  const uint32_t tf32_k_steps   = tf32_encode_k_steps(tf32_ld);
  const uint32_t tf32_row_tiles = tf32_encode_row_tiles(pq_bits, split);
  auto tf32_kernel              = tf32_row_tiles == 2
                                    ? tf32_encode_kernel<2, data_t, MathT, IdxT, label_t>(tf32_k_steps)
                                    : tf32_encode_kernel<1, data_t, MathT, IdxT, label_t>(tf32_k_steps);
  const size_t tf32_smem        = tf32_encode_smem_bytes(pq_n_centers, tf32_ld, tf32_row_tiles);
  auto tf32_codebooks           = raft::make_device_matrix<float, uint32_t, raft::row_major>(
    res, use_tf32 ? pq_centers.extent(0) : 0, tf32_ld);
  if (use_tf32) {
    RAFT_CUDA_TRY(cudaFuncSetAttribute(
      tf32_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(tf32_smem)));
    raft::linalg::map_offset(
      res,
      raft::make_device_vector_view(tf32_codebooks.data_handle(), tf32_codebooks.size()),
      [pq_centers, tf32_ld, split] __device__(uint32_t i) {
        return tf32_codebook_element(pq_centers, i / tf32_ld, i % tf32_ld, split);
      });
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
      dim3 blocks(
        raft::div_rounding_up_safe<ix_t>(dataset_view.extent(0), tf32_encode_rows(tf32_row_tiles)),
        raft::div_rounding_up_safe<ix_t>(pq_dim, kTf32EncodeChunk),
        1);
      tf32_kernel<<<blocks, kTf32EncodeThreads, tf32_smem, stream.get()>>>(
        codes_view,
        dataset_view,
        raft::make_const_mdspan(tf32_codebooks.view()),
        vq_centers,
        raft::make_const_mdspan(labels_view),
        pq_bits,
        pq_len,
        split);
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
