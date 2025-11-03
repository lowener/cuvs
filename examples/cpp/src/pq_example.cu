/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/cluster/kmeans.hpp>
#include <cuvs/core/bitmap.hpp>
#include <cuvs/neighbors/brute_force.hpp>
#include <cuvs/preprocessing/quantize/product.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/random/make_blobs.cuh>

#include <chrono>
#include <iostream>
#include <rmm/mr/device/device_memory_resource.hpp>

void load_dataset(const raft::device_resources& res, float* data_ptr, int n_vectors, int dim)
{
  raft::random::RngState rng(1234ULL);
  raft::random::uniform(res, rng, data_ptr, n_vectors * dim, -10.0f, 10.0f);
}

int main()
{
  using namespace cuvs::neighbors;
  using namespace cuvs::preprocessing;
  using dataset_dtype   = float;
  using indexing_dtype  = int64_t;
  auto dim              = 1024;
  auto n_vectors        = 100000;
  uint32_t pq_bits      = 8;
  uint32_t pq_dim       = 512;
  uint32_t vq_n_centers = 1;
  auto kmeans_type      = cuvs::cluster::kmeans::kmeans_type::KMeans;
  quantize::product::params params{pq_bits, pq_dim, vq_n_centers, 25, 0, 0, kmeans_type};

  raft::device_resources res;
  auto dataset = raft::make_device_matrix<dataset_dtype, indexing_dtype>(res, n_vectors, dim);
  auto reconstructed_dataset =
    raft::make_device_matrix<dataset_dtype, indexing_dtype>(res, n_vectors, dim);
  auto quantized_dim = quantize::product::get_quantized_dim(params);
  auto transformed_dataset =
    raft::make_device_matrix<uint8_t, indexing_dtype>(res, n_vectors, quantized_dim);

  load_dataset(res, dataset.data_handle(), n_vectors, dim);

  raft::resource::sync_stream(res);
  auto start     = std::chrono::high_resolution_clock::now();
  auto quantizer = quantize::product::train(res, params, raft::make_const_mdspan(dataset.view()));
  raft::resource::sync_stream(res);
  auto end      = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "Training time: " << duration.count() << " milliseconds" << std::endl;

  start = std::chrono::high_resolution_clock::now();
  quantize::product::transform(
    res, quantizer, raft::make_const_mdspan(dataset.view()), transformed_dataset.view());
  raft::resource::sync_stream(res);
  end      = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "Transform time: " << duration.count() << " milliseconds" << std::endl;

  start = std::chrono::high_resolution_clock::now();
  quantize::product::inverse_transform(res,
                                       quantizer,
                                       raft::make_const_mdspan(transformed_dataset.view()),
                                       reconstructed_dataset.view());
  raft::resource::sync_stream(res);
  end      = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "Inverse transform time: " << duration.count() << " milliseconds" << std::endl;
  return 0;
}
