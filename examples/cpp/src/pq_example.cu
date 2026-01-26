/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#define PQ_EXAMPLE_IVF_PQ 1
#define PQ_EXAMPLE_FAISS  1

#include <cuvs/cluster/kmeans.hpp>
#include <cuvs/core/bitmap.hpp>
#include <cuvs/neighbors/brute_force.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>
#include <cuvs/preprocessing/quantize/pq.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/resource/cuda_stream_pool.hpp>
#include <raft/random/make_blobs.cuh>

#include <chrono>
#include <cstring>
#ifdef PQ_EXAMPLE_FAISS
#include <faiss/impl/ProductQuantizer.h>
#endif
#include <fstream>
#include <iostream>
#include <rmm/mr/device_memory_resource.hpp>
#include <string>
#include <vector>

#include "ann_utils.cuh"
#include <omp.h>

struct ProgramArgs {
  std::string method = "pq";  // pq, ivfpq, faiss, interop_pq_faiss, interop_faiss_pq, all
  uint32_t pq_bits   = 8;
  uint32_t pq_dim    = 128;
  bool use_balanced  = true;
  bool use_subspace  = true;
  bool use_vq        = false;
  int64_t n_vectors  = 200000;
  int64_t n_dims     = 1024;

  void print_usage(const char* prog_name)
  {
    std::cout << "Usage: " << prog_name << " [options]\n"
              << "Options:\n"
              << "  --method <string>       PQ method (pq, ivfpq, faiss, interop_pq_faiss, "
                 "interop_faiss_pq, all, complete) [default: pq]\n"
              << "  --pq_bits <int>         Number of PQ bits (4-8) [default: 8]\n"
              << "  --pq_dim <int>          PQ dimension [default: 128]\n"
              << "  --use_balanced <0|1>    Use balanced k-means [default: 1]\n"
              << "  --use_subspace <0|1>    Use subspace quantization [default: 1]\n"
              << "  --use_vq <0|1>          Use vector quantization [default: 0]\n"
              << "  --n_vectors <int>       Number of vectors in dataset [default: 200000]\n"
              << "  --n_dims <int>          Number of dimensions in dataset [default: 1024]\n"
              << "  --help                  Show this help message\n";
  }

  bool parse_args(int argc, char** argv)
  {
    for (int i = 1; i < argc; i++) {
      std::string arg = argv[i];

      if (arg == "--help" || arg == "-h") {
        print_usage(argv[0]);
        return false;
      } else if (arg == "--method") {
        if (i + 1 < argc) {
          method = argv[++i];
        } else {
          std::cerr << "Error: --method requires an argument\n";
          return false;
        }
      } else if (arg == "--pq_bits") {
        if (i + 1 < argc) {
          pq_bits = std::stoi(argv[++i]);
        } else {
          std::cerr << "Error: --pq_bits requires an argument\n";
          return false;
        }
      } else if (arg == "--pq_dim") {
        if (i + 1 < argc) {
          pq_dim = std::stoi(argv[++i]);
        } else {
          std::cerr << "Error: --pq_dim requires an argument\n";
          return false;
        }
      } else if (arg == "--use_balanced") {
        if (i + 1 < argc) {
          use_balanced = (std::stoi(argv[++i]) != 0);
        } else {
          std::cerr << "Error: --use_balanced requires an argument\n";
          return false;
        }
      } else if (arg == "--use_subspace") {
        if (i + 1 < argc) {
          use_subspace = (std::stoi(argv[++i]) != 0);
        } else {
          std::cerr << "Error: --use_subspace requires an argument\n";
          return false;
        }
      } else if (arg == "--use_vq") {
        if (i + 1 < argc) {
          use_vq = (std::stoi(argv[++i]) != 0);
        } else {
          std::cerr << "Error: --use_vq requires an argument\n";
          return false;
        }
      } else if (arg == "--n_vectors") {
        if (i + 1 < argc) {
          n_vectors = std::stoll(argv[++i]);
        } else {
          std::cerr << "Error: --n_vectors requires an argument\n";
          return false;
        }
      } else if (arg == "--n_dims") {
        if (i + 1 < argc) {
          n_dims = std::stoll(argv[++i]);
        } else {
          std::cerr << "Error: --n_dims requires an argument\n";
          return false;
        }
      } else {
        std::cerr << "Error: Unknown argument: " << arg << "\n";
        print_usage(argv[0]);
        return false;
      }
    }
    return true;
  }
};

struct TimingResults {
  std::string method;
  uint32_t pq_bits;
  uint32_t pq_dim;
  double training_ms;
  double transform_ms;
  double inverse_transform_ms;
  double recall;
};

std::ostream& operator<<(std::ostream& os, const TimingResults& results)
{
  return os << results.method << "," << results.pq_bits << "," << results.pq_dim << ","
            << results.training_ms << "," << results.transform_ms << ","
            << results.inverse_transform_ms << "," << results.recall << "\n";
}

void load_dataset(const raft::device_resources& res, float* data_ptr, int n_vectors, int dim)
{
  // raft::random::RngState rng(1234ULL);
  // raft::random::uniform(res, rng, data_ptr, n_vectors * dim, -10.0f, 10.0f);
  auto labels = raft::make_device_vector<int64_t, int64_t>(res, n_vectors);
  raft::random::make_blobs<float, int64_t>(
    res,
    raft::make_device_matrix_view<float, int64_t>(data_ptr, n_vectors, dim),
    labels.view(),
    10);
}

void write_csv(const std::string& filename, const std::vector<TimingResults>& all_results)
{
  std::ofstream csv_file(filename);

  // Write header
  csv_file << "method,pq_bits,pq_dim,training_ms,transform_ms,inverse_transform_ms,recall\n";

  // Write data
  for (const auto& result : all_results) {
    csv_file << result;
  }

  csv_file.close();
  std::cout << "Timing results written to " << filename << std::endl;
}

void append_results_to_csv(const std::string& filename,
                           const std::vector<TimingResults>& results,
                           size_t start_idx)
{
  std::ofstream csv_file(filename, std::ios::app);  // Open in append mode

  // Write only the new results starting from start_idx
  for (size_t i = start_idx; i < results.size(); i++) {
    csv_file << results[i];
  }

  csv_file.close();
}

auto brute_force_search(raft::device_resources const& res,
                        raft::device_matrix_view<const float, int64_t, raft::row_major> dataset,
                        raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
                        int64_t k_neighbors)
{
  using namespace cuvs::neighbors;
  using namespace cuvs::preprocessing;
  using dataset_dtype  = float;
  using indexing_dtype = int64_t;
  auto n_queries       = queries.extent(0);
  auto neighbors_gt =
    raft::make_device_matrix<indexing_dtype, indexing_dtype>(res, n_queries, k_neighbors);
  auto distances_gt = raft::make_device_matrix<float, indexing_dtype>(res, n_queries, k_neighbors);
  auto index = cuvs::neighbors::brute_force::build(res, brute_force::index_params{}, dataset);
  cuvs::neighbors::brute_force::search(
    res, brute_force::search_params{}, index, queries, neighbors_gt.view(), distances_gt.view());
  return neighbors_gt;
}

auto brute_force_search(raft::device_resources const& res,
                        raft::host_matrix_view<const float, int64_t, raft::row_major> dataset,
                        raft::host_matrix_view<const float, int64_t, raft::row_major> queries,
                        int64_t k_neighbors)
{
  using namespace cuvs::neighbors;
  using namespace cuvs::preprocessing;
  using dataset_dtype  = float;
  using indexing_dtype = int64_t;
  auto stream          = raft::resource::get_cuda_stream(res);
  auto n_queries       = queries.extent(0);
  auto neighbors_gt =
    raft::make_device_matrix<indexing_dtype, indexing_dtype>(res, n_queries, k_neighbors);
  auto neighbors_gt_cpu =
    raft::make_host_matrix<indexing_dtype, indexing_dtype>(n_queries, k_neighbors);
  auto distances_gt = raft::make_device_matrix<float, indexing_dtype>(res, n_queries, k_neighbors);
  auto dataset_device = raft::make_device_matrix<dataset_dtype, indexing_dtype>(
    res, dataset.extent(0), dataset.extent(1));
  auto queries_device = raft::make_device_matrix<dataset_dtype, indexing_dtype>(
    res, queries.extent(0), queries.extent(1));
  raft::copy(dataset_device.data_handle(),
             dataset.data_handle(),
             dataset.extent(0) * dataset.extent(1),
             stream);
  raft::copy(queries_device.data_handle(),
             queries.data_handle(),
             queries.extent(0) * queries.extent(1),
             stream);
  auto index = cuvs::neighbors::brute_force::build(
    res, brute_force::index_params{}, raft::make_const_mdspan(dataset_device.view()));
  cuvs::neighbors::brute_force::search(res,
                                       brute_force::search_params{},
                                       index,
                                       raft::make_const_mdspan(queries_device.view()),
                                       neighbors_gt.view(),
                                       distances_gt.view());
  raft::copy(
    neighbors_gt_cpu.data_handle(), neighbors_gt.data_handle(), n_queries * k_neighbors, stream);
  raft::resource::sync_stream(res);
  return neighbors_gt_cpu;
}

#ifdef PQ_EXAMPLE_FAISS
void faiss_pq(raft::device_resources const& res,
              raft::host_matrix_view<float, int64_t> dataset_cpu,
              uint32_t pq_bits,
              uint32_t pq_dim,
              std::vector<TimingResults>& all_results,
              raft::host_matrix_view<const float, int64_t, raft::row_major> queries_cpu,
              raft::host_matrix_view<const int64_t, int64_t, raft::row_major> neighbors_gt)
{
  using indexing_dtype       = int64_t;
  int64_t n_vectors          = dataset_cpu.extent(0);
  int64_t dim                = dataset_cpu.extent(1);
  int64_t n_queries          = queries_cpu.extent(0);
  int64_t k_neighbors        = neighbors_gt.extent(1);
  int64_t code_size          = (pq_bits * pq_dim + 7) / 8;
  auto transformed_dataset   = raft::make_host_matrix<uint8_t, int64_t>(n_vectors, code_size);
  auto reconstructed_dataset = raft::make_host_matrix<float, int64_t>(n_vectors, dim);
  faiss::ProductQuantizer pq(dim, pq_dim, pq_bits);

  auto n_vectors_train = std::min(n_vectors, (int64_t)(1 << pq_bits) * 256);
  auto start_time      = std::chrono::high_resolution_clock::now();
  pq.train(n_vectors_train, dataset_cpu.data_handle());
  auto end_time      = std::chrono::high_resolution_clock::now();
  auto duration      = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
  double training_ms = duration.count();

  TimingResults results;
  results.method      = "faiss";
  results.pq_bits     = pq_bits;
  results.pq_dim      = pq_dim;
  results.training_ms = training_ms;

  start_time = std::chrono::high_resolution_clock::now();
  pq.compute_codes(dataset_cpu.data_handle(), transformed_dataset.data_handle(), n_vectors);
  end_time       = std::chrono::high_resolution_clock::now();
  auto uduration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
  results.transform_ms = uduration.count() / 1000.0;

  start_time = std::chrono::high_resolution_clock::now();
  pq.decode(transformed_dataset.data_handle(), reconstructed_dataset.data_handle(), n_vectors);
  end_time = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
  results.inverse_transform_ms = duration.count();

  // Compute recall on reconstructed dataset
  auto neighbors_reconstructed =
    brute_force_search(res, reconstructed_dataset.view(), queries_cpu, k_neighbors);

  std::vector<indexing_dtype> neighbors_gt_vec(
    neighbors_gt.data_handle(), neighbors_gt.data_handle() + n_queries * k_neighbors);
  std::vector<indexing_dtype> neighbors_reconstructed_vec(
    neighbors_reconstructed.data_handle(),
    neighbors_reconstructed.data_handle() + n_queries * k_neighbors);
  auto [recall, match_count, total_count] = cuvs::neighbors::calc_recall(
    neighbors_gt_vec, neighbors_reconstructed_vec, n_queries, k_neighbors);
  results.recall = recall;

  all_results.push_back(results);
  std::cout << results;
}
#endif

#ifdef PQ_EXAMPLE_IVF_PQ
void ivf_pq_api(raft::device_resources const& res,
                raft::device_matrix_view<const float, int64_t, raft::row_major> dataset,
                uint32_t pq_bits,
                uint32_t pq_dim,
                std::vector<TimingResults>& all_results,
                raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
                raft::host_matrix_view<const int64_t, int64_t, raft::row_major> neighbors_gt)
{
  using namespace cuvs::neighbors;  // NOLINT
  using indexing_dtype = int64_t;
  auto stream          = raft::resource::get_cuda_stream(res);
  auto n_queries       = queries.extent(0);
  auto k_neighbors     = neighbors_gt.extent(1);
  auto out =
    raft::make_device_matrix<uint8_t, uint32_t, raft::row_major>(res, dataset.extent(0), pq_dim);
  auto neighbors =
    raft::make_device_matrix<indexing_dtype, indexing_dtype>(res, n_queries, k_neighbors);
  auto distances = raft::make_device_matrix<float, indexing_dtype>(res, n_queries, k_neighbors);

  ivf_pq::index_params index_params;
  index_params.n_lists                  = 1;
  index_params.kmeans_trainset_fraction = 0.2;
  index_params.metric                   = cuvs::distance::DistanceType::L2Expanded;
  index_params.pq_bits                  = pq_bits;
  index_params.pq_dim                   = pq_dim;
  // index_params.max_train_points_per_pq_code = 20000;
  index_params.add_data_on_build = false;

  auto start = std::chrono::high_resolution_clock::now();
  auto index = ivf_pq::build(res, index_params, dataset);
  raft::resource::sync_stream(res);
  auto end           = std::chrono::high_resolution_clock::now();
  auto duration      = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  double training_ms = duration.count();

  TimingResults results;
  results.method      = "ivfpq";
  results.pq_bits     = pq_bits;
  results.pq_dim      = pq_dim;
  results.training_ms = training_ms;

  start               = std::chrono::high_resolution_clock::now();
  auto extended_index = ivf_pq::extend(res, dataset, std::nullopt, index);
  ivf_pq::helpers::codepacker::unpack_list_data(res, extended_index, out.view(), 0, 0);
  raft::resource::sync_stream(res);
  end                  = std::chrono::high_resolution_clock::now();
  auto uduration       = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  results.transform_ms = uduration.count() / 1000.0;

  // IVF-PQ doesn't have a direct inverse transform, so set to 0
  results.inverse_transform_ms = 0.0;

  // Compute recall by searching with IVF-PQ
  ivf_pq::search_params search_params;
  search_params.n_probes = 1;
  ivf_pq::search(res, search_params, extended_index, queries, neighbors.view(), distances.view());
  raft::resource::sync_stream(res);

  std::vector<indexing_dtype> neighbors_gt_vec(
    neighbors_gt.data_handle(), neighbors_gt.data_handle() + n_queries * k_neighbors);
  std::vector<indexing_dtype> neighbors_reconstructed_vec(n_queries * k_neighbors);
  raft::copy(
    neighbors_reconstructed_vec.data(), neighbors.data_handle(), n_queries * k_neighbors, stream);
  raft::resource::sync_stream(res);
  auto [recall, match_count, total_count] =
    calc_recall(neighbors_gt_vec, neighbors_reconstructed_vec, n_queries, k_neighbors);
  results.recall = recall;

  all_results.push_back(results);
  std::cout << results;
}
#endif

void pq_api(raft::device_resources const& res,
            raft::device_matrix_view<const float, int64_t, raft::row_major> dataset,
            uint32_t pq_bits,
            uint32_t pq_dim,
            std::vector<TimingResults>& all_results,
            raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
            raft::host_matrix_view<const int64_t, int64_t, raft::row_major> neighbors_gt,
            bool use_subspaces,
            bool use_vq,
            bool kmeans_balanced)
{
  using namespace cuvs::neighbors;
  using namespace cuvs::preprocessing;
  using dataset_dtype  = float;
  using indexing_dtype = int64_t;
  auto stream          = raft::resource::get_cuda_stream(res);
  auto dim             = dataset.extent(1);
  auto n_vectors       = dataset.extent(0);
  auto n_queries       = queries.extent(0);
  auto k_neighbors     = neighbors_gt.extent(1);
  auto kmeans_type     = kmeans_balanced ? cuvs::cluster::kmeans::kmeans_type::KMeansBalanced
                                         : cuvs::cluster::kmeans::kmeans_type::KMeans;
  auto params =
    quantize::pq::params{pq_bits, pq_dim, use_subspaces, use_vq, 0, 25, kmeans_type, 256, 1024};
  auto reconstructed_dataset =
    raft::make_device_matrix<dataset_dtype, indexing_dtype>(res, n_vectors, dim);
  auto quantized_dim = quantize::pq::get_quantized_dim(params);
  auto vq_labels     = raft::make_device_vector<uint32_t, indexing_dtype>(res, n_vectors);
  std::optional<raft::device_vector_view<uint32_t, indexing_dtype>> vq_labels_view = std::nullopt;
  std::optional<raft::device_vector_view<const uint32_t, indexing_dtype>> vq_labels_const_view =
    std::nullopt;
  if (use_vq) {
    vq_labels_view       = vq_labels.view();
    vq_labels_const_view = raft::make_const_mdspan(vq_labels.view());
  }
  auto transformed_dataset =
    raft::make_device_matrix<uint8_t, indexing_dtype>(res, n_vectors, quantized_dim);

  raft::resource::sync_stream(res);
  auto start     = std::chrono::high_resolution_clock::now();
  auto quantizer = quantize::pq::build(res, params, raft::make_const_mdspan(dataset));
  raft::resource::sync_stream(res);
  auto end           = std::chrono::high_resolution_clock::now();
  auto duration      = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  double training_ms = duration.count();

  TimingResults results;
  results.method = "pq";
  if (use_subspaces) { results.method += "_subspaces"; }
  if (use_vq) { results.method += "_vq"; }
  if (kmeans_balanced) {
    results.method += "_kmeans_balanced";
  } else {
    results.method += "_kmeans_classic";
  }
  results.pq_bits     = params.pq_bits;
  results.pq_dim      = params.pq_dim;
  results.training_ms = training_ms;

  start = std::chrono::high_resolution_clock::now();
  quantize::pq::transform(
    res, quantizer, raft::make_const_mdspan(dataset), transformed_dataset.view(), vq_labels_view);
  raft::resource::sync_stream(res);
  end                  = std::chrono::high_resolution_clock::now();
  auto uduration       = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  results.transform_ms = uduration.count() / 1000.0;

  start = std::chrono::high_resolution_clock::now();
  quantize::pq::inverse_transform(res,
                                  quantizer,
                                  raft::make_const_mdspan(transformed_dataset.view()),
                                  reconstructed_dataset.view(),
                                  vq_labels_const_view);
  raft::resource::sync_stream(res);
  end                          = std::chrono::high_resolution_clock::now();
  duration                     = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  results.inverse_transform_ms = duration.count();

  // Compute recall on reconstructed dataset
  auto neighbors_reconstructed = brute_force_search(
    res, raft::make_const_mdspan(reconstructed_dataset.view()), queries, k_neighbors);
  auto neighbors_reconstructed_cpu =
    raft::make_host_matrix<indexing_dtype, indexing_dtype>(n_queries, k_neighbors);
  raft::copy(neighbors_reconstructed_cpu.data_handle(),
             neighbors_reconstructed.data_handle(),
             n_queries * k_neighbors,
             stream);
  raft::resource::sync_stream(res);

  std::vector<indexing_dtype> neighbors_gt_vec(
    neighbors_gt.data_handle(), neighbors_gt.data_handle() + n_queries * k_neighbors);
  std::vector<indexing_dtype> neighbors_reconstructed_vec(
    neighbors_reconstructed_cpu.data_handle(),
    neighbors_reconstructed_cpu.data_handle() + n_queries * k_neighbors);
  auto [recall, match_count, total_count] =
    calc_recall(neighbors_gt_vec, neighbors_reconstructed_vec, n_queries, k_neighbors);
  results.recall = recall;

  all_results.push_back(results);
  std::cout << results;
}

#ifdef PQ_EXAMPLE_FAISS
void interop_pq_faiss(raft::device_resources const& res,
                      raft::device_matrix_view<const float, int64_t, raft::row_major> dataset,
                      uint32_t pq_bits,
                      uint32_t pq_dim,
                      std::vector<TimingResults>& all_results,
                      raft::host_matrix_view<const float, int64_t, raft::row_major> queries_cpu,
                      raft::host_matrix_view<const int64_t, int64_t, raft::row_major> neighbors_gt)
{
  using namespace cuvs::neighbors;
  using namespace cuvs::preprocessing;
  using dataset_dtype  = float;
  using indexing_dtype = int64_t;
  auto stream          = raft::resource::get_cuda_stream(res);
  auto dim             = dataset.extent(1);
  auto n_vectors       = dataset.extent(0);
  auto n_queries       = queries_cpu.extent(0);
  auto k_neighbors     = neighbors_gt.extent(1);
  auto code_size       = (pq_bits * pq_dim + 7) / 8;
  auto pq_n_centers    = 1 << pq_bits;
  auto use_subspaces   = true;
  auto use_vq          = false;
  auto kmeans_type     = cuvs::cluster::kmeans::kmeans_type::KMeans;
  auto params =
    quantize::pq::params{pq_bits, pq_dim, use_subspaces, use_vq, 0, 25, kmeans_type, 256, 1024};

  auto dataset_cpu           = raft::make_host_matrix<float, int64_t>(n_vectors, dim);
  auto transformed_dataset   = raft::make_host_matrix<uint8_t, int64_t>(n_vectors, code_size);
  auto reconstructed_dataset = raft::make_host_matrix<float, int64_t>(n_vectors, dim);
  raft::copy(dataset_cpu.data_handle(), dataset.data_handle(), n_vectors * dim, stream);

  raft::resource::sync_stream(res);
  auto start     = std::chrono::high_resolution_clock::now();
  auto quantizer = quantize::pq::build(res, params, raft::make_const_mdspan(dataset));
  raft::resource::sync_stream(res);
  auto end           = std::chrono::high_resolution_clock::now();
  auto duration      = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  double training_ms = duration.count();

  faiss::ProductQuantizer faiss_pq(dim, pq_dim, pq_bits);
  faiss_pq.set_derived_values();
  auto centroids_cpu =
    raft::make_host_vector<float, int64_t>(quantizer.vpq_codebooks.pq_code_book.size());
  raft::copy(centroids_cpu.data_handle(),
             quantizer.vpq_codebooks.pq_code_book.data_handle(),
             quantizer.vpq_codebooks.pq_code_book.size(),
             stream);
  raft::resource::sync_stream(res);
  for (int i = 0; i < pq_dim; i++) {
    if (use_subspaces) {
      auto sub_dim_start = i * pq_n_centers * quantizer.vpq_codebooks.pq_code_book.extent(1);
      faiss_pq.set_params(centroids_cpu.data_handle() + sub_dim_start, i);
    } else {
      faiss_pq.set_params(centroids_cpu.data_handle(), i);
    }
  }
  faiss_pq.sync_transposed_centroids();

  TimingResults results;
  results.method      = "interop_train_pq_transform_faiss";
  results.pq_bits     = pq_bits;
  results.pq_dim      = pq_dim;
  results.training_ms = training_ms;

  start = std::chrono::high_resolution_clock::now();
  faiss_pq.compute_codes(dataset_cpu.data_handle(), transformed_dataset.data_handle(), n_vectors);
  end                  = std::chrono::high_resolution_clock::now();
  auto uduration       = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  results.transform_ms = uduration.count() / 1000.0;

  start = std::chrono::high_resolution_clock::now();
  faiss_pq.decode(
    transformed_dataset.data_handle(), reconstructed_dataset.data_handle(), n_vectors);
  end                          = std::chrono::high_resolution_clock::now();
  duration                     = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  results.inverse_transform_ms = duration.count();

  // Compute recall on reconstructed dataset
  auto neighbors_reconstructed =
    brute_force_search(res, reconstructed_dataset.view(), queries_cpu, k_neighbors);

  std::vector<indexing_dtype> neighbors_gt_vec(
    neighbors_gt.data_handle(), neighbors_gt.data_handle() + n_queries * k_neighbors);
  std::vector<indexing_dtype> neighbors_reconstructed_vec(
    neighbors_reconstructed.data_handle(),
    neighbors_reconstructed.data_handle() + n_queries * k_neighbors);
  auto [recall, match_count, total_count] =
    calc_recall(neighbors_gt_vec, neighbors_reconstructed_vec, n_queries, k_neighbors);
  results.recall = recall;

  std::cout << results;
  all_results.push_back(results);
}

void interop_faiss_pq(raft::device_resources const& res,
                      raft::device_matrix_view<const float, int64_t, raft::row_major> dataset,
                      raft::host_matrix_view<const float, int64_t, raft::row_major> dataset_cpu,
                      uint32_t pq_bits,
                      uint32_t pq_dim,
                      std::vector<TimingResults>& all_results,
                      raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
                      raft::host_matrix_view<const int64_t, int64_t, raft::row_major> neighbors_gt)
{
  using namespace cuvs::neighbors;
  using namespace cuvs::preprocessing;
  using dataset_dtype  = float;
  using indexing_dtype = int64_t;
  auto stream          = raft::resource::get_cuda_stream(res);
  auto dim             = dataset.extent(1);
  auto n_vectors       = dataset.extent(0);
  auto n_queries       = queries.extent(0);
  auto k_neighbors     = neighbors_gt.extent(1);
  auto pq_len          = (dim + pq_dim - 1) / pq_dim;
  auto code_size       = (pq_bits * pq_dim + 7) / 8;
  auto pq_n_centers    = 1 << pq_bits;
  auto use_subspaces   = true;
  auto use_vq          = false;
  auto kmeans_type     = cuvs::cluster::kmeans::kmeans_type::KMeans;
  auto params =
    quantize::pq::params{pq_bits, pq_dim, use_subspaces, use_vq, 0, 25, kmeans_type, 256, 1024};

  auto transformed_dataset = raft::make_device_matrix<uint8_t, int64_t>(res, n_vectors, code_size);
  auto reconstructed_dataset = raft::make_device_matrix<float, int64_t>(res, n_vectors, dim);

  faiss::ProductQuantizer faiss_pq(dim, pq_dim, pq_bits);

  auto start = std::chrono::high_resolution_clock::now();
  faiss_pq.train(n_vectors, dataset_cpu.data_handle());
  auto end           = std::chrono::high_resolution_clock::now();
  auto duration      = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  double training_ms = duration.count();
  std::optional<raft::device_matrix<float, uint32_t, raft::row_major>> opt_centroids = std::nullopt;

  if (use_subspaces) {
    opt_centroids = std::make_optional(
      raft::make_device_matrix<float, uint32_t>(res, faiss_pq.centroids.size() / pq_len, pq_len));
    raft::copy(opt_centroids.value().data_handle(),
               faiss_pq.centroids.data(),
               faiss_pq.centroids.size(),
               stream);
  } else {
    opt_centroids =
      std::make_optional(raft::make_device_matrix<float, uint32_t>(res, pq_n_centers, pq_len));
    raft::copy(opt_centroids.value().data_handle(),
               faiss_pq.centroids.data(),
               faiss_pq.centroids.size(),
               stream);
  }
  raft::resource::sync_stream(res);
  quantize::pq::quantizer<float> quantizr = {
    params,
    cuvs::neighbors::vpq_dataset<float, int64_t>{
      raft::make_device_matrix<float, uint32_t>(res, 0, 0),  // vq codebook
      std::move(opt_centroids.value()),                      // pq codebook
      raft::make_device_matrix<uint8_t, int64_t>(res, 0, 0)}};

  TimingResults results;
  results.method      = "interop_train_faiss_transform_pq";
  results.pq_bits     = pq_bits;
  results.pq_dim      = pq_dim;
  results.training_ms = training_ms;

  start = std::chrono::high_resolution_clock::now();
  quantize::pq::transform(res, quantizr, dataset, transformed_dataset.view());
  raft::resource::sync_stream(res);
  end                  = std::chrono::high_resolution_clock::now();
  duration             = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  results.transform_ms = duration.count();

  start = std::chrono::high_resolution_clock::now();
  quantize::pq::inverse_transform(res,
                                  quantizr,
                                  raft::make_const_mdspan(transformed_dataset.view()),
                                  reconstructed_dataset.view());
  raft::resource::sync_stream(res);
  end                          = std::chrono::high_resolution_clock::now();
  duration                     = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  results.inverse_transform_ms = duration.count();

  // Compute recall on reconstructed dataset
  auto neighbors_reconstructed = brute_force_search(
    res, raft::make_const_mdspan(reconstructed_dataset.view()), queries, k_neighbors);

  auto neighbors_reconstructed_cpu =
    raft::make_host_matrix<indexing_dtype, indexing_dtype>(n_queries, k_neighbors);
  raft::copy(neighbors_reconstructed_cpu.data_handle(),
             neighbors_reconstructed.data_handle(),
             n_queries * k_neighbors,
             stream);
  raft::resource::sync_stream(res);

  std::vector<indexing_dtype> neighbors_gt_vec(
    neighbors_gt.data_handle(), neighbors_gt.data_handle() + n_queries * k_neighbors);
  std::vector<indexing_dtype> neighbors_reconstructed_vec(
    neighbors_reconstructed_cpu.data_handle(),
    neighbors_reconstructed_cpu.data_handle() + n_queries * k_neighbors);
  auto [recall, match_count, total_count] =
    calc_recall(neighbors_gt_vec, neighbors_reconstructed_vec, n_queries, k_neighbors);
  results.recall = recall;

  all_results.push_back(results);
  std::cout << results;
}
#endif

int main(int argc, char** argv)
{
  using namespace cuvs::neighbors;
  using namespace cuvs::preprocessing;
  using dataset_dtype  = float;
  using indexing_dtype = int64_t;

  // Parse command line arguments
  ProgramArgs args;
  if (!args.parse_args(argc, argv)) { return 1; }

  // Print configuration
  std::cout << "Configuration:\n"
            << "  Method: " << args.method << "\n"
            << "  PQ bits: " << args.pq_bits << "\n"
            << "  PQ dim: " << args.pq_dim << "\n"
            << "  Use balanced: " << (args.use_balanced ? "true" : "false") << "\n"
            << "  Use subspace: " << (args.use_subspace ? "true" : "false") << "\n"
            << "  Use VQ: " << (args.use_vq ? "true" : "false") << "\n"
            << "  Number of vectors: " << args.n_vectors / 1000 << "k\n"
            << std::endl;

  auto dim         = args.n_dims;
  auto n_vectors   = args.n_vectors;
  auto n_queries   = 1000;
  auto k_neighbors = 50;

  raft::device_resources res;
  raft::resource::set_workspace_to_pool_resource(res, 6 * 1024 * 1024 * 1024ull);
  size_t n_streams = 1;
  raft::resource::set_cuda_stream_pool(res, std::make_shared<rmm::cuda_stream_pool>(n_streams));
  auto stream  = raft::resource::get_cuda_stream(res);
  auto dataset = raft::make_device_matrix<dataset_dtype, indexing_dtype>(res, n_vectors, dim);
  auto queries = raft::make_device_matrix<dataset_dtype, indexing_dtype>(res, n_queries, dim);

  load_dataset(res, dataset.data_handle(), n_vectors, dim);
  load_dataset(res, queries.data_handle(), n_queries, dim);
  auto dataset_cpu      = raft::make_host_matrix<float, int64_t>(n_vectors, dim);
  auto queries_cpu      = raft::make_host_matrix<float, int64_t>(n_queries, dim);
  auto neighbors_gt_cpu = raft::make_host_matrix<indexing_dtype, int64_t>(n_queries, k_neighbors);
  raft::copy(dataset_cpu.data_handle(), dataset.data_handle(), n_vectors * dim, stream);
  raft::copy(queries_cpu.data_handle(), queries.data_handle(), n_queries * dim, stream);

  auto neighbors_gt = brute_force_search(res,
                                         raft::make_const_mdspan(dataset.view()),
                                         raft::make_const_mdspan(queries.view()),
                                         k_neighbors);
  raft::copy(
    neighbors_gt_cpu.data_handle(), neighbors_gt.data_handle(), n_queries * k_neighbors, stream);
  raft::resource::sync_stream(res);

  std::vector<TimingResults> all_results;

  if (args.method != "complete") {
    std::cout << "=== PQ bits: " << args.pq_bits << ", PQ dim: " << args.pq_dim
              << " ===" << std::endl;
    for (int run = 0; run < 4; run++) {
      // Run the specified method or all methods
      if (args.method == "pq" || args.method == "all") {
        pq_api(res,
               raft::make_const_mdspan(dataset.view()),
               args.pq_bits,
               args.pq_dim,
               all_results,
               raft::make_const_mdspan(queries.view()),
               neighbors_gt_cpu.view(),
               args.use_subspace,
               args.use_vq,
               args.use_balanced);
      }

      if (args.method == "ivfpq" || args.method == "all") {
#ifdef PQ_EXAMPLE_IVF_PQ
        ivf_pq_api(res,
                   raft::make_const_mdspan(dataset.view()),
                   args.pq_bits,
                   args.pq_dim,
                   all_results,
                   raft::make_const_mdspan(queries.view()),
                   neighbors_gt_cpu.view());
#endif
      }

#ifdef PQ_EXAMPLE_FAISS
      if (args.method == "faiss" || args.method == "all") {
        faiss_pq(res,
                 dataset_cpu.view(),
                 args.pq_bits,
                 args.pq_dim,
                 all_results,
                 queries_cpu.view(),
                 neighbors_gt_cpu.view());
      }

      if (args.method == "interop_pq_faiss" || args.method == "all") {
        interop_pq_faiss(res,
                         raft::make_const_mdspan(dataset.view()),
                         args.pq_bits,
                         args.pq_dim,
                         all_results,
                         queries_cpu.view(),
                         neighbors_gt_cpu.view());
      }

      if (args.method == "interop_faiss_pq" || args.method == "all") {
        interop_faiss_pq(res,
                         raft::make_const_mdspan(dataset.view()),
                         raft::make_const_mdspan(dataset_cpu.view()),
                         args.pq_bits,
                         args.pq_dim,
                         all_results,
                         raft::make_const_mdspan(queries.view()),
                         neighbors_gt_cpu.view());
      }
#endif
    }
  } else {  // complete benchmark saved to a csv file
    // Write CSV header once at the beginning
    std::ofstream csv_file("pq_timing_results.csv");
    csv_file << "method,pq_bits,pq_dim,training_ms,transform_ms,inverse_transform_ms,recall\n";
    csv_file.close();

    for (uint32_t run = 0; run < 3; run++) {
      std::cout << "=== Run: " << run << " ===" << std::endl;
      for (uint32_t pq_bits : {6u, 7u, 8u, 10u, 14u}) {
        for (uint32_t pq_dim : {32u, 64u, 128u, 256u, 512u}) {
          std::cout << "=== PQ bits: " << pq_bits << ", PQ dim: " << pq_dim << " ===" << std::endl;

          // Track where new results start
          size_t results_start_idx = all_results.size();

          // Test PQ method
          for (bool kmeans_balanced : {false, true}) {
            for (bool use_vq : {false /*, true*/}) {
              for (bool use_subspaces : {false, true}) {
                pq_api(res,
                       raft::make_const_mdspan(dataset.view()),
                       pq_bits,
                       pq_dim,
                       all_results,
                       raft::make_const_mdspan(queries.view()),
                       neighbors_gt_cpu.view(),
                       use_subspaces,
                       use_vq,
                       kmeans_balanced);
              }
            }
          }

          // Test IVF-PQ method
#ifdef PQ_EXAMPLE_IVF_PQ
          ivf_pq_api(res,
                     raft::make_const_mdspan(dataset.view()),
                     pq_bits,
                     pq_dim,
                     all_results,
                     raft::make_const_mdspan(queries.view()),
                     neighbors_gt_cpu.view());
#endif

          // Test FAISS method
#ifdef PQ_EXAMPLE_FAISS
          faiss_pq(res,
                   dataset_cpu.view(),
                   pq_bits,
                   pq_dim,
                   all_results,
                   queries_cpu.view(),
                   neighbors_gt_cpu.view());

          // Test interop PQ FAISS method
          /*interop_pq_faiss(res,
                           raft::make_const_mdspan(dataset.view()),
                           pq_bits,
                           pq_dim,
                           all_results,
                           queries_cpu.view(),
                           neighbors_gt_cpu.view());*/

          // Test interop FAISS PQ method
          /*interop_faiss_pq(res,
                          raft::make_const_mdspan(dataset.view()),
                          raft::make_const_mdspan(dataset_cpu.view()),
                          pq_bits,
                          pq_dim,
                          all_results,
                          raft::make_const_mdspan(queries.view()),
                          neighbors_gt_cpu.view());*/
#endif

          // Append results for this configuration to CSV
          append_results_to_csv("pq_timing_results.csv", all_results, results_start_idx);
        }
      }
    }
    std::cout << "All timing results written to pq_timing_results.csv" << std::endl;
  }
  return 0;
}
