/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

// A/B benchmark of the TF32 tensor-core PQ encoder against the SIMT one. The TF32 path is disabled
// at runtime with CUVS_PQ_DISABLE_TF32 (a temporary switch in process_and_fill_codes_subspaces).

#include <cuvs/neighbors/scann.hpp>
#include <cuvs/neighbors/vamana.hpp>
#include <cuvs/preprocessing/quantize/pq.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/linalg/gemm.cuh>
#include <raft/random/make_blobs.cuh>

#include <rmm/mr/pool_memory_resource.hpp>

#include <thrust/count.h>
#include <thrust/functional.h>
#include <thrust/inner_product.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace pq = cuvs::preprocessing::quantize::pq;
namespace scann = cuvs::neighbors::experimental::scann;
using dmat_view = raft::device_matrix_view<const float, int64_t>;

void set_tf32(bool on)
{
  if (std::getenv("BENCH_SIMT_ONLY")) { on = false; }
  if (on) {
    unsetenv("CUVS_PQ_DISABLE_TF32");
  } else {
    setenv("CUVS_PQ_DISABLE_TF32", "1", 1);
  }
}

/** Median wall time (ms) of `reps` synchronized calls of f, after one warm-up call. */
template <typename F>
double median_ms(raft::device_resources const& res, int reps, F&& f)
{
  f();
  res.sync_stream();
  std::vector<double> t;
  for (int i = 0; i < reps; i++) {
    auto t0 = std::chrono::steady_clock::now();
    f();
    res.sync_stream();
    t.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                  .count());
  }
  std::sort(t.begin(), t.end());
  return t[t.size() / 2];
}

double mse(raft::device_resources const& res, const float* a, const float* b, size_t n)
{
  return thrust::inner_product(
           raft::resource::get_thrust_policy(res),
           a,
           a + n,
           b,
           0.0,
           thrust::plus<double>{},
           [] __device__(float x, float y) -> double { return double(x - y) * double(x - y); }) /
         double(n);
}

void bench_transform(raft::device_resources const& res, dmat_view dataset, int reps)
{
  const int64_t n = dataset.extent(0), dim = dataset.extent(1);
  const uint32_t n_vq = std::getenv("BENCH_VQ") ? std::atoi(std::getenv("BENCH_VQ")) : 0;
  std::printf("\n== pq::transform, %ld x %ld, subspaces, %u VQ centers ==\n", n, dim, n_vq);
  std::printf("%7s %6s %6s | %10s %10s %7s | %12s %12s %12s\n",
              "pq_bits", "pq_dim", "pq_len", "simt_ms", "tf32_ms", "speedup",
              "code_diff_%", "mse_simt", "mse_tf32");
  auto recon = raft::make_device_matrix<float, int64_t>(res, n, dim);
  std::vector<uint32_t> bits_list{8, 4};
  if (const char* e = std::getenv("BENCH_PQ_BITS")) { bits_list = {uint32_t(std::atoi(e))}; }
  for (uint32_t pq_bits : bits_list) {
    std::vector<uint32_t> dims{512, 256, 128, 64, 32};
    if (std::getenv("BENCH_SHORT")) { dims = {512, 256, 128}; }
    for (uint32_t pq_dim : dims) {
      pq::params params{
        pq_bits, pq_dim, true, n_vq > 0, n_vq, 25, cuvs::cluster::kmeans::kmeans_type::KMeansBalanced,
        256, 1024};
      auto quantizer = pq::build(res, params, dataset);
      const int64_t code_len = pq::get_quantized_dim(params);
      auto codes_simt = raft::make_device_matrix<uint8_t, int64_t>(res, n, code_len);
      auto codes_tf32 = raft::make_device_matrix<uint8_t, int64_t>(res, n, code_len);

      set_tf32(false);
      double simt = median_ms(res, reps, [&] { pq::transform(res, quantizer, dataset, codes_simt.view()); });
      set_tf32(true);
      double tf32 = median_ms(res, reps, [&] { pq::transform(res, quantizer, dataset, codes_tf32.view()); });

      auto policy = raft::resource::get_thrust_policy(res);
      size_t n_codes = codes_simt.size();
      size_t n_diff  = n_codes - thrust::inner_product(policy,
                                                      codes_simt.data_handle(),
                                                      codes_simt.data_handle() + n_codes,
                                                      codes_tf32.data_handle(),
                                                      size_t{0},
                                                      thrust::plus<size_t>{},
                                                      thrust::equal_to<uint8_t>{});
      double err[2];
      for (int i = 0; i < 2; i++) {
        pq::inverse_transform(res,
                              quantizer,
                              raft::make_const_mdspan(i == 0 ? codes_simt.view() : codes_tf32.view()),
                              recon.view());
        err[i] = mse(res, recon.data_handle(), dataset.data_handle(), recon.size());
      }
      std::printf("%7u %6u %6ld | %10.2f %10.2f %6.2fx | %12.4f %12.5g %12.5g\n",
                  pq_bits, pq_dim, dim / pq_dim, simt, tf32, simt / tf32,
                  100.0 * n_diff / n_codes, err[0], err[1]);
      std::fflush(stdout);
    }
  }
}

void bench_scann(raft::device_resources const& res, dmat_view dataset, int reps)
{
  std::printf("\n== ScaNN build (n_leaves=1000, defaults otherwise) ==\n");
  std::printf("%7s %6s %6s | %10s %10s %7s | %14s\n",
              "pq_bits", "pq_len", "pq_dim", "simt_ms", "tf32_ms", "speedup", "resid_diff_%");
  for (auto [pq_bits, pq_len] : std::vector<std::pair<uint32_t, uint32_t>>{{8, 8}, {8, 4}, {8, 16}, {4, 8}}) {
    scann::index_params params;
    params.pq_bits = pq_bits;
    params.pq_dim  = pq_len;  // ScaNN's pq_dim is the subspace width
    std::vector<uint8_t> resid[2];
    double ms[2];
    for (int tf32 = 0; tf32 < 2; tf32++) {
      set_tf32(tf32);
      ms[tf32] = median_ms(res, reps, [&] {
        auto idx = scann::build(res, params, dataset);
        auto q   = idx.quantized_residuals();
        resid[tf32].assign(q.data_handle(), q.data_handle() + q.size());
      });
    }
    size_t diff = 0;
    for (size_t i = 0; i < resid[0].size(); i++) { diff += resid[0][i] != resid[1][i]; }
    std::printf("%7u %6u %6ld | %10.1f %10.1f %6.2fx | %14.4f\n",
                pq_bits, pq_len, dataset.extent(1) / pq_len, ms[0], ms[1], ms[0] / ms[1],
                100.0 * diff / resid[0].size());
    std::fflush(stdout);
  }
}

void bench_cagra_q(raft::device_resources const& res, dmat_view dataset, int reps)
{
  std::printf("\n== CAGRA-Q compression (pq::make_vpq_dataset) ==\n");
  std::printf("%7s %6s | %10s %10s %7s\n", "pq_bits", "pq_dim", "simt_ms", "tf32_ms", "speedup");
  for (uint32_t pq_dim : {256u, 128u}) {
    cuvs::neighbors::vpq_params params;
    params.pq_bits = 8;
    params.pq_dim  = pq_dim;
    double ms[2];
    for (int tf32 = 0; tf32 < 2; tf32++) {
      set_tf32(tf32);
      ms[tf32] = median_ms(res, reps, [&] { auto v = pq::make_vpq_dataset(res, params, dataset); });
    }
    std::printf("%7u %6u | %10.1f %10.1f %6.2fx\n", 8u, pq_dim, ms[0], ms[1], ms[0] / ms[1]);
    std::fflush(stdout);
  }
}

void bench_vamana(raft::device_resources const& res, dmat_view dataset, int reps)
{
  const int64_t dim = dataset.extent(1);
  std::printf("\n== Vamana build with PQ codebooks (graph_degree=32, identity rotation) ==\n");
  std::printf("%7s %6s %6s | %10s %10s %7s | %12s\n",
              "pq_bits", "pq_dim", "pq_len", "simt_ms", "tf32_ms", "speedup", "code_diff_%");
  for (uint32_t pq_dim : {256u, 128u, 64u}) {
    const int64_t pq_len = dim / pq_dim;
    pq::params p{
      8, pq_dim, true, false, 0, 25, cuvs::cluster::kmeans::kmeans_type::KMeansBalanced, 256, 1024};
    auto quantizer = pq::build(res, p, dataset);
    auto book = quantizer.vpq_codebooks.const_manifest_view().pq_code_book;  // [pq_dim * 256, pq_len]
    std::vector<float> h_book(book.size());
    raft::copy(h_book.data(), book.data_handle(), book.size(), res.get_stream());
    res.sync_stream();

    // DiskANN layout: center c of subspace s at [c][s * pq_len, (s + 1) * pq_len)
    cuvs::neighbors::vamana::codebook_params<float> cb{256, int(pq_dim), {}, {}};
    cb.pq_encoding_table.resize(256 * dim);
    for (int64_t s = 0; s < pq_dim; s++)
      for (int64_t c = 0; c < 256; c++)
        for (int64_t k = 0; k < pq_len; k++)
          cb.pq_encoding_table[c * dim + s * pq_len + k] = h_book[(s * 256 + c) * pq_len + k];
    cb.rotation_matrix.assign(dim * dim, 0.0f);
    for (int64_t i = 0; i < dim; i++) { cb.rotation_matrix[i * dim + i] = 1.0f; }

    cuvs::neighbors::vamana::index_params vp;
    vp.codebooks = cb;
    std::vector<uint8_t> codes[2];
    double ms[2];
    for (int tf32 = 0; tf32 < 2; tf32++) {
      set_tf32(tf32);
      ms[tf32] = median_ms(res, reps, [&] {
        auto idx = cuvs::neighbors::vamana::build(res, vp, dataset);
        auto q   = idx.quantized_data();
        codes[tf32].resize(q.size());
        raft::copy(codes[tf32].data(), q.data_handle(), q.size(), res.get_stream());
      });
    }
    size_t diff = 0;
    for (size_t i = 0; i < codes[0].size(); i++) { diff += codes[0][i] != codes[1][i]; }
    // Vamana's encoding step alone: per 65536-row batch, rotate (GEMM) then pq::transform
    const int64_t n = dataset.extent(0), batch = 65536;
    auto rot = raft::make_device_matrix<float, int64_t>(res, dim, dim);
    raft::copy(rot.data_handle(), cb.rotation_matrix.data(), rot.size(), res.get_stream());
    auto rotated = raft::make_device_matrix<float, int64_t>(res, batch, dim);
    auto out     = raft::make_device_matrix<uint8_t, int64_t>(res, n, pq::get_quantized_dim(p));
    double enc[2];
    for (int tf32 = 0; tf32 < 2; tf32++) {
      set_tf32(tf32);
      enc[tf32] = median_ms(res, 5, [&] {
        for (int64_t r = 0; r < n; r += batch) {
          const int64_t m = std::min(batch, n - r);
          auto src = raft::make_device_matrix_view<float, int64_t>(
            const_cast<float*>(dataset.data_handle()) + r * dim, m, dim);
          auto dst = raft::make_device_matrix_view<float, int64_t>(rotated.data_handle(), m, dim);
          raft::linalg::gemm(res, src, rot.view(), dst);
          pq::transform(res,
                        quantizer,
                        raft::make_const_mdspan(dst),
                        raft::make_device_matrix_view<uint8_t, int64_t>(
                          out.data_handle() + r * out.extent(1), m, out.extent(1)));
        }
      });
    }
    std::printf("%7u %6u %6ld | %10.1f %10.1f %6.2fx | %12.4f | encode step %7.1f -> %6.1f ms (%.2fx)\n",
                8u, pq_dim, pq_len, ms[0], ms[1], ms[0] / ms[1], 100.0 * diff / codes[0].size(),
                enc[0], enc[1], enc[0] / enc[1]);
    std::fflush(stdout);
  }
}

int main(int argc, char** argv)
{
  const std::string what = argc > 1 ? argv[1] : "all";
  const int64_t n_rows = 1000000, dim = 1024;
  raft::device_resources res;
  rmm::mr::pool_memory_resource pool(rmm::mr::get_current_device_resource_ref(), 40ull << 30);
  rmm::mr::set_current_device_resource(pool);

  auto dataset = raft::make_device_matrix<float, int64_t>(res, n_rows, dim);
  auto labels  = raft::make_device_vector<int64_t, int64_t>(res, n_rows);
  raft::random::make_blobs<float, int64_t>(res, dataset.view(), labels.view(), 1024);
  auto view = raft::make_const_mdspan(dataset.view());

  if (what == "all" || what == "pq") { bench_transform(res, view, 5); }
  if (what == "all" || what == "scann") { bench_scann(res, view, 3); }
  if (what == "all" || what == "cagra") { bench_cagra_q(res, view, 3); }
  if (what == "all" || what == "vamana") { bench_vamana(res, view, 3); }
  return 0;
}
