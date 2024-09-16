/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include "device_common.hpp"
#include "hashmap.hpp"
#include "utils.hpp"

#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/common.hpp>
#include <raft/core/logger-macros.hpp>
#include <raft/core/operators.hpp>

// TODO: This shouldn't be invoking spatial/knn
#include "../ann_utils.cuh"

#include <raft/util/device_loads_stores.cuh>
#include <raft/util/vectorized.cuh>

#include <type_traits>

namespace cuvs::neighbors::cagra::detail {

template <typename DataT, typename IndexT, typename DistanceT>
struct alignas(device::LOAD_128BIT_T) dataset_descriptor_base_t {
  using base_type  = dataset_descriptor_base_t<DataT, IndexT, DistanceT>;
  using LOAD_T     = device::LOAD_128BIT_T;
  using DATA_T     = DataT;
  using INDEX_T    = IndexT;
  using DISTANCE_T = DistanceT;

  struct alignas(LOAD_T) args_t {
    void* extra_ptr1;
    void* extra_ptr2;
    /** Pointer to the workspace in the shared memory (filled in every copy by a thread block). */
    uint32_t smem_ws_ptr;
    /** Dimensionality of the data/queries. */
    uint32_t dim;
    uint32_t extra_word1;
    uint32_t extra_word2;

    RAFT_DEVICE_INLINE_FUNCTION auto load() const -> args_t
    {
      constexpr int kCount = sizeof(*this) / sizeof(LOAD_T);
      using blob_type      = LOAD_T[kCount];
      args_t r;
      auto& src = reinterpret_cast<const blob_type&>(*this);
      auto& dst = reinterpret_cast<blob_type&>(r);
#pragma unroll
      for (int i = 0; i < kCount; i++) {
        device::lds(dst[i], src + i);
      }
      return r;
    }
  };

  struct smem_and_team_size_t {
    uint32_t value;
    RAFT_INLINE_FUNCTION constexpr smem_and_team_size_t(uint32_t smem_size_bytes,
                                                        uint32_t team_size_bitshift)
      : value{(team_size_bitshift << 24) | smem_size_bytes}
    {
    }
    /** Total dynamic shared memory required by the descriptor.  */
    RAFT_INLINE_FUNCTION constexpr auto smem_ws_size_in_bytes() const noexcept -> uint32_t
    {
      return value & 0xffffffu;
    }
    RAFT_INLINE_FUNCTION constexpr auto team_size_bitshift() const noexcept -> uint32_t
    {
      return (value >> 24) & 0xffu;
    }
    /** How many threads are involved in computing a single distance. */
    RAFT_INLINE_FUNCTION constexpr auto team_size() const noexcept -> uint32_t
    {
      return 1u << team_size_bitshift();
    }
  };
  static_assert(sizeof(smem_and_team_size_t) == sizeof(uint32_t));

  using setup_workspace_type  = const base_type*(const base_type*, void*, const DATA_T*, uint32_t);
  using compute_distance_type = DISTANCE_T(const args_t, const INDEX_T);

  args_t args;

  /** Copy the descriptor and the query into shared memory and do any other work, such as
   * initializing the codebook. */
  setup_workspace_type* setup_workspace_impl;
  /** Compute the distance from the query vector (stored in the smem_workspace) and a dataset vector
   * given by the dataset_index. */
  compute_distance_type* compute_distance_impl;
  void* extra_ptr3;
  smem_and_team_size_t smem_and_team_size;

  /** Number of records in the database. */
  INDEX_T size;

  RAFT_INLINE_FUNCTION dataset_descriptor_base_t(setup_workspace_type* setup_workspace_impl,
                                                 compute_distance_type* compute_distance_impl,
                                                 INDEX_T size,
                                                 uint32_t dim,
                                                 uint32_t team_size_bitshift,
                                                 uint32_t smem_ws_size_in_bytes)
    : setup_workspace_impl(setup_workspace_impl),
      compute_distance_impl(compute_distance_impl),
      size(size),
      smem_and_team_size(smem_ws_size_in_bytes, team_size_bitshift),
      args{nullptr, nullptr, 0, dim, 0, 0}
  {
  }

  /** Total dynamic shared memory required by the descriptor.  */
  RAFT_INLINE_FUNCTION constexpr auto smem_ws_size_in_bytes() const noexcept -> uint32_t
  {
    return smem_and_team_size.smem_ws_size_in_bytes();
  }
  RAFT_INLINE_FUNCTION constexpr auto team_size_bitshift() const noexcept -> uint32_t
  {
    return smem_and_team_size.team_size_bitshift();
  }
  RAFT_DEVICE_INLINE_FUNCTION constexpr auto team_size_bitshift_from_smem() const noexcept
    -> uint32_t
  {
    uint32_t sts;
    raft::lds(sts, reinterpret_cast<const uint32_t*>(&smem_and_team_size));
    return reinterpret_cast<smem_and_team_size_t&>(sts).team_size_bitshift();
  }

  /** How many threads are involved in computing a single distance. */
  RAFT_INLINE_FUNCTION constexpr auto team_size() const noexcept -> uint32_t
  {
    return smem_and_team_size.team_size();
  }

  RAFT_DEVICE_INLINE_FUNCTION auto setup_workspace(void* smem_ptr,
                                                   const DATA_T* queries_ptr,
                                                   uint32_t query_id) const -> const base_type*
  {
    return setup_workspace_impl(this, smem_ptr, queries_ptr, query_id);
  }

  RAFT_DEVICE_INLINE_FUNCTION auto compute_distance(INDEX_T dataset_index, bool valid) const
    -> DISTANCE_T
  {
    auto per_thread_distances = valid ? compute_distance_impl(args.load(), dataset_index) : 0;
    return device::team_sum(per_thread_distances, team_size_bitshift_from_smem());
  }
};

template <typename DataT, typename IndexT, typename DistanceT>
struct dataset_descriptor_host {
  using dev_descriptor_t         = dataset_descriptor_base_t<DataT, IndexT, DistanceT>;
  dev_descriptor_t* dev_ptr      = nullptr;
  uint32_t smem_ws_size_in_bytes = 0;
  uint32_t team_size             = 0;
  uint32_t dataset_block_dim     = 0;

  template <typename DescriptorImpl>
  dataset_descriptor_host(const DescriptorImpl& dd_host,
                          rmm::cuda_stream_view stream,
                          uint32_t dataset_block_dim)
    : stream_{stream},
      smem_ws_size_in_bytes{dd_host.smem_ws_size_in_bytes()},
      team_size{dd_host.team_size()},
      dataset_block_dim{dataset_block_dim}
  {
    RAFT_CUDA_TRY(cudaMallocAsync(&dev_ptr, sizeof(DescriptorImpl), stream_));
  }

  ~dataset_descriptor_host() noexcept
  {
    if (dev_ptr == nullptr) { return; }
    RAFT_CUDA_TRY_NO_THROW(cudaFreeAsync(dev_ptr, stream_));
  }

  dataset_descriptor_host(dataset_descriptor_host&& other)
  {
    std::swap(this->dev_ptr, other.dev_ptr);
    std::swap(this->smem_ws_size_in_bytes, other.smem_ws_size_in_bytes);
    std::swap(this->stream_, other.stream_);
    std::swap(this->team_size, other.team_size);
    std::swap(this->dataset_block_dim, other.dataset_block_dim);
  }
  dataset_descriptor_host& operator=(dataset_descriptor_host&& b)
  {
    auto& a = *this;
    std::swap(a.dev_ptr, b.dev_ptr);
    std::swap(a.smem_ws_size_in_bytes, b.smem_ws_size_in_bytes);
    std::swap(a.stream_, b.stream_);
    std::swap(a.team_size, b.team_size);
    std::swap(a.dataset_block_dim, b.dataset_block_dim);
    return a;
  }
  dataset_descriptor_host(const dataset_descriptor_host&)            = delete;
  dataset_descriptor_host& operator=(const dataset_descriptor_host&) = delete;

 private:
  rmm::cuda_stream_view stream_;
};

template <typename DataT, typename IndexT, typename DistanceT, typename DatasetT>
using init_desc_type =
  dataset_descriptor_host<DataT, IndexT, DistanceT> (*)(const cagra::search_params&,
                                                        const DatasetT&,
                                                        cuvs::distance::DistanceType,
                                                        rmm::cuda_stream_view);

template <typename DataT, typename IndexT, typename DistanceT>
struct instance_spec {
  using data_type     = DataT;
  using index_type    = IndexT;
  using distance_type = DistanceT;
  using host_type     = dataset_descriptor_host<DataT, IndexT, DistanceT>;
  /** Use this to constrain the input dataset type. */
  template <typename DatasetT>
  constexpr static inline bool accepts_dataset()
  {
    return false;
  }
};

template <typename InstanceSpec,
          typename DataT,
          typename IndexT,
          typename DistanceT,
          typename DatasetT>
constexpr bool spec_sound = std::is_same_v<DataT, typename InstanceSpec::data_type> &&
                            std::is_same_v<IndexT, typename InstanceSpec::index_type> &&
                            std::is_same_v<DistanceT, typename InstanceSpec::distance_type> &&
                            InstanceSpec::template accepts_dataset<DatasetT>();

template <typename InstanceSpec,
          typename DataT,
          typename IndexT,
          typename DistanceT,
          typename DatasetT>
constexpr auto spec_match(const cagra::search_params& params,
                          const DatasetT& dataset,
                          cuvs::distance::DistanceType metric)
  -> std::tuple<init_desc_type<DataT, IndexT, DistanceT, DatasetT>, double>
{
  if constexpr (spec_sound<InstanceSpec, DataT, IndexT, DistanceT, DatasetT>) {
    return std::make_tuple(InstanceSpec::template init<DatasetT>,
                           InstanceSpec::template priority(params, dataset, metric));
  }
  return std::make_tuple(nullptr, -1.0);
}

template <typename... Specs>
struct instance_selector {
  template <typename DataT, typename IndexT, typename DistanceT, typename DatasetT>
  static auto select(const cagra::search_params&, const DatasetT&, cuvs::distance::DistanceType)
    -> std::tuple<init_desc_type<DataT, IndexT, DistanceT, DatasetT>, double>
  {
    return std::make_tuple(nullptr, -1.0);
  }
};

template <typename Spec, typename... Specs>
struct instance_selector<Spec, Specs...> {
  template <typename DataT, typename IndexT, typename DistanceT, typename DatasetT>
  static auto select(const cagra::search_params& params,
                     const DatasetT& dataset,
                     cuvs::distance::DistanceType metric)
    -> std::enable_if_t<spec_sound<Spec, DataT, IndexT, DistanceT, DatasetT>,
                        std::tuple<init_desc_type<DataT, IndexT, DistanceT, DatasetT>, double>>
  {
    auto s0 = spec_match<Spec, DataT, IndexT, DistanceT, DatasetT>(params, dataset, metric);
    auto ss = instance_selector<Specs...>::template select<DataT, IndexT, DistanceT, DatasetT>(
      params, dataset, metric);
    return std::get<1>(s0) >= std::get<1>(ss) ? s0 : ss;
  }

  template <typename DataT, typename IndexT, typename DistanceT, typename DatasetT>
  static auto select(const cagra::search_params& params,
                     const DatasetT& dataset,
                     cuvs::distance::DistanceType metric)
    -> std::enable_if_t<!spec_sound<Spec, DataT, IndexT, DistanceT, DatasetT>,
                        std::tuple<init_desc_type<DataT, IndexT, DistanceT, DatasetT>, double>>
  {
    return instance_selector<Specs...>::template select<DataT, IndexT, DistanceT, DatasetT>(
      params, dataset, metric);
  }
};

}  // namespace cuvs::neighbors::cagra::detail
