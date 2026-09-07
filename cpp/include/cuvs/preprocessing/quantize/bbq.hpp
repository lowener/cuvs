/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/core/export.hpp>
#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/common.hpp>

#include <raft/core/device_mdspan.hpp>

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace CUVS_EXPORT cuvs {

namespace preprocessing::quantize::bbq {

/**
 * @defgroup bbq Better Binary Quantization utilities
 * @{
 */

/**
 * Storage layout of BBQ/OSQ quantized component codes in each dataset row.
 *
 * `single_bit` Binary-packed bitstream (packAsBinary). Used for bits == 1.
 * `dibit` matches Lucene's transposeDibit. Used for bits == 2.
 * `unsigned_byte` one uint8_t per component. Used for bits == 7 and 8.
 */
enum class bbq_code_layout {
  single_bit, /** Each dimension is quantized to a single bit and packed into bytes. Reflects
               * OptimizedScalarQuantizer.packAsBinary. During query time, the query vector is
               * quantized to 4 bits per dimension. */
  dibit,      /** Each dimension is quantized to 2 bits (dibit) and packed into bytes. Reflects
               * OptimizedScalarQuantizer.transposeDibit. During query time, the query vector is
               * quantized to 4 bits per dimension. */
  transpose_half_byte, /** Each dimension is quantized to 4 bits, optimized for bitwise operations.
                        * Reflects OptimizedScalarQuantizer.transposeHalfByte. the first bit of
                        * every dimension is in the first set dimensions bits, or (dimensions/8)
                        * bytes. The second, third, and fourth bits are in the second, third, and
                        * fourth set of dimensions bits, respectively. Format used for queries. */
  packed_nibble, /** Each dimension is quantized to 4 bits, two values are packed into each output
                  * byte. Reflects OffHeapScalarQuantizedVectorValues.packNibbles. */
  seven_bit,     /** Each dimension is quantized to 7 bits and treated as a signed value. */
  unsigned_byte, /** Each dimension is quantized to 8 bits and treated as an unsigned value. */

};
template <typename DataT, typename IdxT>
struct bbq_quantizer {
  raft::device_mdarray<uint8_t, raft::matrix_extent<IdxT>> codes;
  raft::device_mdarray<float, raft::vector_extent<IdxT>> lower_intervals;
  raft::device_mdarray<float, raft::vector_extent<IdxT>> upper_intervals;
  raft::device_mdarray<float, raft::vector_extent<IdxT>> additional_corrections;
  raft::device_mdarray<int32_t, raft::vector_extent<IdxT>> quantized_component_sums;
  raft::device_mdarray<DataT, raft::vector_extent<IdxT>> centroid;

  uint32_t bits{};
  bbq_code_layout layout{bbq_code_layout::single_bit};
  cuvs::distance::DistanceType metric{cuvs::distance::DistanceType::L2Expanded};
  float centroid_norm_sq{};

  bbq_quantizer(
    raft::device_mdarray<uint8_t, raft::matrix_extent<IdxT>>&& codes_,
    raft::device_mdarray<float, raft::vector_extent<IdxT>>&& lower_intervals_,
    raft::device_mdarray<float, raft::vector_extent<IdxT>>&& upper_intervals_,
    raft::device_mdarray<float, raft::vector_extent<IdxT>>&& additional_corrections_,
    raft::device_mdarray<int32_t, raft::vector_extent<IdxT>>&& quantized_component_sums_,
    raft::device_mdarray<DataT, raft::vector_extent<IdxT>>&& centroid_,
    uint32_t bits,
    bbq_code_layout layout,
    cuvs::distance::DistanceType metric,
    float centroid_norm_sq)
    : codes{std::move(codes_)},
      lower_intervals{std::move(lower_intervals_)},
      upper_intervals{std::move(upper_intervals_)},
      additional_corrections{std::move(additional_corrections_)},
      quantized_component_sums{std::move(quantized_component_sums_)},
      centroid{std::move(centroid_)},
      bits{bits},
      layout{layout},
      metric{metric},
      centroid_norm_sq{centroid_norm_sq}
  {
    const auto n_rows = static_cast<int64_t>(codes.extent(0));
    RAFT_EXPECTS(bits >= 1 && bits <= 8, "BBQ bits must be in [1, 8].");
    RAFT_EXPECTS(codes.extent(1) == static_cast<int64_t>(encoded_row_length()),
                 "BBQ code row length does not match dim, bits, and layout.");
    RAFT_EXPECTS(lower_intervals.extent(0) == n_rows && upper_intervals.extent(0) == n_rows &&
                   additional_corrections.extent(0) == n_rows &&
                   quantized_component_sums.extent(0) == n_rows,
                 "Every BBQ correction array must contain one value per row.");
  }

  [[nodiscard]] auto n_rows() const noexcept -> IdxT { return codes.extent(0); }
  [[nodiscard]] auto dim() const noexcept -> uint32_t
  {
    return static_cast<uint32_t>(centroid.extent(0));
  }
  [[nodiscard]] constexpr auto encoded_row_length() const noexcept -> uint32_t
  {
    auto const d = dim();
    switch (layout) {
      case bbq_code_layout::single_bit: return (d * bits + 7) / 8;
      case bbq_code_layout::dibit: return bits * ((d + 7) / 8);
      case bbq_code_layout::packed_nibble: return (d + 1) / 2;
      case bbq_code_layout::seven_bit: return d;
      case bbq_code_layout::unsigned_byte: return d;
      case bbq_code_layout::transpose_half_byte: return 4 * ((d + 7) / 8);
    }
    return 0;
  }
};

template <typename DataT, typename IdxT>
struct bbq_quantizer_view {
  using owning_storage = bbq_quantizer<DataT, IdxT>;
  raft::device_mdspan<const uint8_t, raft::matrix_extent<IdxT>, raft::layout_c_contiguous> codes;
  raft::device_mdspan<const float, raft::vector_extent<IdxT>, raft::layout_c_contiguous>
    lower_intervals;
  raft::device_mdspan<const float, raft::vector_extent<IdxT>, raft::layout_c_contiguous>
    upper_intervals;
  raft::device_mdspan<const float, raft::vector_extent<IdxT>, raft::layout_c_contiguous>
    additional_corrections;
  raft::device_mdspan<const int32_t, raft::vector_extent<IdxT>, raft::layout_c_contiguous>
    quantized_component_sums;
  raft::device_mdspan<const DataT, raft::vector_extent<IdxT>, raft::layout_c_contiguous> centroid;

  uint32_t bits{};
  bbq_code_layout layout{bbq_code_layout::single_bit};
  cuvs::distance::DistanceType metric{cuvs::distance::DistanceType::L2Expanded};
  float centroid_norm_sq{};

  bbq_quantizer_view(
    raft::device_mdspan<const uint8_t, raft::matrix_extent<IdxT>, raft::layout_c_contiguous> codes_,
    raft::device_mdspan<const float, raft::vector_extent<IdxT>, raft::layout_c_contiguous>
      lower_intervals_,
    raft::device_mdspan<const float, raft::vector_extent<IdxT>, raft::layout_c_contiguous>
      upper_intervals_,
    raft::device_mdspan<const float, raft::vector_extent<IdxT>, raft::layout_c_contiguous>
      additional_corrections_,
    raft::device_mdspan<const int32_t, raft::vector_extent<IdxT>, raft::layout_c_contiguous>
      quantized_component_sums_,
    raft::device_mdspan<const DataT, raft::vector_extent<IdxT>, raft::layout_c_contiguous>
      centroid_,
    uint32_t bits_,
    bbq_code_layout layout_,
    cuvs::distance::DistanceType metric_,
    float centroid_norm_sq_) noexcept
    : codes{codes_.view()},
      lower_intervals{lower_intervals_.view()},
      upper_intervals{upper_intervals_.view()},
      additional_corrections{additional_corrections_.view()},
      quantized_component_sums{quantized_component_sums_.view()},
      centroid{centroid_.view()},
      bits{bits_},
      layout{layout_},
      metric{metric_},
      centroid_norm_sq{centroid_norm_sq_}
  {
  }

  bbq_quantizer_view(const owning_storage& quantizer) noexcept
    : codes{quantizer.codes.view()},
      lower_intervals{quantizer.lower_intervals.view()},
      upper_intervals{quantizer.upper_intervals.view()},
      additional_corrections{quantizer.additional_corrections.view()},
      quantized_component_sums{quantizer.quantized_component_sums.view()},
      centroid{quantizer.centroid.view()},
      bits{quantizer.bits},
      layout{quantizer.layout},
      metric{quantizer.metric},
      centroid_norm_sq{quantizer.centroid_norm_sq}
  {
  }

  [[nodiscard]] constexpr auto n_rows() const noexcept -> IdxT { return codes.extent(0); }
  [[nodiscard]] constexpr auto dim() const noexcept -> uint32_t
  {
    return static_cast<uint32_t>(centroid.extent(0));
  }
};

template <typename DataT, typename IdxT>
using device_bbq_quantizer_view = bbq_quantizer_view<DataT, IdxT>;

/** @} */  // end of bbq group

}  // namespace preprocessing::quantize::bbq

namespace neighbors {
struct bbq_dataset_container {
  template <typename DataT, typename IdxT>
  using owning_storage = cuvs::preprocessing::quantize::bbq::bbq_quantizer<DataT, IdxT>;
  template <typename DataT, typename IdxT>
  using view_storage = cuvs::preprocessing::quantize::bbq::bbq_quantizer_view<DataT, IdxT>;
};

template <typename DataT, typename IdxT, typename Accessor>
struct dataset<bbq_dataset_container, DataT, IdxT, Accessor> {
  using owning_storage_type = bbq_dataset_container::owning_storage<DataT, IdxT>;
  std::vector<owning_storage_type> quantizers;

  dataset(owning_storage_type&& quantizer) noexcept { add_quantizer(std::move(quantizer)); }
  [[nodiscard]] auto as_dataset_view() const noexcept
    -> dataset_view<bbq_dataset_container,
                    DataT,
                    IdxT,
                    detail::dataset_view_accessor_for_owning<DataT, Accessor>>
  {
    return dataset_view<bbq_dataset_container,
                        DataT,
                        IdxT,
                        detail::dataset_view_accessor_for_owning<DataT, Accessor>>{quantizers};
  }
  [[nodiscard]] constexpr auto n_rows() const noexcept -> IdxT
  {
    return quantizers.size() > 0 ? quantizers[0].n_rows() : 0;
  }
  [[nodiscard]] constexpr auto dim() const noexcept -> uint32_t
  {
    return quantizers.size() > 0 ? quantizers[0].dim() : 0;
  }

  void add_quantizer(owning_storage_type&& quantizer)
  {
    RAFT_EXPECTS(!has_bit_and_layout(quantizer.bits, quantizer.layout),
                 "Quantizer already exists with bits and layout.");
    this->quantizers.push_back(std::move(quantizer));
  }
  bool has_bit_and_layout(uint32_t bits,
                          cuvs::preprocessing::quantize::bbq::bbq_code_layout layout) const noexcept
  {
    for (uint32_t i = 0; i < quantizers.size(); i++) {
      if (quantizers[i].bits == bits && quantizers[i].layout == layout) { return true; }
    }
    return false;
  }
};

template <typename DataT, typename IdxT, typename Accessor>
struct dataset_view<bbq_dataset_container, DataT, IdxT, Accessor> {
  using owning_storage_type = bbq_dataset_container::owning_storage<DataT, IdxT>;
  using view_storage_type   = bbq_dataset_container::view_storage<DataT, IdxT>;
  std::vector<view_storage_type> quantizers;

  dataset_view() noexcept = default;

  dataset_view(const std::vector<owning_storage_type>& quantizers) noexcept
  {
    for (const auto& quantizer : quantizers) {
      add_quantizer(quantizer);
    }
  }
  [[nodiscard]] constexpr auto n_rows() const noexcept -> IdxT
  {
    return quantizers.size() > 0 ? quantizers[0].n_rows() : 0;
  }
  [[nodiscard]] constexpr auto dim() const noexcept -> uint32_t
  {
    return quantizers.size() > 0 ? quantizers[0].dim() : 0;
  }

  void add_quantizer(view_storage_type quantizer)
  {
    RAFT_EXPECTS(!has_bit_and_layout(quantizer.bits, quantizer.layout),
                 "Quantizer already exists with bits and layout.");
    this->quantizers.push_back(quantizer);
  }
  void add_quantizer(const owning_storage_type& quantizer)
  {
    RAFT_EXPECTS(!has_bit_and_layout(quantizer.bits, quantizer.layout),
                 "Quantizer already exists with bits and layout.");
    this->quantizers.push_back(view_storage_type(quantizer));
  }
  bool has_bit_and_layout(uint32_t bits,
                          cuvs::preprocessing::quantize::bbq::bbq_code_layout layout) const noexcept
  {
    for (uint32_t i = 0; i < quantizers.size(); i++) {
      if (quantizers[i].bits == bits && quantizers[i].layout == layout) { return true; }
    }
    return false;
  }
  view_storage_type get_quantizer(uint32_t bits,
                                  cuvs::preprocessing::quantize::bbq::bbq_code_layout layout) const
  {
    for (uint32_t i = 0; i < quantizers.size(); i++) {
      if (quantizers[i].bits == bits && quantizers[i].layout == layout) { return quantizers[i]; }
    }
    throw std::runtime_error("No quantizer found with bits and layout.");
  }
};

template <typename DataT, typename IdxT>
using device_bbq_dataset =
  dataset<bbq_dataset_container, DataT, IdxT, detail::device_owning_accessor<DataT>>;

template <typename DataT, typename IdxT>
using device_bbq_dataset_view =
  dataset_view<bbq_dataset_container, DataT, IdxT, detail::device_view_accessor<DataT>>;

template <typename DataT, typename IdxT>
struct owning_dataset_for_view<device_bbq_dataset_view<DataT, IdxT>> {
  using type = device_bbq_dataset<DataT, IdxT>;
};

template <typename DatasetT>
struct is_bbq_dataset : std::false_type {};

template <typename DataT, typename IdxT, typename Accessor>
struct is_bbq_dataset<dataset<bbq_dataset_container, DataT, IdxT, Accessor>> : std::true_type {};

template <typename DatasetT>
inline constexpr bool is_bbq_dataset_v = is_bbq_dataset<DatasetT>::value;

template <typename DataT, typename IdxT, typename Accessor>
struct dataset_view_kind_of<dataset_view<bbq_dataset_container, DataT, IdxT, Accessor>> {
  static constexpr dataset_view_kind value = dataset_view_kind::bbq;
};

template <typename DataT, typename IdxT>
struct cagra_view_element_type<device_bbq_dataset_view<DataT, IdxT>> {
  using type = DataT;
};

template <typename V>
inline constexpr bool is_device_bbq_dataset_view_v =
  dataset_view_kind_v<V> == dataset_view_kind::bbq && dataset_view_is_device_accessible_v<V>;

template <typename V>
inline constexpr bool is_host_bbq_dataset_view_v =
  dataset_view_kind_v<V> == dataset_view_kind::bbq && !dataset_view_is_device_accessible_v<V>;

template <typename V>
inline constexpr bool is_bbq_dataset_view_v =
  is_device_bbq_dataset_view_v<V> || is_host_bbq_dataset_view_v<V>;

}  // namespace neighbors

}  // namespace CUVS_EXPORT cuvs
