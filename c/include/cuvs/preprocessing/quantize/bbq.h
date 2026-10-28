/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/core/c_api.h>
#include <cuvs/core/dataset.h>
#include <cuvs/distance/distance.h>

#include <dlpack/dlpack.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup preprocessing_c_bbq C API for Better Binary Quantization datasets
 * @{
 */

/**
 * Storage layout of BBQ/OSQ quantized component codes in each dataset row.
 *
 * The numeric values are part of the ABI: they never change, and new layouts are only appended.
 *
 * CUVS_BBQ_CODE_LAYOUT_PACKED_1B: Each dimension is quantized to a single bit and packed into bytes. Reflects
 * Lucene's OptimizedScalarQuantizer.packAsBinary.
 * CUVS_BBQ_CODE_LAYOUT_TRANSPOSED_2B: Each dimension is quantized to 2 bits, stored as 2 bitplanes.
 * Reflects Lucene's OptimizedScalarQuantizer.transposeDibit. SIMT popc path only
 * (paired with a transposed_4b or packed_1b operand);
 * CUVS_BBQ_CODE_LAYOUT_TRANSPOSED_4B: Each dimension is quantized to 4 bits, optimized for bitwise operations.
 * Reflects Lucene's OptimizedScalarQuantizer.transposeHalfByte. the first bit of
 * every dimension is in the first set dimensions bits, or (dimensions/8)
 * bytes. The second, third, and fourth bits are in the second, third, and
 * fourth set of dimensions bits, respectively. Format used for queries.
 * CUVS_BBQ_CODE_LAYOUT_PACKED_4B: Each dimension is quantized to 4 bits, two values are packed into each output
 * byte.
 * CUVS_BBQ_CODE_LAYOUT_PACKED_7B: Each dimension is quantized to 7 bits and treated as a signed value.
 * CUVS_BBQ_CODE_LAYOUT_PACKED_8B: Each dimension is quantized to 8 bits and treated as an unsigned value.
 */
typedef enum {
  CUVS_BBQ_CODE_LAYOUT_PACKED_1B     = 0,
  CUVS_BBQ_CODE_LAYOUT_TRANSPOSED_2B = 1,
  CUVS_BBQ_CODE_LAYOUT_TRANSPOSED_4B = 2,
  CUVS_BBQ_CODE_LAYOUT_PACKED_4B     = 3,
  CUVS_BBQ_CODE_LAYOUT_PACKED_7B     = 4,
  CUVS_BBQ_CODE_LAYOUT_PACKED_8B     = 5
} cuvsBbqCodeLayout_t;

/**
 * @brief Parameters of a BBQ quantizer.
 *
 * Always allocate this struct with `cuvsBbqQuantizerParamsCreate` and release it with
 * `cuvsBbqQuantizerParamsDestroy`: new fields may be appended in later releases, so declaring it on
 * the stack or with `malloc` is not ABI-stable.
 *
 * The set of layouts a quantizer encodes must be one a BBQ dataset supports: a single layout
 * (`doc_layout == query_layout`) other than `CUVS_BBQ_CODE_LAYOUT_TRANSPOSED_4B`, or one of the
 * (document, query) pairs (`PACKED_1B`, `PACKED_4B`), (`PACKED_1B`, `TRANSPOSED_2B`),
 * (`PACKED_1B`, `TRANSPOSED_4B`), (`TRANSPOSED_2B`, `TRANSPOSED_4B`).
 */
struct cuvsBbqQuantizerParams {
  /**
   * Code layout of the encoded rows on the document side of a distance.
   * Default: `CUVS_BBQ_CODE_LAYOUT_PACKED_1B`.
   */
  cuvsBbqCodeLayout_t doc_layout;
  /**
   * Code layout of the encoded rows on the query side of a distance. When it differs from
   * `doc_layout`, the dataset is encoded a second time in this layout.
   * Default: `CUVS_BBQ_CODE_LAYOUT_TRANSPOSED_4B`.
   */
  cuvsBbqCodeLayout_t query_layout;
  /**
   * Distance metric the encoded dataset is compared with. Selects the per-row additional
   * correction: ||x - c||^2 for `L2Expanded` and `L2SqrtExpanded`, x.c otherwise.
   * Default: `L2Expanded`.
   */
  cuvsDistanceType metric;
  /**
   * Weight of the quantization error against the norm-preserving term of the interval loss, within
   * [0, 1]. Default: 0.1.
   */
  float lambda;
  /**
   * Coordinate-descent iterations refining the quantization interval of each row. Zero keeps the
   * initial interval. Default: 5.
   */
  uint32_t iters;
};

typedef struct cuvsBbqQuantizerParams* cuvsBbqQuantizerParams_t;

/**
 * @brief Allocate BBQ quantizer params, and populate them with default values.
 *
 * Release with `cuvsBbqQuantizerParamsDestroy`.
 *
 * @param[out] params newly allocated params
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsBbqQuantizerParamsCreate(cuvsBbqQuantizerParams_t* params);

/**
 * @brief De-allocate BBQ quantizer params.
 *
 * Accepts NULL.
 *
 * @param[in] params params to destroy
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsBbqQuantizerParamsDestroy(cuvsBbqQuantizerParams_t params);

/**
 * @brief Better Binary Quantization
 * ([BBQ](https://www.elastic.co/search-labs/blog/better-binary-quantization-lucene-elasticsearch))
 * is a vector-quantization approach used in Elasticsearch and Apache Lucene. It builds on ideas
 * introduced in RaBitQ([Gao and Long](https://arxiv.org/pdf/2405.12497, [Gao et
 * al.](https://arxiv.org/pdf/2409.09913)): residual binary codes around a centroid, corrective
 * factors, and efficient bitwise comparison of codes at different bit widths. Lucene implements
 * this as optimized scalar quantization (OSQ) with packed and bit-plane layouts; Elasticsearch
 * exposes it as BBQ.
 *
 * BBQ in cuVS designed to be compatible with the Lucene/Elasticsearch dataset: a single shared
 * centroid, no random rotation, and OSQ codes.
 *
 * RaBitQ and BBQ in cuVS both compress centroid-relative vectors to low-bit codes and retain
 * additional per-vector information so search is better than naïve sign-bit comparison. They differ
 * in transformation and scale representation. RaBitQ commonly separates residual magnitude from
 * direction, then applies a random orthogonal rotation before binary coding; BBQ uses per-vector
 * scalar intervals to interpret the compressed residual codes.
 *
 * A handle is obtained in one of two ways, and `is_owning` tells them apart:
 * - `cuvsBbqQuantizerCreate` followed by `cuvsBbqQuantizerBuild` yields an owning quantizer: it
 *   holds the centroid it computed and its parameters, and encodes datasets with
 *   `cuvsBbqQuantizerTransform`.
 * - `cuvsBbqQuantizerCreateView` yields a non-owning quantizer over rows that were already encoded,
 *   for example by Lucene, to be assembled into a dataset with `cuvsDatasetMakeBbqView`.
 *
 * Either kind is released with `cuvsBbqQuantizerDestroy`. The fields are managed by cuVS; read
 * them through the `cuvsBbqQuantizerGet*` functions and never allocate or modify the struct
 * directly.
 */
typedef struct cuvsBbqQuantizer {
  /** Opaque pointer to the implementation state. */
  uintptr_t addr;
  /** Releases `addr`; called by `cuvsBbqQuantizerDestroy`. */
  void (*destroy_addr)(void*);
  /** Element type of the centroid. */
  DLDataType dtype;
  /** Whether the handle owns the state it refers to, as opposed to viewing caller-owned tensors. */
  bool is_owning;
} cuvsBbqQuantizer;
typedef cuvsBbqQuantizer* cuvsBbqQuantizer_t;

/**
 * @brief Allocate an empty, owning BBQ quantizer to be built with `cuvsBbqQuantizerBuild`.
 *
 * Release with `cuvsBbqQuantizerDestroy`.
 *
 * @param[out] quantizer newly allocated quantizer handle
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsBbqQuantizerCreate(cuvsBbqQuantizer_t* quantizer);

/**
 * @brief Build a BBQ quantizer from a dataset: compute the centroid shared by every row.
 *
 * The quantizer keeps a copy of `params`, which `cuvsBbqQuantizerTransform` encodes with. This
 * function synchronizes the stream of `res`.
 *
 * @code{.c}
 * #include <cuvs/core/c_api.h>
 * #include <cuvs/core/dataset.h>
 * #include <cuvs/preprocessing/quantize/bbq.h>
 *
 * cuvsResources_t res;
 * cuvsResourcesCreate(&res);
 *
 * // A row-major float32 matrix in device memory
 * DLManagedTensor *dataset;
 *
 * cuvsBbqQuantizerParams_t params;
 * cuvsBbqQuantizerParamsCreate(&params);
 *
 * cuvsBbqQuantizer_t quantizer;
 * cuvsBbqQuantizerCreate(&quantizer);
 * cuvsBbqQuantizerBuild(res, params, dataset, quantizer);
 *
 * cuvsDataset_t bbq_dataset;
 * cuvsDatasetCreate(&bbq_dataset);
 * cuvsBbqQuantizerTransform(res, quantizer, dataset, bbq_dataset);
 *
 * // bbq_dataset can now be passed to cuvsCagraBuild
 *
 * cuvsDatasetDestroy(bbq_dataset);
 * cuvsBbqQuantizerDestroy(quantizer);
 * cuvsBbqQuantizerParamsDestroy(params);
 * cuvsResourcesDestroy(res);
 * @endcode
 *
 * @param[in] res cuVS resources
 * @param[in] params quantizer parameters
 * @param[in] dataset non-empty, row-major float32 matrix in device-accessible memory
 * @param[in,out] quantizer empty quantizer from `cuvsBbqQuantizerCreate`; a quantizer can only be
 * built once
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsBbqQuantizerBuild(cuvsResources_t res,
                                              cuvsBbqQuantizerParams_t params,
                                              DLManagedTensor* dataset,
                                              cuvsBbqQuantizer_t quantizer);

/**
 * @brief Encode a dataset with a built BBQ quantizer into a new owning BBQ dataset.
 *
 * The rows are encoded in the quantizer's `doc_layout`, and again in its `query_layout` when it
 * differs. The resulting dataset has layout `CUVS_DATASET_LAYOUT_BBQ`, resides in device memory,
 * owns a copy of the centroid, and does not reference `quantizer` or `dataset` afterwards. The work
 * is ordered on the stream of `res`.
 *
 * @param[in] res cuVS resources
 * @param[in] quantizer quantizer built with `cuvsBbqQuantizerBuild`
 * @param[in] dataset non-empty, row-major float32 matrix in device-accessible memory, with as many
 * columns as the quantizer's dimension
 * @param[in,out] bbq_dataset empty dataset handle from `cuvsDatasetCreate`, filled with the encoded
 * dataset; release it with `cuvsDatasetDestroy`
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsBbqQuantizerTransform(cuvsResources_t res,
                                                  cuvsBbqQuantizer_t quantizer,
                                                  DLManagedTensor* dataset,
                                                  cuvsDataset_t bbq_dataset);

/**
 * @brief Get the dimension of the vectors a BBQ quantizer encodes.
 *
 * @param[in] quantizer built or view quantizer
 * @param[out] dim number of components of each vector
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsBbqQuantizerGetDim(cuvsBbqQuantizer_t quantizer, uint32_t* dim);

/**
 * @brief Get a non-owning view of the centroid of a BBQ quantizer.
 *
 * The tensor points to device memory that stays valid as long as `quantizer` (or, for a view
 * quantizer, the caller's centroid tensor) does. Call `centroid->deleter` once done with it to
 * release the tensor's shape metadata.
 *
 * @param[in] quantizer built or view quantizer
 * @param[out] centroid one-dimensional device tensor of `dim` elements of the quantizer's dtype
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsBbqQuantizerGetCentroid(cuvsBbqQuantizer_t quantizer,
                                                    DLManagedTensor* centroid);

/**
 * @brief Get the squared L2 norm of the centroid of a BBQ quantizer.
 *
 * @param[in] quantizer built or view quantizer
 * @param[out] centroid_norm_sq squared norm of the centroid
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsBbqQuantizerGetCentroidNormSq(cuvsBbqQuantizer_t quantizer,
                                                          float* centroid_norm_sq);

/**
 * @brief Get the number of bytes one encoded row occupies in a code layout.
 *
 * Gives the column count of the `codes` matrix passed to `cuvsBbqQuantizerCreateView`.
 *
 * @param[in] dim number of components of each vector
 * @param[in] layout code layout
 * @param[out] row_length bytes per encoded row
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsBbqGetEncodedRowLength(uint32_t dim,
                                                   cuvsBbqCodeLayout_t layout,
                                                   uint32_t* row_length);

/**
 * @brief Create a BBQ quantizer view from caller-owned device tensors.
 *
 * Tensors are not copied and must remain valid while a derived dataset is in use.
 *
 * @param[in] codes uint8 device matrix containing encoded rows
 * @param[in] lower_intervals float32 device vector with one lower interval per row
 * @param[in] upper_intervals float32 device vector with one upper interval per row
 * @param[in] additional_corrections float32 device vector with one correction per row
 * @param[in] quantized_component_sums int32 device vector with one component sum per row
 * @param[in] centroid device vector containing the dataset centroid
 * @param[in] dequant_delta float32 device vector with one dequantization delta per row
 * @param[in] dequant_sum_delta float32 device vector with one delta-times-sum value per row
 * @param[in] row_norm float32 device vector with one original-space squared norm per row
 * @param[in] layout encoded code layout
 * @param[in] metric distance metric associated with the encoded dataset
 * @param[in] centroid_norm_sq squared norm of the centroid
 * @param[out] quantizer newly allocated non-owning quantizer handle
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsBbqQuantizerCreateView(
  DLManagedTensor* codes,
  DLManagedTensor* lower_intervals,
  DLManagedTensor* upper_intervals,
  DLManagedTensor* additional_corrections,
  DLManagedTensor* quantized_component_sums,
  DLManagedTensor* centroid,
  DLManagedTensor* dequant_delta,
  DLManagedTensor* dequant_sum_delta,
  DLManagedTensor* row_norm,
  cuvsBbqCodeLayout_t layout,
  cuvsDistanceType metric,
  float centroid_norm_sq,
  cuvsBbqQuantizer_t* quantizer);

/**
 * @brief Destroy a BBQ quantizer.
 *
 * Releases the state of a quantizer from `cuvsBbqQuantizerCreate`; the caller-owned tensors of a
 * quantizer from `cuvsBbqQuantizerCreateView` are left untouched. Accepts NULL.
 *
 * @param[in] quantizer quantizer handle to destroy
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsBbqQuantizerDestroy(cuvsBbqQuantizer_t quantizer);

/**
 * @brief Create a non-owning device BBQ dataset view.
 *
 * Accepts one symmetric quantizer or two compatible asymmetric quantizers, all created with
 * `cuvsBbqQuantizerCreateView`. A dataset encoded by `cuvsBbqQuantizerTransform` is already a BBQ
 * dataset and does not go through this function.
 *
 * @param[in] res cuVS resources
 * @param[in] quantizers array containing one or two BBQ quantizer handles
 * @param[in] num_quantizers number of elements in `quantizers`
 * @param[out] dataset newly allocated non-owning BBQ dataset handle
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsDatasetMakeBbqView(cuvsResources_t res,
                                               cuvsBbqQuantizer_t* quantizers,
                                               size_t num_quantizers,
                                               cuvsDataset_t* dataset);

/** @} */

#ifdef __cplusplus
}
#endif
