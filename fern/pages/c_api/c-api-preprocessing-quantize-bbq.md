---
slug: api-reference/c-api-preprocessing-quantize-bbq
---

# Bbq

_Source header: `cuvs/preprocessing/quantize/bbq.h`_

## C API for Better Binary Quantization datasets

<a id="cuvsbbqcodelayout-t"></a>
### cuvsBbqCodeLayout_t

Storage layout of BBQ/OSQ quantized component codes in each dataset row.

The numeric values are part of the ABI: they never change, and new layouts are only appended.

```c
typedef enum {
  CUVS_BBQ_CODE_LAYOUT_PACKED_1B = 0,
  CUVS_BBQ_CODE_LAYOUT_TRANSPOSED_2B = 1,
  CUVS_BBQ_CODE_LAYOUT_TRANSPOSED_4B = 2,
  CUVS_BBQ_CODE_LAYOUT_PACKED_4B = 3,
  CUVS_BBQ_CODE_LAYOUT_PACKED_7B = 4,
  CUVS_BBQ_CODE_LAYOUT_PACKED_8B = 5
} cuvsBbqCodeLayout_t;
```

**Values**

| Name | Value | Description |
| --- | --- | --- |
| `CUVS_BBQ_CODE_LAYOUT_PACKED_1B` | `0` | Each dimension is quantized to a single bit and packed into bytes. Reflects Lucene's OptimizedScalarQuantizer.packAsBinary. |
| `CUVS_BBQ_CODE_LAYOUT_TRANSPOSED_2B` | `1` | Each dimension is quantized to 2 bits, stored as 2 bitplanes. Reflects Lucene's OptimizedScalarQuantizer.transposeDibit. SIMT popc path only (paired with a transposed_4b or packed_1b operand); |
| `CUVS_BBQ_CODE_LAYOUT_TRANSPOSED_4B` | `2` | Each dimension is quantized to 4 bits, optimized for bitwise operations. Reflects Lucene's OptimizedScalarQuantizer.transposeHalfByte. the first bit of every dimension is in the first set dimensions bits, or (dimensions/8) bytes. The second, third, and fourth bits are in the second, third, and fourth set of dimensions bits, respectively. Format used for queries. |
| `CUVS_BBQ_CODE_LAYOUT_PACKED_4B` | `3` | Each dimension is quantized to 4 bits, two values are packed into each output byte. |
| `CUVS_BBQ_CODE_LAYOUT_PACKED_7B` | `4` | Each dimension is quantized to 7 bits and treated as a signed value. |
| `CUVS_BBQ_CODE_LAYOUT_PACKED_8B` | `5` | Each dimension is quantized to 8 bits and treated as an unsigned value. |

<a id="cuvsbbqquantizerparams"></a>
### cuvsBbqQuantizerParams

Parameters of a BBQ quantizer.

Always allocate this struct with `cuvsBbqQuantizerParamsCreate` and release it with `cuvsBbqQuantizerParamsDestroy`: new fields may be appended in later releases, so declaring it on the stack or with `malloc` is not ABI-stable.

The set of layouts a quantizer encodes must be one a BBQ dataset supports: a single layout (`doc_layout == query_layout`) other than `CUVS_BBQ_CODE_LAYOUT_TRANSPOSED_4B`, or one of the (document, query) pairs (`PACKED_1B`, `PACKED_4B`), (`PACKED_1B`, `TRANSPOSED_2B`), (`PACKED_1B`, `TRANSPOSED_4B`), (`TRANSPOSED_2B`, `TRANSPOSED_4B`).

```c
struct cuvsBbqQuantizerParams {
  cuvsBbqCodeLayout_t doc_layout;
  cuvsBbqCodeLayout_t query_layout;
  cuvsDistanceType metric;
  float lambda;
  uint32_t iters;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `doc_layout` | [`cuvsBbqCodeLayout_t`](/api-reference/c-api-preprocessing-quantize-bbq#cuvsbbqcodelayout-t) | Code layout of the encoded rows on the document side of a distance.<br />Default: `CUVS_BBQ_CODE_LAYOUT_PACKED_1B`. |
| `query_layout` | [`cuvsBbqCodeLayout_t`](/api-reference/c-api-preprocessing-quantize-bbq#cuvsbbqcodelayout-t) | Code layout of the encoded rows on the query side of a distance. When it differs from `doc_layout`, the dataset is encoded a second time in this layout.<br />Default: `CUVS_BBQ_CODE_LAYOUT_TRANSPOSED_4B`. |
| `metric` | [`cuvsDistanceType`](/api-reference/c-api-distance-distance#cuvsdistancetype) | Distance metric the encoded dataset is compared with. Selects the per-row additional correction: \|\|x - c\|\|^2 for `L2Expanded` and `L2SqrtExpanded`, x.c otherwise.<br />Default: `L2Expanded`. |
| `lambda` | `float` | Weight of the quantization error against the norm-preserving term of the interval loss, within [0, 1]. Default: 0.1. |
| `iters` | `uint32_t` | Coordinate-descent iterations refining the quantization interval of each row. Zero keeps the initial interval. Default: 5. |

<a id="cuvsbbqquantizerparamscreate"></a>
### cuvsBbqQuantizerParamsCreate

Allocate BBQ quantizer params, and populate them with default values.

```c
cuvsError_t cuvsBbqQuantizerParamsCreate(cuvsBbqQuantizerParams_t* params);
```

Release with `cuvsBbqQuantizerParamsDestroy`.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `params` | out | [`cuvsBbqQuantizerParams_t*`](/api-reference/c-api-preprocessing-quantize-bbq#cuvsbbqquantizerparams) | newly allocated params |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsbbqquantizerparamsdestroy"></a>
### cuvsBbqQuantizerParamsDestroy

De-allocate BBQ quantizer params.

```c
cuvsError_t cuvsBbqQuantizerParamsDestroy(cuvsBbqQuantizerParams_t params);
```

Accepts NULL.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `params` | in | [`cuvsBbqQuantizerParams_t`](/api-reference/c-api-preprocessing-quantize-bbq#cuvsbbqquantizerparams) | params to destroy |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsbbqquantizer"></a>
### cuvsBbqQuantizer

Better Binary Quantization ([BBQ](https://www.elastic.co/search-labs/blog/better-binary-quantization-lucene-elasticsearch)) is a vector-quantization approach used in Elasticsearch and Apache Lucene. It builds on ideas introduced in RaBitQ([Gao and Long](https://arxiv.org/pdf/2405.12497, [Gao et al.](https://arxiv.org/pdf/2409.09913)): residual binary codes around a centroid, corrective factors, and efficient bitwise comparison of codes at different bit widths. Lucene implements this as optimized scalar quantization (OSQ) with packed and bit-plane layouts; Elasticsearch exposes it as BBQ.

BBQ in cuVS designed to be compatible with the Lucene/Elasticsearch dataset: a single shared centroid, no random rotation, and OSQ codes.

RaBitQ and BBQ in cuVS both compress centroid-relative vectors to low-bit codes and retain additional per-vector information so search is better than naïve sign-bit comparison. They differ in transformation and scale representation. RaBitQ commonly separates residual magnitude from direction, then applies a random orthogonal rotation before binary coding; BBQ uses per-vector scalar intervals to interpret the compressed residual codes.

A handle is obtained in one of two ways, and `is_owning` tells them apart:

- `cuvsBbqQuantizerCreate` followed by `cuvsBbqQuantizerBuild` yields an owning quantizer: it holds the centroid it computed and its parameters, and encodes datasets with `cuvsBbqQuantizerTransform`.
- `cuvsBbqQuantizerCreateView` yields a non-owning quantizer over rows that were already encoded, for example by Lucene, to be assembled into a dataset with `cuvsDatasetMakeBbqView`.

Either kind is released with `cuvsBbqQuantizerDestroy`. The fields are managed by cuVS; read them through the `cuvsBbqQuantizerGet*` functions and never allocate or modify the struct directly.

```c
typedef struct cuvsBbqQuantizer {
  uintptr_t addr;
  DLDataType dtype;
  bool is_owning;
} cuvsBbqQuantizer;
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `addr` | `uintptr_t` | Opaque pointer to the implementation state. |
| `dtype` | `DLDataType` | Element type of the centroid. |
| `is_owning` | `bool` | Whether the handle owns the state it refers to, as opposed to viewing caller-owned tensors. |

<a id="cuvsbbqquantizer-void"></a>
### cuvsBbqQuantizer::void

Releases `addr`; called by `cuvsBbqQuantizerDestroy`.

```c
void (*destroy_addr)(void*);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `destroy_addr` |  | `*` |  |

**Returns**

`void`

<a id="cuvsbbqquantizercreate"></a>
### cuvsBbqQuantizerCreate

Allocate an empty, owning BBQ quantizer to be built with `cuvsBbqQuantizerBuild`.

```c
cuvsError_t cuvsBbqQuantizerCreate(cuvsBbqQuantizer_t* quantizer);
```

Release with `cuvsBbqQuantizerDestroy`.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `quantizer` | out | [`cuvsBbqQuantizer_t*`](/api-reference/c-api-preprocessing-quantize-bbq#cuvsbbqquantizer) | newly allocated quantizer handle |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsbbqquantizerbuild"></a>
### cuvsBbqQuantizerBuild

Build a BBQ quantizer from a dataset: compute the centroid shared by every row.

```c
cuvsError_t cuvsBbqQuantizerBuild(cuvsResources_t res,
cuvsBbqQuantizerParams_t params,
DLManagedTensor* dataset,
cuvsBbqQuantizer_t quantizer);
```

The quantizer keeps a copy of `params`, which `cuvsBbqQuantizerTransform` encodes with. This function synchronizes the stream of `res`.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | cuVS resources |
| `params` | in | [`cuvsBbqQuantizerParams_t`](/api-reference/c-api-preprocessing-quantize-bbq#cuvsbbqquantizerparams) | quantizer parameters |
| `dataset` | in | `DLManagedTensor*` | non-empty, row-major float32 matrix in device-accessible memory |
| `quantizer` | in,out | [`cuvsBbqQuantizer_t`](/api-reference/c-api-preprocessing-quantize-bbq#cuvsbbqquantizer) | empty quantizer from `cuvsBbqQuantizerCreate`; a quantizer can only be built once |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsbbqquantizertransform"></a>
### cuvsBbqQuantizerTransform

Encode a dataset with a built BBQ quantizer into a new owning BBQ dataset.

```c
cuvsError_t cuvsBbqQuantizerTransform(cuvsResources_t res,
cuvsBbqQuantizer_t quantizer,
DLManagedTensor* dataset,
cuvsDataset_t bbq_dataset);
```

The rows are encoded in the quantizer's `doc_layout`, and again in its `query_layout` when it differs. The resulting dataset has layout `CUVS_DATASET_LAYOUT_BBQ`, resides in device memory, owns a copy of the centroid, and does not reference `quantizer` or `dataset` afterwards. The work is ordered on the stream of `res`.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | cuVS resources |
| `quantizer` | in | [`cuvsBbqQuantizer_t`](/api-reference/c-api-preprocessing-quantize-bbq#cuvsbbqquantizer) | quantizer built with `cuvsBbqQuantizerBuild` |
| `dataset` | in | `DLManagedTensor*` | non-empty, row-major float32 matrix in device-accessible memory, with as many columns as the quantizer's dimension |
| `bbq_dataset` | in,out | `cuvsDataset_t` | empty dataset handle from `cuvsDatasetCreate`, filled with the encoded dataset; release it with `cuvsDatasetDestroy` |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsbbqquantizergetdim"></a>
### cuvsBbqQuantizerGetDim

Get the dimension of the vectors a BBQ quantizer encodes.

```c
cuvsError_t cuvsBbqQuantizerGetDim(cuvsBbqQuantizer_t quantizer, uint32_t* dim);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `quantizer` | in | [`cuvsBbqQuantizer_t`](/api-reference/c-api-preprocessing-quantize-bbq#cuvsbbqquantizer) | built or view quantizer |
| `dim` | out | `uint32_t*` | number of components of each vector |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsbbqquantizergetcentroid"></a>
### cuvsBbqQuantizerGetCentroid

Get a non-owning view of the centroid of a BBQ quantizer.

```c
cuvsError_t cuvsBbqQuantizerGetCentroid(cuvsBbqQuantizer_t quantizer,
DLManagedTensor* centroid);
```

The tensor points to device memory that stays valid as long as `quantizer` (or, for a view quantizer, the caller's centroid tensor) does. Call `centroid->deleter` once done with it to release the tensor's shape metadata.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `quantizer` | in | [`cuvsBbqQuantizer_t`](/api-reference/c-api-preprocessing-quantize-bbq#cuvsbbqquantizer) | built or view quantizer |
| `centroid` | out | `DLManagedTensor*` | one-dimensional device tensor of `dim` elements of the quantizer's dtype |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsbbqquantizergetcentroidnormsq"></a>
### cuvsBbqQuantizerGetCentroidNormSq

Get the squared L2 norm of the centroid of a BBQ quantizer.

```c
cuvsError_t cuvsBbqQuantizerGetCentroidNormSq(cuvsBbqQuantizer_t quantizer,
float* centroid_norm_sq);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `quantizer` | in | [`cuvsBbqQuantizer_t`](/api-reference/c-api-preprocessing-quantize-bbq#cuvsbbqquantizer) | built or view quantizer |
| `centroid_norm_sq` | out | `float*` | squared norm of the centroid |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsbbqgetencodedrowlength"></a>
### cuvsBbqGetEncodedRowLength

Get the number of bytes one encoded row occupies in a code layout.

```c
cuvsError_t cuvsBbqGetEncodedRowLength(uint32_t dim,
cuvsBbqCodeLayout_t layout,
uint32_t* row_length);
```

Gives the column count of the `codes` matrix passed to `cuvsBbqQuantizerCreateView`.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `dim` | in | `uint32_t` | number of components of each vector |
| `layout` | in | [`cuvsBbqCodeLayout_t`](/api-reference/c-api-preprocessing-quantize-bbq#cuvsbbqcodelayout-t) | code layout |
| `row_length` | out | `uint32_t*` | bytes per encoded row |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsbbqquantizercreateview"></a>
### cuvsBbqQuantizerCreateView

Create a BBQ quantizer view from caller-owned device tensors.

```c
cuvsError_t cuvsBbqQuantizerCreateView(
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
```

Tensors are not copied and must remain valid while a derived dataset is in use.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `codes` | in | `DLManagedTensor*` | uint8 device matrix containing encoded rows |
| `lower_intervals` | in | `DLManagedTensor*` | float32 device vector with one lower interval per row |
| `upper_intervals` | in | `DLManagedTensor*` | float32 device vector with one upper interval per row |
| `additional_corrections` | in | `DLManagedTensor*` | float32 device vector with one correction per row |
| `quantized_component_sums` | in | `DLManagedTensor*` | int32 device vector with one component sum per row |
| `centroid` | in | `DLManagedTensor*` | device vector containing the dataset centroid |
| `dequant_delta` | in | `DLManagedTensor*` | float32 device vector with one dequantization delta per row |
| `dequant_sum_delta` | in | `DLManagedTensor*` | float32 device vector with one delta-times-sum value per row |
| `row_norm` | in | `DLManagedTensor*` | float32 device vector with one original-space squared norm per row |
| `layout` | in | [`cuvsBbqCodeLayout_t`](/api-reference/c-api-preprocessing-quantize-bbq#cuvsbbqcodelayout-t) | encoded code layout |
| `metric` | in | [`cuvsDistanceType`](/api-reference/c-api-distance-distance#cuvsdistancetype) | distance metric associated with the encoded dataset |
| `centroid_norm_sq` | in | `float` | squared norm of the centroid |
| `quantizer` | out | [`cuvsBbqQuantizer_t*`](/api-reference/c-api-preprocessing-quantize-bbq#cuvsbbqquantizer) | newly allocated non-owning quantizer handle |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsbbqquantizerdestroy"></a>
### cuvsBbqQuantizerDestroy

Destroy a BBQ quantizer.

```c
cuvsError_t cuvsBbqQuantizerDestroy(cuvsBbqQuantizer_t quantizer);
```

Releases the state of a quantizer from `cuvsBbqQuantizerCreate`; the caller-owned tensors of a quantizer from `cuvsBbqQuantizerCreateView` are left untouched. Accepts NULL.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `quantizer` | in | [`cuvsBbqQuantizer_t`](/api-reference/c-api-preprocessing-quantize-bbq#cuvsbbqquantizer) | quantizer handle to destroy |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsdatasetmakebbqview"></a>
### cuvsDatasetMakeBbqView

Create a non-owning device BBQ dataset view.

```c
cuvsError_t cuvsDatasetMakeBbqView(cuvsResources_t res,
cuvsBbqQuantizer_t* quantizers,
size_t num_quantizers,
cuvsDataset_t* dataset);
```

Accepts one symmetric quantizer or two compatible asymmetric quantizers, all created with `cuvsBbqQuantizerCreateView`. A dataset encoded by `cuvsBbqQuantizerTransform` is already a BBQ dataset and does not go through this function.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | cuVS resources |
| `quantizers` | in | [`cuvsBbqQuantizer_t*`](/api-reference/c-api-preprocessing-quantize-bbq#cuvsbbqquantizer) | array containing one or two BBQ quantizer handles |
| `num_quantizers` | in | `size_t` | number of elements in `quantizers` |
| `dataset` | out | `cuvsDataset_t*` | newly allocated non-owning BBQ dataset handle |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)
