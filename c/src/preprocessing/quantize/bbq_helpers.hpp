/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/preprocessing/quantize/bbq.h>
#include <cuvs/preprocessing/quantize/bbq.hpp>

#include <raft/core/error.hpp>

#include <cstddef>

namespace cuvs::preprocessing::quantize::bbq::c_api {

inline auto to_cpp_layout(cuvsBbqCodeLayout_t layout, char const* caller) -> bbq_code_layout
{
  switch (layout) {
    case CUVS_BBQ_CODE_LAYOUT_PACKED_1B: return bbq_code_layout::packed_1b;
    case CUVS_BBQ_CODE_LAYOUT_TRANSPOSED_2B: return bbq_code_layout::transposed_2b;
    case CUVS_BBQ_CODE_LAYOUT_TRANSPOSED_4B: return bbq_code_layout::transposed_4b;
    case CUVS_BBQ_CODE_LAYOUT_PACKED_4B: return bbq_code_layout::packed_4b;
    case CUVS_BBQ_CODE_LAYOUT_PACKED_7B: return bbq_code_layout::packed_7b;
    case CUVS_BBQ_CODE_LAYOUT_PACKED_8B: return bbq_code_layout::packed_8b;
  }
  RAFT_FAIL("%s: invalid BBQ code layout %d", caller, static_cast<int>(layout));
}

inline auto to_c_layout(bbq_code_layout layout) -> cuvsBbqCodeLayout_t
{
  switch (layout) {
    case bbq_code_layout::packed_1b: return CUVS_BBQ_CODE_LAYOUT_PACKED_1B;
    case bbq_code_layout::transposed_2b: return CUVS_BBQ_CODE_LAYOUT_TRANSPOSED_2B;
    case bbq_code_layout::transposed_4b: return CUVS_BBQ_CODE_LAYOUT_TRANSPOSED_4B;
    case bbq_code_layout::packed_4b: return CUVS_BBQ_CODE_LAYOUT_PACKED_4B;
    case bbq_code_layout::packed_7b: return CUVS_BBQ_CODE_LAYOUT_PACKED_7B;
    case bbq_code_layout::packed_8b: return CUVS_BBQ_CODE_LAYOUT_PACKED_8B;
  }
  RAFT_FAIL("invalid BBQ code layout %d", static_cast<int>(layout));
}

/**
 * Every `CUVS_DATASET_LAYOUT_BBQ` dataset the C API produces holds one of the layout sets accepted
 * here, whether it was assembled from views or encoded by a quantizer.
 */
inline void expect_supported_layout_set(bbq_code_layout const* layouts,
                                        std::size_t n_layouts,
                                        char const* caller)
{
  RAFT_EXPECTS(
    n_layouts == 1 || n_layouts == 2, "%s: a BBQ dataset holds one or two code layouts", caller);
  if (n_layouts == 1) {
    RAFT_EXPECTS(layouts[0] != bbq_code_layout::transposed_4b,
                 "%s: transposed_4b requires an asymmetric layout pair",
                 caller);
    return;
  }
  RAFT_EXPECTS(layouts[0] != layouts[1], "%s: the two BBQ code layouts must differ", caller);
  auto has = [&](bbq_code_layout layout) { return layouts[0] == layout || layouts[1] == layout; };
  const bool has_1b  = has(bbq_code_layout::packed_1b);
  const bool has_2bt = has(bbq_code_layout::transposed_2b);
  const bool has_4b  = has(bbq_code_layout::packed_4b);
  const bool has_4bt = has(bbq_code_layout::transposed_4b);
  RAFT_EXPECTS((has_1b && (has_4b || has_2bt || has_4bt)) || (has_2bt && has_4bt),
               "%s: unsupported asymmetric BBQ layout pair",
               caller);
}

}  // namespace cuvs::preprocessing::quantize::bbq::c_api
