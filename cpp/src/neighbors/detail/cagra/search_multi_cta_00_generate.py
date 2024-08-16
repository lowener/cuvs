# Copyright (c) 2023-2024, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

header = """/*
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

/*
 * NOTE: this file is generated by search_multi_cta_00_generate.py
 *
 * Make changes there and run in this directory:
 *
 * > python search_multi_cta_00_generate.py
 *
 */

#include "search_multi_cta_inst.cuh"

#include "compute_distance.hpp"

namespace cuvs::neighbors::cagra::detail::multi_cta_search {
"""

trailer = """
}  // namespace cuvs::neighbors::cagra::detail::multi_cta_search
"""

# block = [(64, 16), (128, 8), (256, 4), (512, 2), (1024, 1)]
# mxelem = [64, 128, 256]
load_types = ["uint4"]
search_types = dict(
    float_uint32=(
        "float",
        "uint32_t",
        "float",
    ),  # data_t, vec_idx_t, distance_t
    half_uint32=("half", "uint32_t", "float"),
    int8_uint32=("int8_t", "uint32_t", "float"),
    uint8_uint32=("uint8_t", "uint32_t", "float"),
    float_uint64=("float", "uint64_t", "float"),
    half_uint64=("half", "uint64_t", "float"),
)
# knn
for type_path, (data_t, idx_t, distance_t) in search_types.items():
    path = f"search_multi_cta_{type_path}.cu"
    with open(path, "w") as f:
        f.write(header)
        f.write(
                f"instantiate_kernel_selection(\n  {data_t}, {idx_t}, {distance_t}, cuvs::neighbors::filtering::none_cagra_sample_filter);\n"
        )
        f.write(trailer)
        # For pasting into CMakeLists.txt
    print(f"src/neighbors/detail/cagra/{path}")
