/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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
 * NOTE: this file is generated by compute_distance_00_generate.py
 *
 * Make changes there and run in this directory:
 *
 * > python compute_distance_00_generate.py
 *
 */

#include "compute_distance-ext.cuh"

namespace cuvs::neighbors::cagra::detail {

template struct instance_selector<
  standard_descriptor_spec<8, 128, float, uint32_t, float>,
  vpq_descriptor_spec<8, 128, 8, 2, half, float, uint32_t, float>,
  vpq_descriptor_spec<8, 128, 8, 4, half, float, uint32_t, float>,
  standard_descriptor_spec<16, 256, float, uint32_t, float>,
  vpq_descriptor_spec<16, 256, 8, 2, half, float, uint32_t, float>,
  vpq_descriptor_spec<16, 256, 8, 4, half, float, uint32_t, float>,
  standard_descriptor_spec<32, 512, float, uint32_t, float>,
  vpq_descriptor_spec<32, 512, 8, 2, half, float, uint32_t, float>,
  vpq_descriptor_spec<32, 512, 8, 4, half, float, uint32_t, float>,
  standard_descriptor_spec<8, 128, half, uint32_t, float>,
  vpq_descriptor_spec<8, 128, 8, 2, half, half, uint32_t, float>,
  vpq_descriptor_spec<8, 128, 8, 4, half, half, uint32_t, float>,
  standard_descriptor_spec<16, 256, half, uint32_t, float>,
  vpq_descriptor_spec<16, 256, 8, 2, half, half, uint32_t, float>,
  vpq_descriptor_spec<16, 256, 8, 4, half, half, uint32_t, float>,
  standard_descriptor_spec<32, 512, half, uint32_t, float>,
  vpq_descriptor_spec<32, 512, 8, 2, half, half, uint32_t, float>,
  vpq_descriptor_spec<32, 512, 8, 4, half, half, uint32_t, float>,
  standard_descriptor_spec<8, 128, int8_t, uint32_t, float>,
  vpq_descriptor_spec<8, 128, 8, 2, half, int8_t, uint32_t, float>,
  vpq_descriptor_spec<8, 128, 8, 4, half, int8_t, uint32_t, float>,
  standard_descriptor_spec<16, 256, int8_t, uint32_t, float>,
  vpq_descriptor_spec<16, 256, 8, 2, half, int8_t, uint32_t, float>,
  vpq_descriptor_spec<16, 256, 8, 4, half, int8_t, uint32_t, float>,
  standard_descriptor_spec<32, 512, int8_t, uint32_t, float>,
  vpq_descriptor_spec<32, 512, 8, 2, half, int8_t, uint32_t, float>,
  vpq_descriptor_spec<32, 512, 8, 4, half, int8_t, uint32_t, float>,
  standard_descriptor_spec<8, 128, uint8_t, uint32_t, float>,
  vpq_descriptor_spec<8, 128, 8, 2, half, uint8_t, uint32_t, float>,
  vpq_descriptor_spec<8, 128, 8, 4, half, uint8_t, uint32_t, float>,
  standard_descriptor_spec<16, 256, uint8_t, uint32_t, float>,
  vpq_descriptor_spec<16, 256, 8, 2, half, uint8_t, uint32_t, float>,
  vpq_descriptor_spec<16, 256, 8, 4, half, uint8_t, uint32_t, float>,
  standard_descriptor_spec<32, 512, uint8_t, uint32_t, float>,
  vpq_descriptor_spec<32, 512, 8, 2, half, uint8_t, uint32_t, float>,
  vpq_descriptor_spec<32, 512, 8, 4, half, uint8_t, uint32_t, float>,
  standard_descriptor_spec<8, 128, float, uint64_t, float>,
  vpq_descriptor_spec<8, 128, 8, 2, half, float, uint64_t, float>,
  vpq_descriptor_spec<8, 128, 8, 4, half, float, uint64_t, float>,
  standard_descriptor_spec<16, 256, float, uint64_t, float>,
  vpq_descriptor_spec<16, 256, 8, 2, half, float, uint64_t, float>,
  vpq_descriptor_spec<16, 256, 8, 4, half, float, uint64_t, float>,
  standard_descriptor_spec<32, 512, float, uint64_t, float>,
  vpq_descriptor_spec<32, 512, 8, 2, half, float, uint64_t, float>,
  vpq_descriptor_spec<32, 512, 8, 4, half, float, uint64_t, float>,
  standard_descriptor_spec<8, 128, half, uint64_t, float>,
  vpq_descriptor_spec<8, 128, 8, 2, half, half, uint64_t, float>,
  vpq_descriptor_spec<8, 128, 8, 4, half, half, uint64_t, float>,
  standard_descriptor_spec<16, 256, half, uint64_t, float>,
  vpq_descriptor_spec<16, 256, 8, 2, half, half, uint64_t, float>,
  vpq_descriptor_spec<16, 256, 8, 4, half, half, uint64_t, float>,
  standard_descriptor_spec<32, 512, half, uint64_t, float>,
  vpq_descriptor_spec<32, 512, 8, 2, half, half, uint64_t, float>,
  vpq_descriptor_spec<32, 512, 8, 4, half, half, uint64_t, float>>;

}  // namespace cuvs::neighbors::cagra::detail
