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

#include <gtest/gtest.h>

#include "../ann_ivf_flat.cuh"

namespace cuvs::neighbors::ivf_flat {

typedef AnnIVFFlatTest<float, float, int64_t> AnnIVFFlatTestF_float;
TEST_P(AnnIVFFlatTestF_float, AnnIVFFlat) {
    this->testIVFFlat();
    this->testPacker();
}

INSTANTIATE_TEST_CASE_P(AnnIVFFlatTest, AnnIVFFlatTestF_float, ::testing::ValuesIn(inputs));

typedef AnnIVFFlatTest<float, float, int64_t> AnnIVFFlatTestF_cosine_float;
TEST_P(AnnIVFFlatTestF_cosine_float, AnnIVFFlat) {
    this->testIVFFlatCosine();
}
INSTANTIATE_TEST_CASE_P(AnnIVFFlatTest, AnnIVFFlatTestF_cosine_float, ::testing::ValuesIn(inputs_cosine));

}  // namespace cuvs::neighbors::ivf_flat
