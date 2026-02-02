/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"
#include "ann_utils.cuh"

#include <cuvs/neighbors/recall.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/random/rng.cuh>
#include <raft/util/itertools.hpp>

#include <gtest/gtest.h>

#include <vector>

namespace cuvs::neighbors {

struct RecallInputs {
  int64_t n_queries;
  int64_t k;
  bool host_data;
};

template <typename IdxT>
class RecallTest : public ::testing::TestWithParam<RecallInputs> {
 public:
  RecallTest()
    : stream_(raft::resource::get_cuda_stream(handle_)),
      params_(::testing::TestWithParam<RecallInputs>::GetParam())
  {
  }

 protected:
  void SetUp() override
  {
    // Generate ground truth indices
    ground_truth_host_.resize(params_.n_queries * params_.k);
    actual_indices_host_.resize(params_.n_queries * params_.k);

    // Create ground truth: sequential indices for each query
    for (int64_t i = 0; i < params_.n_queries; ++i) {
      for (int64_t j = 0; j < params_.k; ++j) {
        ground_truth_host_[i * params_.k + j] = static_cast<IdxT>(j);
      }
    }

    // Create actual indices with known recall
    // First half matches perfectly, second half doesn't match
    for (int64_t i = 0; i < params_.n_queries; ++i) {
      for (int64_t j = 0; j < params_.k; ++j) {
        if (j < params_.k / 2) {
          // Match with ground truth
          actual_indices_host_[i * params_.k + j] = static_cast<IdxT>(j);
        } else {
          // No match - use indices beyond ground truth range
          actual_indices_host_[i * params_.k + j] = static_cast<IdxT>(params_.k + j);
        }
      }
    }

    // Allocate device memory if needed
    if (!params_.host_data) {
      ground_truth_device_ =
        raft::make_device_matrix<IdxT, int64_t>(handle_, params_.n_queries, params_.k);
      actual_indices_device_ =
        raft::make_device_matrix<IdxT, int64_t>(handle_, params_.n_queries, params_.k);

      raft::copy(ground_truth_device_.data_handle(),
                 ground_truth_host_.data(),
                 ground_truth_host_.size(),
                 stream_);
      raft::copy(actual_indices_device_.data_handle(),
                 actual_indices_host_.data(),
                 actual_indices_host_.size(),
                 stream_);
      raft::resource::sync_stream(handle_);
    } else {
      ground_truth_host_matrix_ =
        raft::make_host_matrix<IdxT, int64_t>(params_.n_queries, params_.k);
      actual_indices_host_matrix_ =
        raft::make_host_matrix<IdxT, int64_t>(params_.n_queries, params_.k);

      std::copy(ground_truth_host_.begin(),
                ground_truth_host_.end(),
                ground_truth_host_matrix_.data_handle());
      std::copy(actual_indices_host_.begin(),
                actual_indices_host_.end(),
                actual_indices_host_matrix_.data_handle());
    }
  }

  void testRecall()
  {
    float recall;

    if (params_.host_data) {
      recall = cuvs::neighbors::compute_recall(
        handle_, ground_truth_host_matrix_.view(), actual_indices_host_matrix_.view());
    } else {
      recall = cuvs::neighbors::compute_recall(
        handle_, ground_truth_device_.view(), actual_indices_device_.view());
    }

    // Expected recall: half of the neighbors match
    float expected_recall = 0.5f;
    float tolerance       = 0.01f;

    ASSERT_NEAR(recall, expected_recall, tolerance)
      << "Recall should be approximately " << expected_recall << " but got " << recall;
  }

  void testPerfectRecall()
  {
    // Modify actual_indices to match ground truth perfectly
    for (int64_t i = 0; i < params_.n_queries; ++i) {
      for (int64_t j = 0; j < params_.k; ++j) {
        actual_indices_host_[i * params_.k + j] = ground_truth_host_[i * params_.k + j];
      }
    }

    if (!params_.host_data) {
      raft::copy(actual_indices_device_.data_handle(),
                 actual_indices_host_.data(),
                 actual_indices_host_.size(),
                 stream_);
      raft::resource::sync_stream(handle_);
    } else {
      std::copy(actual_indices_host_.begin(),
                actual_indices_host_.end(),
                actual_indices_host_matrix_.data_handle());
    }

    float recall;

    if (params_.host_data) {
      recall = cuvs::neighbors::compute_recall(
        handle_, ground_truth_host_matrix_.view(), actual_indices_host_matrix_.view());
    } else {
      recall = cuvs::neighbors::compute_recall(
        handle_, ground_truth_device_.view(), actual_indices_device_.view());
    }

    // Expected recall: 1.0 (perfect match)
    ASSERT_FLOAT_EQ(recall, 1.0f) << "Recall should be 1.0 for perfect match";
  }

  void testZeroRecall()
  {
    // Modify actual_indices to not match ground truth at all
    for (int64_t i = 0; i < params_.n_queries; ++i) {
      for (int64_t j = 0; j < params_.k; ++j) {
        // Use indices that don't appear in ground truth
        actual_indices_host_[i * params_.k + j] = static_cast<IdxT>(params_.k * 2 + j);
      }
    }

    if (!params_.host_data) {
      raft::copy(actual_indices_device_.data_handle(),
                 actual_indices_host_.data(),
                 actual_indices_host_.size(),
                 stream_);
      raft::resource::sync_stream(handle_);
    } else {
      std::copy(actual_indices_host_.begin(),
                actual_indices_host_.end(),
                actual_indices_host_matrix_.data_handle());
    }

    float recall;

    if (params_.host_data) {
      recall = cuvs::neighbors::compute_recall(
        handle_, ground_truth_host_matrix_.view(), actual_indices_host_matrix_.view());
    } else {
      recall = cuvs::neighbors::compute_recall(
        handle_, ground_truth_device_.view(), actual_indices_device_.view());
    }

    // Expected recall: 0.0 (no matches)
    ASSERT_FLOAT_EQ(recall, 0.0f) << "Recall should be 0.0 for no match";
  }

 private:
  raft::resources handle_;
  rmm::cuda_stream_view stream_;
  RecallInputs params_;

  std::vector<IdxT> ground_truth_host_;
  std::vector<IdxT> actual_indices_host_;

  raft::device_matrix<IdxT, int64_t> ground_truth_device_;
  raft::device_matrix<IdxT, int64_t> actual_indices_device_;

  raft::host_matrix<IdxT, int64_t> ground_truth_host_matrix_;
  raft::host_matrix<IdxT, int64_t> actual_indices_host_matrix_;
};

const std::vector<RecallInputs> inputs = raft::util::itertools::product<RecallInputs>(
  {static_cast<int64_t>(10), static_cast<int64_t>(100), static_cast<int64_t>(1000)},
  {static_cast<int64_t>(10), static_cast<int64_t>(32), static_cast<int64_t>(64)},
  {false, true});

typedef RecallTest<int64_t> RecallTestI64;
TEST_P(RecallTestI64, HalfRecall) { this->testRecall(); }
TEST_P(RecallTestI64, PerfectRecall) { this->testPerfectRecall(); }
TEST_P(RecallTestI64, ZeroRecall) { this->testZeroRecall(); }

INSTANTIATE_TEST_CASE_P(RecallTest, RecallTestI64, ::testing::ValuesIn(inputs));

typedef RecallTest<int32_t> RecallTestI32;
TEST_P(RecallTestI32, HalfRecall) { this->testRecall(); }
TEST_P(RecallTestI32, PerfectRecall) { this->testPerfectRecall(); }
TEST_P(RecallTestI32, ZeroRecall) { this->testZeroRecall(); }

INSTANTIATE_TEST_CASE_P(RecallTest, RecallTestI32, ::testing::ValuesIn(inputs));

typedef RecallTest<uint32_t> RecallTestU32;
TEST_P(RecallTestU32, HalfRecall) { this->testRecall(); }
TEST_P(RecallTestU32, PerfectRecall) { this->testPerfectRecall(); }
TEST_P(RecallTestU32, ZeroRecall) { this->testZeroRecall(); }

INSTANTIATE_TEST_CASE_P(RecallTest, RecallTestU32, ::testing::ValuesIn(inputs));

}  // namespace cuvs::neighbors
