# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cupy as cp
import numpy as np
import itertools
import csv
from vpq import faiss_vpq, faiss_brute_force_search, compute_recall

if __name__ == "__main__":
    # Test parameters
    n_rows = 100000
    n_cols = 1024
    dtype = np.float32
    kmeans_types = ["kmeans"]
    pq_dims_array = [128, 256, 512]
    pq_bits_array = [6, 7, 8, 10]
    vq_n_centers_array = [0]
    pq_kmeans_trainset_fraction_array = [1]
    kmeans_n_iters_array = [30, 100]

    # Parameters for recall evaluation
    n_queries = 1000
    k_neighbors = 50

    input_data = np.random.random_sample((n_rows, n_cols)).astype(dtype)
    queries = np.random.random_sample((n_queries, n_cols)).astype(dtype)
    input_device = cp.array(input_data)
    queries_device = cp.array(queries)

    # Prepare CSV file
    csv_filename = "vpq_comparison_results.csv"
    csv_headers = [
        "method",
        "kmeans_type",
        "pq_dim",
        "pq_bits",
        "vq_n_centers",
        "pq_kmeans_trainset_fraction",
        "kmeans_n_iters",
        "reconstruction_error",
        "train_time",
        "encode_time",
        "decode_time",
        "recall_at_k",
        "k_neighbors",
        "n_rows",
        "n_cols",
        "n_queries",
    ]

    results = []

    # Compute ground truth once
    print("Computing ground truth...")
    distances_ground_truth, indices_ground_truth = faiss_brute_force_search(
        input_data, queries, k_neighbors
    )

    total_configs = len(
        list(
            itertools.product(
                kmeans_types,
                pq_dims_array,
                pq_bits_array,
                vq_n_centers_array,
                pq_kmeans_trainset_fraction_array,
                kmeans_n_iters_array,
            )
        )
    )
    current_config = 0

    for (
        kmeans,
        pq_dim,
        pq_bits,
        vq_n_centers,
        pq_kmeans_trainset_fraction,
        kmeans_n_iters,
    ) in itertools.product(
        kmeans_types,
        pq_dims_array,
        pq_bits_array,
        vq_n_centers_array,
        pq_kmeans_trainset_fraction_array,
        kmeans_n_iters_array,
    ):
        current_config += 1
        print(
            f"Processing configuration {current_config}/{total_configs}: pq_dim={pq_dim}, "
            f"pq_bits={pq_bits}, kmeans={kmeans}, vq_centers={vq_n_centers}, pq_kmeans_trainset_fraction={pq_kmeans_trainset_fraction}"
        )
        try:
            # Test FAISS product quantization
            (
                reconstructed_faiss,
                reconstruction_error_faiss,
                train_time_faiss,
                encode_time_faiss,
                decode_time_faiss,
            ) = faiss_vpq(input_data, n_rows, n_cols, dtype, pq_dim, pq_bits)

            # Compute recall for FAISS
            reconstructed_faiss_device = cp.array(reconstructed_faiss)
            distances_faiss, indices_faiss = faiss_brute_force_search(
                reconstructed_faiss_device, queries_device, k_neighbors
            )
            recall_faiss = compute_recall(
                indices_ground_truth, indices_faiss, k_neighbors
            )

            # Store FAISS results
            results.append(
                [
                    "FAISS",
                    kmeans,
                    pq_dim,
                    pq_bits,
                    vq_n_centers,
                    pq_kmeans_trainset_fraction,
                    kmeans_n_iters,
                    reconstruction_error_faiss,
                    train_time_faiss,
                    encode_time_faiss,
                    decode_time_faiss,
                    recall_faiss,
                    k_neighbors,
                    n_rows,
                    n_cols,
                    n_queries,
                ]
            )
            # print(f"FAISS results: {results[-1]}")

        except Exception as e:
            print(f"FAISS failed for config {current_config}: {e}")
            results.append(
                [
                    "FAISS",
                    kmeans,
                    pq_dim,
                    pq_bits,
                    vq_n_centers,
                    pq_kmeans_trainset_fraction,
                    kmeans_n_iters,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    k_neighbors,
                    n_rows,
                    n_cols,
                    n_queries,
                ]
            )

    # Write results to CSV
    print(f"\nWriting results to {csv_filename}...")
    with open(csv_filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(csv_headers)
        writer.writerows(results)

    print(f"Results saved to {csv_filename}")
    print(f"Total configurations tested: {len(results)}")
    print(
        f"CSV file contains {len(csv_headers)} columns and {len(results)} data rows"
    )
