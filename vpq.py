# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cupy as cp
import numpy as np
import itertools
import time

import csv
import sklearn


def cuvs_vpq(
    input,
    n_rows,
    n_cols,
    dtype,
    pq_kmeans_type,
    pq_dim,
    pq_bits,
    use_vq,
    use_subspaces,
    kmeans_n_iters,
):
    from pylibraft.common import device_ndarray

    input1_device = device_ndarray(input)

    params = pq.QuantizerParams(
        pq_bits=pq_bits,
        pq_dim=pq_dim,
        pq_kmeans_type=pq_kmeans_type,
        use_vq=use_vq,
        use_subspaces=use_subspaces,
        kmeans_n_iters=kmeans_n_iters,
    )
    start_time = time.time()
    quantizer = pq.train(params, input1_device)
    end_time = time.time()
    train_time = end_time - start_time
    # print(f"CUVS Training time: {train_time} seconds")

    start_time = time.time()
    transformed = pq.transform(quantizer, input1_device)
    end_time = time.time()
    encode_time = end_time - start_time
    # print(f"CUVS Transform time: {encode_time} seconds")

    reconstructed = cp.empty((n_rows, n_cols), dtype=dtype)
    start_time = time.time()
    pq.inverse_transform(quantizer, transformed, reconstructed)
    end_time = time.time()
    decode_time = end_time - start_time
    # print(f"CUVS Decode time: {decode_time} seconds")
    reconstructed = cp.array(reconstructed)
    # reconstruction_error = cp.linalg.norm(cp.array(input1_device) - reconstructed, axis=1)
    return (
        reconstructed,
        train_time * 1000,
        encode_time * 1000,
        decode_time * 1000,
    )


def faiss_vpq(input, n_rows, n_cols, dtype, pq_dim, pq_bits):
    """
    Perform product quantization using FAISS

    Args
    ----
        input: Input data array
        n_rows: Number of rows in input
        n_cols: Number of columns in input
        dtype: Data type
        pq_dim: Product quantization dimension (subvector size)
        pq_bits: Number of bits per subquantizer

    Returns
    -------
        reconstructed: Reconstructed data
        reconstruction_error: Mean reconstruction error
    """
    import faiss

    # Convert to numpy if needed and ensure correct dtype
    if hasattr(input, "get"):  # CuPy array
        input_np = input.get().astype(dtype)
    else:
        input_np = np.array(input, dtype=dtype)

    # Ensure input is contiguous and C-ordered for FAISS
    input_np = np.ascontiguousarray(input_np)

    # Calculate number of subquantizers
    # n_subquantizers = n_cols // pq_dim
    if n_cols % pq_dim != 0:
        raise ValueError(
            f"n_cols ({n_cols}) must be divisible by pq_dim ({pq_dim})"
        )

    # Create ProductQuantizer
    pq = faiss.ProductQuantizer(n_cols, pq_dim, pq_bits)

    # print(f"FAISS PQ: {pq_dim} dimensions, {pq_bits} bits each")

    # Train the quantizer
    start_time = time.time()
    pq.train(input_np)
    end_time = time.time()
    train_time = end_time - start_time
    # print(f"FAISS Training time: {train_time} seconds")

    # Encode (quantize) the data
    start_time = time.time()
    codes = pq.compute_codes(input_np)
    end_time = time.time()
    encode_time = end_time - start_time
    # print(f"FAISS Encode time: {encode_time} seconds")

    # Decode (reconstruct) the data
    start_time = time.time()
    reconstructed = pq.decode(codes)
    end_time = time.time()
    decode_time = end_time - start_time
    # print(f"FAISS Decode time: {decode_time} seconds")

    # Calculate reconstruction error
    # reconstruction_error = np.linalg.norm(input_np - reconstructed, axis=1)

    return (
        reconstructed,
        train_time * 1000,
        encode_time * 1000,
        decode_time * 1000,
    )


def cuvs_brute_force_search(index, queries, k=10):
    """
    Perform brute force k-NN search using CUVS

    Args
    ----
        database: Database vectors (CuPy array or numpy array)
        queries: Query vectors (CuPy array or numpy array)
        k: Number of nearest neighbors to find

    Returns
    -------
        distances: Distances to k nearest neighbors
        indices: Indices of k nearest neighbors
    """
    # Perform brute force search
    distances, indices = brute_force.search(index, queries, k)
    distances = distances.copy_to_host()
    indices = indices.copy_to_host()

    return distances, indices


def faiss_brute_force_search(database, queries, k=10):
    """
    Perform brute force k-NN search using FAISS

    Args
    ----
        database: Database vectors (numpy array or CuPy array)
        queries: Query vectors (numpy array or CuPy array)
        k: Number of nearest neighbors to find

    Returns
    -------
        distances: Distances to k nearest neighbors
        indices: Indices of k nearest neighbors
    """
    import faiss

    # Convert to numpy if needed
    if hasattr(database, "get"):  # CuPy array
        database_np = database.get()
    else:
        database_np = np.ascontiguousarray(database)

    if hasattr(queries, "get"):  # CuPy array
        queries_np = queries.get()
    else:
        queries_np = np.ascontiguousarray(queries)

    # Get dimensionality
    d = database_np.shape[1]

    # Create FAISS brute force index (L2 distance)
    index = faiss.IndexFlatL2(d)

    # Add database vectors to the index
    index.add(database_np)

    # Perform search
    distances, indices = index.search(queries_np, k)

    return distances, indices


def compute_recall(true_indices, pred_indices, k=10):
    """
    Compute recall@k between true and predicted nearest neighbors

    Args
    ----
        true_indices: Ground truth nearest neighbor indices
        pred_indices: Predicted nearest neighbor indices
        k: Number of top neighbors to consider

    Returns
    -------
        recall: Recall@k score
    """
    if len(true_indices.shape) == 1:
        true_indices = true_indices.reshape(1, -1)
    if len(pred_indices.shape) == 1:
        pred_indices = pred_indices.reshape(1, -1)

    # Take only top k
    true_k = true_indices[:, :k]
    pred_k = pred_indices[:, :k]

    recalls = []
    for i in range(len(true_k)):
        true_set = set(true_k[i])
        pred_set = set(pred_k[i])
        intersection = len(true_set.intersection(pred_set))
        recall = intersection / len(true_set) if len(true_set) > 0 else 0.0
        recalls.append(recall)

    return np.mean(recalls)


if __name__ == "__main__":
    from cuvs.preprocessing.quantize import pq
    from cuvs.neighbors import brute_force

    # Test parameters
    n_rows = 100000
    n_cols = 1024
    dtype = np.float32
    kmeans_types = ["kmeans_balanced"]  # , "kmeans"]
    pq_dims_array = [128, 256, 512]
    pq_bits_array = [6, 7, 8, 9, 10]
    kmeans_n_iters_array = [30]
    use_vq_array = [True, False]
    use_subspaces_array = [True, False]

    # Parameters for recall evaluation
    n_queries = 1000
    k_neighbors = 50

    input_data = sklearn.datasets.make_blobs(
        n_samples=n_rows, n_features=n_cols, centers=30, random_state=42
    )[0].astype(dtype)
    queries = sklearn.datasets.make_blobs(
        n_samples=n_queries, n_features=n_cols, centers=30, random_state=42
    )[0].astype(dtype)
    rng = np.random.default_rng(42)
    # input_data = rng.random((n_rows, n_cols), dtype=dtype) * 10
    # queries = rng.random((n_queries, n_cols), dtype=dtype) * 10
    input_device = cp.array(input_data)
    queries_device = cp.array(queries)

    # Prepare CSV file
    csv_filename = f"cuvs_bench_{n_rows}.csv"
    csv_headers = [
        "method",
        "kmeans_type",
        "pq_dim",
        "pq_bits",
        "use_vq",
        "use_subspaces",
        "kmeans_n_iters",
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
    index_original = brute_force.build(input_device)
    distances_ground_truth, indices_ground_truth = cuvs_brute_force_search(
        index_original, queries_device, k_neighbors
    )

    total_configs = len(
        list(
            itertools.product(
                kmeans_types,
                pq_dims_array,
                pq_bits_array,
                use_vq_array,
                use_subspaces_array,
                kmeans_n_iters_array,
            )
        )
    )
    current_config = 0

    for (
        kmeans,
        pq_dim,
        pq_bits,
        use_vq,
        use_subspaces,
        kmeans_n_iters,
    ) in itertools.product(
        kmeans_types,
        pq_dims_array,
        pq_bits_array,
        use_vq_array,
        use_subspaces_array,
        kmeans_n_iters_array,
    ):
        current_config += 1
        print(
            f"Processing configuration {current_config}/{total_configs}: pq_dim={pq_dim}, "
            f"pq_bits={pq_bits}, kmeans={kmeans}, use_vq={use_vq}, use_subspaces={use_subspaces}, kmeans_n_iters={kmeans_n_iters}"
        )

        try:
            # CUVS product quantization
            (
                reconstructed_cuvs,
                train_time_cuvs,
                encode_time_cuvs,
                decode_time_cuvs,
            ) = cuvs_vpq(
                input_data,
                n_rows,
                n_cols,
                dtype,
                kmeans,
                pq_dim,
                pq_bits,
                use_vq,
                use_subspaces,
                kmeans_n_iters,
            )

            # Compute recall for CUVS
            index_cuvs = brute_force.build(reconstructed_cuvs)
            distances_cuvs, indices_cuvs = cuvs_brute_force_search(
                index_cuvs, queries_device, k_neighbors
            )
            recall_cuvs = compute_recall(
                indices_ground_truth, indices_cuvs, k_neighbors
            )

            # Store CUVS results
            results.append(
                [
                    "CUVS",
                    kmeans,
                    pq_dim,
                    pq_bits,
                    use_vq,
                    use_subspaces,
                    kmeans_n_iters,
                    f"{train_time_cuvs:.2f}",
                    f"{encode_time_cuvs:.2f}",
                    f"{decode_time_cuvs:.2f}",
                    f"{recall_cuvs:.2f}",
                    k_neighbors,
                    n_rows,
                    n_cols,
                    n_queries,
                ]
            )
            print(f"CUVS results: {results[-1]}")

        except Exception as e:
            print(f"CUVS failed for config {current_config}: {e}")
            results.append(
                [
                    "CUVS",
                    kmeans,
                    pq_dim,
                    pq_bits,
                    use_vq,
                    use_subspaces,
                    kmeans_n_iters,
                    "-1",
                    "-1",
                    "-1",
                    "-1",
                    k_neighbors,
                    n_rows,
                    n_cols,
                    n_queries,
                ]
            )
        if kmeans == "kmeans_balanced" or use_vq != 0 or use_subspaces != 0:
            continue
        # try:
        #    # Test FAISS product quantization
        #    reconstructed_faiss, train_time_faiss, encode_time_faiss, decode_time_faiss = faiss_vpq(
        #        input_data, n_rows, n_cols, dtype, pq_dim, pq_bits
        #    )
        #
        #    # Compute recall for FAISS
        #    reconstructed_faiss_device = cp.array(reconstructed_faiss)
        #    index_faiss = brute_force.build(reconstructed_faiss_device)
        #    distances_faiss, indices_faiss = cuvs_brute_force_search(index_faiss, queries_device, k_neighbors)
        #    recall_faiss = compute_recall(indices_ground_truth, indices_faiss, k_neighbors)
        #
        #    # Store FAISS results
        #    results.append([
        #        "FAISS", kmeans, pq_dim, pq_bits, use_vq, use_subspaces, kmeans_n_iters,
        #        train_time_faiss, encode_time_faiss, decode_time_faiss,
        #        recall_faiss, k_neighbors, n_rows, n_cols, n_queries
        #    ])
        #    print(f"FAISS results: {results[-1]}")
        #
        # except Exception as e:
        #    print(f"FAISS failed for config {current_config}: {e}")
        #    results.append([
        #        "FAISS", kmeans, pq_dim, pq_bits, use_vq, use_subspaces, kmeans_n_iters,
        #        -1, -1, -1, -1, k_neighbors, n_rows, n_cols, n_queries
        #    ])

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
