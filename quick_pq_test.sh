#! /bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

prog="./examples/cpp/build/PQ_EXAMPLE"
args="--method pq  --n_vectors 100000"
for pq_bits in 7 8 10; do
    for pq_dim in 128 256 512; do
        CUDA_VISIBLE_DEVICES=7 $prog "${args}" --pq_dim $pq_dim --pq_bits $pq_bits | tail -n 4 >> quick_pq_test.txt
    done
done

CUDA_VISIBLE_DEVICES=7 $prog "${args}" --pq_dim 64 --pq_bits 10 --n_dims 384 --use_vq 1 | tail -n 4 >> quick_pq_test.txt
echo "--------------------------------" >> quick_pq_test.txt
