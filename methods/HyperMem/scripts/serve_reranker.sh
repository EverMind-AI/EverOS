#!/bin/bash
# Start Qwen3-Reranker-4B service for stage4 retrieval reranking
CUDA_VISIBLE_DEVICES="4,5,6,7" \
vllm serve /Evermind/sh_evermind/yuejuwei/models/Qwen/Qwen3-Reranker-4B \
--served-model-name Qwen3-Reranker-4B \
--host 0.0.0.0 \
--port 12810 \
--tensor-parallel-size 4 \
--gpu-memory-utilization 0.9
