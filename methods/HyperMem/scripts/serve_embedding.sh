#!/bin/bash
# Start Qwen3-Embedding-4B service for stage3 vector index building
CUDA_VISIBLE_DEVICES="0,1,2,3" \
vllm serve /Evermind/sh_evermind/yuejuwei/models/Qwen/Qwen3-Embedding-4B \
--served-model-name Qwen3-Embedding-4B \
--host 0.0.0.0 \
--port 11810 \
--tensor-parallel-size 4 \
--gpu-memory-utilization 0.9 \
--convert embed
