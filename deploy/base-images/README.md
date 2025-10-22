# Base Images

This directory documents the canonical container roots used across services.

* `python:3.11.8-slim` is the foundation for CPU services.
* `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04` seeds CUDA-enabled workers.
* Future ROCm builds will derive from `rocm/dev-ubuntu-22.04` once validated.
