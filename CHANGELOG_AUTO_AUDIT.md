# CHANGELOG_AUTO_AUDIT

## 2024-06-05
- feat: replace monolithic app with service-oriented architecture (`deploy/api`, `deploy/worker-vision`).
- feat: add hardware-aware bootstrapper (`tools/bootstrap.sh`) and detection probe (`tools/hw_probe.sh`).
- chore: pin Python/Torch/ONNX/OpenCV stacks per service via dedicated requirement files.
- chore: publish docker-compose profiles (cpu/gpu) with Redis queue and shared artifact volume.
- test: introduce unit coverage for API queueing and worker artifacts plus dockerized smoke test harness.
- docs: rewrite README, add migration guide, and refresh audit reports for the new workflow.
