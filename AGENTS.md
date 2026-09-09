# Repository execution rules

- Server experiments may use **physical NVIDIA GPU 1 and GPU 2 only**. This is a hard user constraint. Never use GPU 0, GPU 3, an unrestricted CUDA device list, or a fallback GPU.
- Resolve physical indices through `nvidia-smi` and bind child processes to the corresponding GPU UUIDs. CUDA logical device 0 inside that restricted process is permitted; it is not physical GPU 0.
- Use `python -m experiments.supplement run ... --execute` for the supplementary experiment suite. It owns its solver/policy services and refuses occupied GPUs/ports. Do not invoke legacy launchers: some use GPU 0/3 or kill unrelated services.
- Do not run model loading, training, inference, or GPU experiments on the user's local Mac. Static source checks are allowed; runtime tests and experiments run on the server.
- Never kill unrelated GPU jobs or reuse a model endpoint whose GPU placement has not been established.
