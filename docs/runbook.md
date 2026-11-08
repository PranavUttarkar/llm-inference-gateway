# Runbook and GPU rental gate

## Before spending

- Run the local tests and load generator against the mock. Mock data is not a GPU result.
- Select and pin mutually compatible Triton vLLM backend image, OpenAI frontend, model revision, driver requirements and dependencies. Verify their official release documentation. This has not yet been done.
- Confirm the frontend implements streaming and that the gateway handles its exact SSE format, errors and disconnects. Add tests before using it for measurements.
- Add a fixed, token-counted workload, warmup, per-run metadata, robust completion detection and a results analyzer. Capture actual output token counts and metrics. The current load generator is a functional starting point only.
- Prepare Compose or equivalent deployment, metrics collection, and an export procedure. Never expose unauthenticated services to the internet.
- Check instance price, storage, bandwidth, availability and total budget. Reserve funds for startup and retries.

## GPU session

Use two GPUs on one host. Record model revision, container versions, GPU type, interconnect (`nvidia-smi topo -m`), driver and hourly price. Start with TP=1, verify model and streaming, then run a fixed workload at concurrency 1, 4, 8, 16 and 32 where feasible. Restart for TP=2 and repeat identical inputs and sampling parameters. Record failures, per-request TTFT and end-to-end latency, actual output tokens, GPU utilization and memory. Export artifacts before terminating the instance and check persistent-storage charges.

Report actual whole-instance rental cost separately from any hypothetical single-GPU cost. Continuous batching is provided by vLLM; increasing concurrency alone does not isolate its causal effect. TP=2 versus TP=1 does not compare against two independent TP=1 replicas.

Known version-specific issue: NVIDIA documents limitations for TP>1 with explicit model control and default distributed executor; check the release notes for the selected image before rental.

## Candidate version set (not yet validated on GPU)

NVIDIA documents `nvcr.io/nvidia/tritonserver:25.08-vllm-python-py3` with its bundled OpenAI frontend and vLLM `0.9.2+4ef1e343.nv25.8.post1.cu130`; use the components from this same image rather than installing a separate frontend or vLLM. See the [compatibility matrix](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/introduction/compatibility.html), [25.08 frontend instructions](https://docs.nvidia.com/deeplearning/triton-inference-server/archives/triton-inference-server-2600/user-guide/docs/client_guide/openai_readme.html), and [release notes and driver requirements](https://docs.nvidia.com/deeplearning/triton-inference-server/release-notes/rel-25-08.html). A tag is not a content digest: resolve and record the image digest and model/tokenizer revision before an experiment. NVIDIA's example uses gated Llama 3.1 8B and needs model access and sufficient VRAM. Set `tensor_parallel_size` in the model's `model.json` and use `KIND_MODEL`; avoid explicit model control with the documented TP limitation. Follow the frontend documentation for launch and verify streaming, usage, and metrics in the actual image before spending on a benchmark. No GPU deployment has been verified here.
