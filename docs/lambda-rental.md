# Two-GPU rental: Lambda walkthrough

This is a plan, not a reservation. Prices and inventory were checked on 2026-11-08; confirm the exact offer and total before launching. No one has validated this project's GPU deployment yet.

## Why Lambda

Choose **Lambda On-Demand Cloud, 2× NVIDIA A6000**, if available at a suitable price. Lambda documents a single Linux VM with two 48 GB A6000 GPUs and a 1 TiB root volume. Its default Lambda Stack includes Docker, NVIDIA drivers and the NVIDIA Container Toolkit. Verify the installed versions on the instance. Sources: [instance specifications](https://docs.lambda.ai/public-cloud/on-demand/) and [default image](https://docs.lambda.ai/public-cloud/on-demand/getting-started/).

The [pricing page](https://lambda.ai/instances), checked 2026-11-08, lists the 2× A6000 configuration at **$1.09 per GPU-hour**, or **$2.18 per instance-hour**, before tax, but says that displayed price takes effect **November 16, 2026 at 00:00 UTC**. Until then, Lambda says to log in for current pricing. At the displayed future rate, two hours would be about $4.36 and four hours about $8.72 in compute. These are estimates, not a current quote or spending guarantee. Inventory, region, taxes and other charges can change. Lambda says on-demand billing is in minute increments while the instance is running, and separate filesystems continue to bill while they exist; see [billing](https://docs.lambda.ai/on-demand-cloud/billing/).

Target **at most four hours** at that rate, and set alarms for two and three hours. Keep roughly $3 of the $12 ceiling for taxes, delays or a retry; stop earlier if the checkout price or other charges leave less room. The cap is manual, not an automatic spending limit. If 2× A6000 is unavailable, wait or inspect a different provider's two-GPU offer; do not substitute a substantially more expensive instance without recalculating the budget.

## 1. Prepare on Windows before the clock starts

1. Create a Lambda account, complete payment setup, and sign in to the [Lambda Cloud console](https://cloud.lambda.ai/). Do not launch yet.
2. Prepare an SSH key if you do not already have one for Lambda. In PowerShell, first check whether `~/.ssh/lambda_gateway` already exists. If it does, use your existing key or choose a different filename; do not overwrite it. Otherwise:

   ```powershell
   ssh-keygen -t ed25519 -f "$env:USERPROFILE\.ssh\lambda_gateway" -C "lambda-gateway"
   Get-Content "$env:USERPROFILE\.ssh\lambda_gateway.pub"
   ```

   Add the displayed **public** key in Lambda's SSH keys settings. Keep the private key on your computer.
3. Prepare a private repository bundle locally, from the project directory. This includes Git history, so treat it as private. It includes the committed project, not uncommitted files or secrets excluded from Git. Check `git status` first.

   ```powershell
   git status --short
   git bundle create "$env:TEMP\gateway.bundle" HEAD main
   git rev-parse HEAD
   ```

   Record the last command's commit ID. If the bundle file already exists, pick a new filename rather than overwriting a file you need.
4. Read [the experiment runbook](runbook.md), have the commands ready, and arrange local space for results. Decide the hard stop time using the actual quoted rate and total budget. Do not create a separate billable filesystem for this short session unless needed.

## 2. Launch exactly one two-GPU VM

1. In the console choose **On-Demand Cloud → Instances → Launch instance**. Select **2× A6000**, an available region, the default Lambda Stack image, your SSH public key, and **quantity 1**. UI labels can change. Confirm the listed price is for the complete two-GPU instance; if shown per GPU, multiply by two. Check applicable taxes and any other charges.
2. Launch only when ready to work. Record the instance name, region, price and billing start time. Set the alarms and hard stop. Wait until the console reports the instance ready and provides an IP address and SSH command.
3. In PowerShell set the IP from the console and copy the bundle. Use the console's SSH username if it differs from `ubuntu`:

   ```powershell
   $gpuIp = 'REPLACE_WITH_INSTANCE_IP'
   scp -i "$env:USERPROFILE\.ssh\lambda_gateway" "$env:TEMP\gateway.bundle" "ubuntu@${gpuIp}:~/gateway.bundle"
   ssh -i "$env:USERPROFILE\.ssh\lambda_gateway" "ubuntu@${gpuIp}"
   ```

   Verify the host key fingerprint using the provider's information if available. Do not upload your private SSH key or GitHub credentials.

## 3. Check the VM and transfer the project

Run these commands on the VM. If any prerequisite fails, troubleshoot briefly within the time budget; export diagnostics and terminate rather than letting billing continue indefinitely.

```sh
nvidia-smi
nvidia-smi topo -m
docker version
docker compose version
docker run --rm --gpus all ubuntu:22.04 nvidia-smi
df -h
git clone ~/gateway.bundle ~/llm-inference-gateway
cd ~/llm-inference-gateway
git rev-parse HEAD
python3 -m unittest discover -v
```

Confirm two A6000s are visible both on the host and inside the test container, and that the commit ID matches the local one. Check driver compatibility against [Triton 25.08's release notes](https://docs.nvidia.com/deeplearning/triton-inference-server/release-notes/rel-25-08.html). Record topology rather than assuming NVLink. Check Docker Compose is installed and your user can access Docker; resolve setup problems before launching the stack. The GPU image and model are downloaded on first startup, consuming billed time and disk.

## 4. Start TP=1 and perform the smoke test

On the VM:

```sh
mkdir -p artifacts
date -u > artifacts/session-start-utc.txt
nvidia-smi > artifacts/nvidia-smi.txt
nvidia-smi topo -m > artifacts/topology.txt
git rev-parse HEAD > artifacts/git-revision.txt
docker version > artifacts/docker-version.txt
nvidia-smi --query-gpu=timestamp,index,name,driver_version,memory.total,utilization.gpu,memory.used,power.draw --format=csv -l 5 > artifacts/gpu-samples.csv &
GPU_MONITOR_PID=$!
TP_SIZE=1 docker compose -f compose.gpu.yaml up -d --build
docker compose -f compose.gpu.yaml logs -f backend
```

Wait for the backend to load the model and for its frontend to listen; Ctrl-C exits the log viewer. Then:

```sh
python3 benchmarks/run_load.py --model qwen --workload workloads/prompts.json --warmup 1 --requests 1 --concurrency 1 --include-usage --label smoke-tp1 --output artifacts/smoke-tp1.jsonl
cat artifacts/smoke-tp1.jsonl
nvidia-smi
```

Proceed only if the result reports HTTP 200, completion, nonempty output and sensible prompt/completion token usage. Compare input usage with the expected counts in the runbook. Diagnose or stop if the backend fails, usage is missing, or streaming is incomplete. Gateway `/healthz` alone does not establish backend readiness.

## 5. Run matched TP=1 and TP=2 trials

The full matrix is in [the runbook](runbook.md). To fit a short first session, begin with concurrency **1 and 4**, 32 requests each, two repetitions, and four warmups. Add 8, 16 and 32 only if time remains. Keep the same settings for both TP modes; save every raw result. This is one model instance on the same paid two-GPU host, sharded across both GPUs for TP=2. It is not a comparison against two independent TP=1 replicas.

```sh
for C in 1 4; do
  for R in 1 2; do
    python3 benchmarks/run_load.py --model qwen --workload workloads/prompts.json --warmup 4 --requests 32 --concurrency "$C" --max-tokens 128 --timeout 120 --include-usage --label "tp1-c${C}-r${R}" --output "artifacts/tp1-c${C}-r${R}.jsonl"
  done
done
docker compose -f compose.gpu.yaml logs --no-color > artifacts/tp1-container-logs.txt
TP_SIZE=2 docker compose -f compose.gpu.yaml up -d --force-recreate backend
docker compose -f compose.gpu.yaml logs -f backend
```

Wait for readiness, exit the log viewer, and repeat the smoke test using `smoke-tp2` in the label and filename. Confirm both GPUs are used. Then run the same loop, replacing `tp1` with `tp2` in labels and filenames. Compare each matching pair, for example:

```sh
python3 benchmarks/compare.py artifacts/tp1-c4-r1.jsonl.summary.json artifacts/tp2-c4-r1.jsonl.summary.json > artifacts/compare-c4-r1.json
docker compose -f compose.gpu.yaml logs --no-color > artifacts/tp2-container-logs.txt
```

Review failures, missing usage and 429s before interpreting ratios. Percentiles based on 32 samples, especially p99, are imprecise. The 1.5B model and small prompt set limit generalization. Keep budget time for copying results and terminating.

## 6. Export and terminate

On the VM, record the end time, provider price and deviations in `artifacts/metadata.txt`, then stop GPU sampling and package results:

```sh
date -u > artifacts/session-end-utc.txt
kill "$GPU_MONITOR_PID"
tar -czf ~/gateway-artifacts.tar.gz artifacts
```

On Windows, in another PowerShell window, download the archive to the current directory and confirm it opens and contains the expected files:

```powershell
scp -i "$env:USERPROFILE\.ssh\lambda_gateway" "ubuntu@${gpuIp}:~/gateway-artifacts.tar.gz" .
tar -tzf gateway-artifacts.tar.gz
```

If `$gpuIp` is unset in the new window, set it again. After verifying the download, **terminate the instance in the Lambda console** and verify its status; stopping Docker or closing SSH does not terminate the VM. Export any other needed files first because termination can destroy instance-local data. Check for separately created billable filesystems and delete only those no longer needed. Review the actual invoice and artifacts for private details before publishing.
