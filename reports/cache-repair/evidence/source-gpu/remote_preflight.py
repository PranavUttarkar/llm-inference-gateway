"""Read-only GPU readiness check; exits before inference on incompatible builds."""
import argparse
import json
import platform
import subprocess
from pathlib import Path


def check():
    import torch
    import transformers
    failures = []
    if not torch.cuda.is_available():
        failures.append('CUDA unavailable')
    info = {'python': platform.python_version(), 'torch': torch.__version__,
            'transformers': transformers.__version__, 'cuda_runtime': torch.version.cuda,
            'cuda_available': torch.cuda.is_available(), 'devices': []}
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            info['devices'].append({'index': index, 'name': properties.name,
                                    'capability': [properties.major, properties.minor],
                                    'total_memory_bytes': properties.total_memory})
            # Execute on each device, not only on the default CUDA device.
            try:
                a = torch.ones(64, 64, device=f'cuda:{index}', dtype=torch.bfloat16)
                result = a @ a
                torch.cuda.synchronize(index)
                if not torch.all(result == 64).item():
                    failures.append(f'GPU {index} BF16 matmul returned unexpected values')
            except Exception as exc:
                failures.append(f'GPU {index} kernel smoke failed: {type(exc).__name__}: {exc}')
        if any(device['capability'][0] >= 12 for device in info['devices']):
            version = tuple(map(int, (torch.version.cuda or '0.0').split('.')[:2]))
            if version < (12, 8):
                failures.append('RTX 5090 requires a Blackwell-compatible CUDA build; use CUDA >=12.8')
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=15)
        info['nvidia_smi'] = result.stdout
    except (OSError, subprocess.TimeoutExpired) as exc:
        info['nvidia_smi_error'] = str(exc)
    info['failures'] = failures
    info['ready'] = not failures
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('refusing to overwrite preflight evidence')
    result = check()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result, indent=2))
    if not result['ready']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
