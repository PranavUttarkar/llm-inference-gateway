"""CPU numerical adapter to pinned, inspected KVarN PyTorch reference sources.

Packed records are real integer tensors, but attention uses dequantized tensors:
this is numerical emulation, NOT a fused packed-KV serving performance backend.
"""
import ast
import hashlib
import importlib.util
import math
import types
from pathlib import Path


def load_references(root):
    """Execute official pure-Torch code, injecting only its unavailable import.

    The store module's sole vLLM dependency is variance_normalize. Source bytes
    and the dependency substitution are recorded; algorithm bodies are unedited.
    """
    root = Path(root)
    sink = root/'vllm/model_executor/layers/quantization/kvarn/sinkhorn.py'
    store = root/'vllm/v1/attention/ops/kvarn_store.py'
    spec = importlib.util.spec_from_file_location('pilot_official_sinkhorn', sink)
    sink_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sink_module)
    tree = ast.parse(store.read_text(encoding='utf-8'))
    tree.body = [node for node in tree.body if not
                 (isinstance(node, ast.ImportFrom) and node.module and node.module.startswith('vllm.'))]
    module = types.ModuleType('pilot_official_store')
    module.variance_normalize = sink_module.variance_normalize
    exec(compile(tree, str(store), 'exec'), module.__dict__)
    module.variance_normalize_batched = sink_module.variance_normalize_batched
    module.source_sha256 = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                            for path in (sink, store)}
    return module


def hadamard(d, torch):
    if d < 1 or d & (d-1):
        raise ValueError('Hadamard head dimension must be a power of two')
    matrix = torch.ones(1, 1)
    while matrix.shape[0] < d:
        matrix = torch.cat([torch.cat([matrix, matrix], 1), torch.cat([matrix, -matrix], 1)], 0)
    return matrix/math.sqrt(d)


class Quantizer:
    def __init__(self, official, key_bits=4, value_bits=2, group=128, sink=128):
        import torch
        self.torch, self.official = torch, official
        self.key_bits, self.value_bits, self.group, self.sink = key_bits, value_bits, group, sink
        self.matrices = {}

    def encode(self, tensor, kind, frozen=None):
        """tensor [B,H,G,D]; operator includes pre-storage-rounding scales."""
        torch = self.torch
        b, h, g, d = tensor.shape
        matrix = self.matrices.setdefault((d, str(tensor.device)), hadamard(d, torch).to(tensor.device))
        rotated = tensor.float() @ matrix
        tiles = rotated.reshape(b*h, g, d)
        if kind == 'k':
            tiles = tiles.transpose(-2, -1).contiguous()
        bits = self.key_bits if kind == 'k' else self.value_bits
        if frozen is None:
            balanced, col, row = self.official.variance_normalize_batched(tiles, iterations=8)
            low = balanced.amin(-1, keepdim=True)
            step = ((balanced.amax(-1, keepdim=True)-low)/((1 << bits)-1)).clamp_min(1e-10)
            operator = {'col': col, 'row': row, 'low': low, 'step': step}
        else:
            operator = frozen
            col, row, low, step = [operator[key] for key in ('col', 'row', 'low', 'step')]
            balanced = tiles/col/row
        if frozen is None:
            function = (self.official.kvarn_store_tile_k_batch_from_sinkhorn if kind == 'k' else
                        self.official.kvarn_store_tile_v_batch_from_sinkhorn)
            record = function(balanced, col.squeeze(1), row.squeeze(2), bits)
        else:
            q = ((balanced-low)/step).round().clamp(0, (1 << bits)-1).to(torch.int32)
            packed = self.official._pack_lowbit(q, bits)
            if kind == 'k':
                record = {'q_packed_uint8': packed, 's_col_K': (row*step).squeeze(-1).half(),
                          'zp_K': (row*low).squeeze(-1).half(), 's_row_K': col.squeeze(1).half()}
            else:
                record = {'q_packed_uint8': packed, 's_col_V': col.squeeze(1).half(),
                          's_row_V': (row*step).squeeze(-1).half(), 'zp_V': (row*low).squeeze(-1).half()}
        packed = record['q_packed_uint8']
        factor = 8//bits
        q = torch.stack([(packed >> (bits*j)) & ((1 << bits)-1) for j in range(factor)], -1)
        q = q.flatten(-2).float()
        if kind == 'k':
            reconstructed = ((q*record['s_col_K'].float().unsqueeze(-1) +
                              record['zp_K'].float().unsqueeze(-1))*record['s_row_K'].float().unsqueeze(-2)).transpose(-2, -1)
        else:
            reconstructed = ((q*record['s_row_V'].float().unsqueeze(-1) +
                              record['zp_V'].float().unsqueeze(-1))*record['s_col_V'].float().unsqueeze(-2))
        reconstructed = (reconstructed @ matrix).reshape(b, h, g, d).to(tensor.dtype)
        if not torch.isfinite(reconstructed).all():
            raise ValueError('nonfinite official-reference reconstruction; do not score invalid cache')
        packed_bytes = sum(value.numel()*value.element_size() for value in record.values())
        return reconstructed, operator, packed_bytes

    def apply(self, raw, frozen=None):
        """Quantize complete post-sink tiles only. Recent incomplete tile stays BF16."""
        result, operators, total_bytes = [], {}, 0
        for layer, pair in enumerate(raw):
            new_pair = []
            for kind, original in zip(('k', 'v'), pair):
                reconstructed = original.clone()
                length = original.shape[-2]
                quantized_tokens = 0
                for start in range(self.sink, length-self.group+1, self.group):
                    key = (layer, kind, start)
                    value, operator, payload = self.encode(original[:,:,start:start+self.group], kind,
                                                           frozen[key] if frozen is not None else None)
                    reconstructed[:,:,start:start+self.group] = value
                    operators[key] = operator
                    quantized_tokens += self.group
                    total_bytes += payload
                # Persistent numerical buffers here are BF16, not the packed byte estimate.
                total_bytes += original.shape[0]*original.shape[1]*(length-quantized_tokens)*original.shape[-1]*2
                new_pair.append(reconstructed)
            result.append(tuple(new_pair))
        return tuple(result), operators, total_bytes
