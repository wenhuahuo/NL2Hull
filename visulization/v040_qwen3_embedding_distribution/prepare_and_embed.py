"""Embed all seven-model FFD descriptions, then project them to two PCA axes."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap
import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.decomposition import PCA

MODEL_ID = 'Qwen/Qwen3-Embedding-0.6B'
MODEL_NAMES = (
    'wokey/claude-opus-5',
    'wokey/kimi-k3',
    'wokey/glm-5.3',
    'wokey/grok-4.7',
    'deepseek/deepseek-flash',
    'wokey/gpt-5.6-sol',
    'openrouter/xiaomi/mimo-v2.6-pro',
)


def load_samples(path: Path) -> tuple[list[str], np.ndarray, Counter[str]]:
    texts: list[str] = []
    model_indices: list[int] = []
    counts: Counter[str] = Counter()
    models = {name: index for index, name in enumerate(MODEL_NAMES)}
    with path.open(encoding='utf-8') as source:
        for line in source:
            sample = json.loads(line)
            name = sample['model']
            if name not in models:  # synthetic rollback templates are not a generator model
                if name != 'synthetic_rollback_v1':
                    raise ValueError(f'Unexpected source model: {name}')
                continue
            texts.append(sample['text'])
            model_indices.append(models[name])
            counts[name] += 1
    if not texts or any(counts[name] == 0 for name in MODEL_NAMES):
        raise ValueError('Expected nonempty examples for all seven generator models')
    return texts, np.asarray(model_indices, dtype=np.uint8), counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Output already exists: {args.output}')
    args.output.mkdir(parents=True)
    input_hash = hashlib.sha256(args.input.read_bytes()).hexdigest()
    texts, labels, counts = load_samples(args.input)
    print(f'Loaded {len(texts)} model-generated texts; counts={dict(counts)}', flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the full embedding run')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side='left')
    model = AutoModel.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, attn_implementation='sdpa'
    ).cuda().eval()
    dimension = model.config.hidden_size
    vectors = open_memmap(args.output / 'embeddings.npy', mode='w+', dtype=np.float32,
                          shape=(len(texts), dimension))
    with torch.inference_mode():
        for start in range(0, len(texts), args.batch_size):
            stop = min(start + args.batch_size, len(texts))
            batch = tokenizer(texts[start:stop], padding=True, truncation=True,
                              max_length=512, return_tensors='pt').to('cuda')
            hidden = model(**batch).last_hidden_state
            last_token = hidden[:, -1, :]
            vectors[start:stop] = torch.nn.functional.normalize(
                last_token, p=2, dim=1
            ).float().cpu().numpy()
            if stop % 4096 < args.batch_size or stop == len(texts):
                vectors.flush()
                print(f'embedded={stop}/{len(texts)}', flush=True)
    del model, tokenizer, texts
    vectors.flush()
    projection = PCA(n_components=2, svd_solver='randomized', random_state=20260925)
    coordinates = projection.fit_transform(vectors).astype(np.float32)
    np.savez_compressed(args.output / 'projection.npz', xy=coordinates, model=labels)
    report = {
        'input': str(args.input), 'input_sha256': input_hash,
        'model_id': MODEL_ID, 'model_counts': dict(counts),
        'sentence_count': len(labels), 'embedding_dimension': vectors.shape[1],
        'normalization': 'unit L2', 'pooling': 'last non-padding token',
        'max_seq_length': 512,
        'batch_size': args.batch_size, 'projection': 'PCA on all embeddings',
        'pca_explained_variance_ratio': projection.explained_variance_ratio_.tolist(),
        'model_order': MODEL_NAMES, 'random_seed': 20260925,
    }
    (args.output / 'run_manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
