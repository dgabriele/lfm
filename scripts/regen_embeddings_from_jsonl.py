"""Regenerate embeddings.npy from passages.jsonl.

Output shape: (N, n_pos, D) — n_pos uniformly-binned Qwen hidden states per
passage, mirroring _last_k_hidden on the alien side so position p of the
source corresponds to position p of the alien generation.
"""
import json, logging, numpy as np, torch
from pathlib import Path
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)

STORE = Path('data/embeddings_qwen')
MODEL = 'Qwen/Qwen2.5-0.5B'
BATCH = 128
DEVICE = 'cuda'
DIM = 896
N_POS = 8   # must match n_source_positions in synth_contrastive_multisent.yaml

def uniform_bin_hidden(hidden, mask, n_pos):
    """(B, S, D) + (B, S) bool → (B, n_pos, D) by uniform bucket mean-pool."""
    B, S, D = hidden.shape
    lengths = mask.sum(dim=1).long()   # (B,)
    out = torch.zeros(B, n_pos, D, device=hidden.device, dtype=hidden.dtype)
    counts = torch.zeros(B, n_pos, device=hidden.device, dtype=hidden.dtype)
    pos = torch.arange(S, device=hidden.device).unsqueeze(0).expand(B, -1)  # (B, S)
    L = lengths.unsqueeze(1).clamp(min=1).float()    # (B, 1)
    bin_idx = (pos.float() * n_pos / L).long().clamp(max=n_pos - 1)  # (B, S)
    bin_idx = bin_idx.masked_fill(~mask.bool(), n_pos)               # trash bin
    ext = torch.zeros(B, n_pos + 1, D, device=hidden.device, dtype=hidden.dtype)
    ext.scatter_add_(1, bin_idx.unsqueeze(-1).expand(-1, -1, D), hidden)
    cnt = torch.zeros(B, n_pos + 1, device=hidden.device, dtype=hidden.dtype)
    cnt.scatter_add_(1, bin_idx, torch.ones_like(bin_idx, dtype=hidden.dtype))
    out = ext[:, :n_pos] / cnt[:, :n_pos].unsqueeze(-1).clamp(min=1)
    # L2-normalise each position independently; zero bins → zero vector after
    # normalize (avoids NaN from div-by-zero on empty bins).
    out = torch.nn.functional.normalize(out, dim=-1)
    out = torch.nan_to_num(out, nan=0.0)
    return out  # (B, n_pos, D)

log.info('Loading passages...')
texts = [json.loads(l)['text'] for l in (STORE / 'passages.jsonl').open()]
N = len(texts)
log.info(f'{N} passages, n_pos={N_POS}')

log.info('Loading model...')
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL, dtype=torch.float16).to(DEVICE).eval()

emb = np.zeros((N, N_POS, DIM), dtype=np.float16)
with torch.no_grad():
    for i in range(0, N, BATCH):
        batch = texts[i:i+BATCH]
        enc = tok(batch, padding=True, truncation=True, max_length=512,
                  return_tensors='pt').to(DEVICE)
        out = model(**enc)
        pooled = uniform_bin_hidden(
            out.last_hidden_state,
            enc['attention_mask'],
            N_POS,
        ).half().cpu().numpy()
        emb[i:i+len(batch)] = pooled
        if (i // BATCH) % 50 == 0:
            log.info(f'  {i+len(batch)}/{N}')

log.info(f'Saving {STORE}/embeddings.npy  shape={emb.shape}...')
np.save(STORE / 'embeddings.npy', emb)
log.info('Done.')
