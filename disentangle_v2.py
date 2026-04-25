import torch, yaml, sys
sys.path.insert(0, "/workspace/lfm/src")
from lfm.generator.dep_tree_vae.config import DepTreeVAEConfig
from lfm.generator.dep_tree_vae.model import DepTreeVAE
from lfm.generator.dep_tree_vae.data import build_dataloaders
from lfm.generator.dep_tree_vae.trainer import _greedy_decode
from pathlib import Path
import numpy as np

cfg_dict = yaml.safe_load(open("/workspace/lfm/configs/dep_tree_vae_vast.yaml"))
cfg_dict.pop("model_type", None)
cfg = DepTreeVAEConfig(**cfg_dict)
device = torch.device("cuda")
_, _, sp, vocab_size = build_dataloaders(cfg)
model = DepTreeVAE(cfg, vocab_size).to(device)
ckpt = torch.load("/workspace/lfm/data/models/dep_tree_vae_v1/resume.pt", map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state"], strict=False)
model.eval()
print(f"Step {ckpt['global_step']}")

sd = cfg.latent.struct_dim
n = 32
eps = 0.5

def jaccard(a, b, min_len=0):
    wa = set(w for w in a.split() if len(w) >= min_len)
    wb = set(w for w in b.split() if len(w) >= min_len)
    if not wa and not wb: return 1.0
    return len(wa & wb) / max(len(wa | wb), 1)

with torch.no_grad():
    z_base = torch.randn(n, cfg.latent.total_dim, device=device)
    z_sp = z_base.clone(); z_sp[:, :sd] += eps * torch.randn(n, sd, device=device)
    z_cp = z_base.clone(); z_cp[:, sd:] += eps * torch.randn(n, cfg.latent.content_dim, device=device)

    base = _greedy_decode(model, z_base, device, cfg, sp)
    struct = _greedy_decode(model, z_sp, device, cfg, sp)
    content = _greedy_decode(model, z_cp, device, cfg, sp)

for min_len in [0, 4, 6]:
    js = np.mean([jaccard(base[i][0], struct[i][0], min_len) for i in range(n)])
    jc = np.mean([jaccard(base[i][0], content[i][0], min_len) for i in range(n)])
    label = f"all words" if min_len == 0 else f"words>={min_len} chars"
    gap = js - jc
    print(f"Jaccard ({label:>16}):  struct_perturb={js:.3f}  content_perturb={jc:.3f}  gap={gap:+.3f}  {'GOOD' if gap > 0.03 else 'WEAK'}")
