import torch, yaml, sys
import torch.nn.functional as F
from collections import Counter
sys.path.insert(0, "/workspace/lfm/src")
from lfm.generator.dep_tree_vae.config import DepTreeVAEConfig
from lfm.generator.dep_tree_vae.model import DepTreeVAE
from lfm.generator.dep_tree_vae.data import DepTreeDataset, collate_dep_tree, build_dataloaders
from lfm.generator.dep_tree_vae.trainer import _greedy_decode
from lfm.generator.dep_tree_vae.skeleton import SKEL_BOS, SKEL_EOS
from lfm.generator.dep_tree_vae.config import DEP_RELATIONS
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

with torch.no_grad():
    z_base = torch.randn(n, cfg.latent.total_dim, device=device)

    # Perturbations
    z_struct_pert = z_base.clone()
    z_struct_pert[:, :sd] += eps * torch.randn(n, sd, device=device)

    z_content_pert = z_base.clone()
    z_content_pert[:, sd:] += eps * torch.randn(n, cfg.latent.content_dim, device=device)

    # Decode all three
    base_results = _greedy_decode(model, z_base, device, cfg, sp)
    struct_results = _greedy_decode(model, z_struct_pert, device, cfg, sp)
    content_results = _greedy_decode(model, z_content_pert, device, cfg, sp)

    # Get skeletons for all
    def get_skeleton(z):
        z_s, _ = model.latent.split(z)
        skel = model.skeleton_decoder(z_s)[0]
        roles_list = []
        for i in range(z.size(0)):
            roles = []
            for t in skel[i]:
                v = t.item()
                if v == SKEL_BOS: continue
                if v == SKEL_EOS: break
                if v < len(DEP_RELATIONS): roles.append(v)
            roles_list.append(tuple(roles))
        return roles_list

    skel_base = get_skeleton(z_base)
    skel_struct = get_skeleton(z_struct_pert)
    skel_content = get_skeleton(z_content_pert)

    # 1. Skeleton change rate
    skel_change_struct = sum(1 for a, b in zip(skel_base, skel_struct) if a != b) / n
    skel_change_content = sum(1 for a, b in zip(skel_base, skel_content) if a != b) / n

    # 2. Word-level Jaccard similarity (token overlap)
    def jaccard(text_a, text_b):
        a = set(text_a.split())
        b = set(text_b.split())
        if not a and not b: return 1.0
        return len(a & b) / max(len(a | b), 1)

    word_sim_struct = np.mean([jaccard(base_results[i][0], struct_results[i][0]) for i in range(n)])
    word_sim_content = np.mean([jaccard(base_results[i][0], content_results[i][0]) for i in range(n)])

    # 3. Length change
    len_base = [len(r[0].split()) for r in base_results]
    len_struct = [len(r[0].split()) for r in struct_results]
    len_content = [len(r[0].split()) for r in content_results]
    len_change_struct = np.mean([abs(a - b) for a, b in zip(len_base, len_struct)])
    len_change_content = np.mean([abs(a - b) for a, b in zip(len_base, len_content)])

    # 4. Cross-recombination: struct from A, content from B
    z_cross = torch.cat([z_base[:n//2, :sd], z_base[n//2:, sd:]], dim=-1)
    cross_results = _greedy_decode(model, z_cross, device, cfg, sp)
    skel_cross = get_skeleton(z_cross)

    # Does cross-recombo have A's skeleton?
    skel_match_a = sum(1 for a, c in zip(skel_base[:n//2], skel_cross) if a == c) / (n//2)
    # Word overlap with B (content donor)?
    word_sim_with_b = np.mean([jaccard(cross_results[i][0], base_results[i + n//2][0]) for i in range(n//2)])
    # Word overlap with A (struct donor)?
    word_sim_with_a = np.mean([jaccard(cross_results[i][0], base_results[i][0]) for i in range(n//2)])

print(f"\n=== Disentanglement Metrics (eps={eps}, n={n}) ===")
print(f"\nSkeleton change rate:")
print(f"  struct perturb: {skel_change_struct:.0%} changed  (should be HIGH)")
print(f"  content perturb: {skel_change_content:.0%} changed  (should be LOW)")

print(f"\nWord Jaccard similarity to base:")
print(f"  struct perturb: {word_sim_struct:.2f}  (should be HIGH — same words)")
print(f"  content perturb: {word_sim_content:.2f}  (should be LOW — different words)")

print(f"\nLength change from base:")
print(f"  struct perturb: ±{len_change_struct:.1f} words  (structure changes length)")
print(f"  content perturb: ±{len_change_content:.1f} words  (content shouldn't change length much)")

print(f"\nCross-recombination (struct from A, content from B):")
print(f"  skeleton matches A: {skel_match_a:.0%}  (should be HIGH)")
print(f"  word overlap with B (content donor): {word_sim_with_b:.2f}  (should be HIGHER)")
print(f"  word overlap with A (struct donor): {word_sim_with_a:.2f}  (should be LOWER)")
