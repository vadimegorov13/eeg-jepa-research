"""Part A: fix the rich-embedding hook in liu2024_sjepa_embeddings_lda.ipynb (in place)."""
import json

NB = "liu2024_sjepa_embeddings_lda.ipynb"
nb = json.load(open(NB))

def src(i): return ''.join(nb['cells'][i]['source'])
def setsrc(i, s): nb['cells'][i]['source'] = s.splitlines(keepends=True)

# ---- Cell 4: add embedding_hook key + richer pool comment ----
c4 = src(4)
old_pool = '    "embedding_pool": "mean",  # \'mean\' = mean over token sequence; \'flatten\' = full flatten\n'
assert old_pool in c4, "pool line not found"
new_pool = (
    '    # Rich embedding via forward hook (resolved limitation, see markdown below).\n'
    '    "embedding_hook": "feature_encoder",  # \'feature_encoder\' (rich tokens) or \'spatial_conv\'\n'
    '    "embedding_pool": "mean",  # \'mean\'|\'max\'|\'meanmax\' over tokens, or \'flatten\'\n'
)
c4 = c4.replace(old_pool, new_pool)
setsrc(4, c4)

# ---- Cell 12: replace extract_embeddings with a real forward-hook implementation ----
c12 = src(12)
start = c12.index('@torch.no_grad()')
end_marker = 'return embeddings.astype(np.float32)'
end = c12.index(end_marker) + len(end_marker)
new_fn = '''@torch.no_grad()
def extract_embeddings(model, X, cfg, device):
    """
    Extract per-trial RICH embeddings from S-JEPA PreLocal via a forward hook.

    PreLocal forward is: spatial_conv -> feature_encoder -> final_layer (2-D head).
    We hook an intermediate module and pool to a fixed (n_trials, D) matrix:
      cfg['embedding_hook']:
        'feature_encoder' (default): rich local-token tensor (B, n_tokens, emb_dim) -> pool over tokens
        'spatial_conv'             : spatially-filtered signal (B, n_spat_filters, n_times) -> pool over time
      cfg['embedding_pool']:
        'mean' | 'max' | 'meanmax' over the pooled axis, or 'flatten' (full flatten).
    Returns: numpy (n_trials, D).  (D probed at runtime; no longer the 2-D logits.)
    """
    model.eval()
    pool = cfg.get("embedding_pool", "mean")
    hook_name = cfg.get("embedding_hook", "feature_encoder")
    target = getattr(model, hook_name, None)
    if target is None:
        raise AttributeError(f"model has no submodule '{hook_name}' to hook")

    captured = {}
    def _hook(module, inp, out):
        captured["z"] = out.detach()
    handle = target.register_forward_hook(_hook)

    ds = TrialDataset(X, np.zeros(len(X), dtype=np.int64))
    dl = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False)
    all_embs = []
    try:
        for xb, _ in dl:
            xb = xb.to(device)
            _ = model(xb)                      # full forward; hook captures the intermediate
            z = captured["z"]
            if z.dim() == 2:
                feat = z
            else:
                # feature_encoder -> pool over tokens (dim 1); spatial_conv -> pool over time (dim 2)
                axis = 2 if hook_name == "spatial_conv" else 1
                if pool == "flatten":
                    feat = z.flatten(start_dim=1)
                elif pool == "max":
                    feat = z.max(dim=axis).values.flatten(start_dim=1)
                elif pool == "meanmax":
                    feat = torch.cat([z.mean(dim=axis), z.max(dim=axis).values], dim=-1).flatten(start_dim=1)
                else:  # mean
                    feat = z.mean(dim=axis).flatten(start_dim=1)
            all_embs.append(feat.cpu().numpy())
    finally:
        handle.remove()

    embeddings = np.concatenate(all_embs, axis=0).astype(np.float32)
    if not hasattr(extract_embeddings, "_printed"):
        print(f"  [hook={hook_name} pool={pool}] embedding matrix: {embeddings.shape}")
        extract_embeddings._printed = True
    return embeddings'''
c12 = c12[:start] + new_fn + c12[end:]
setsrc(12, c12)

# ---- Cell 13: update markdown ----
setsrc(13, (
"> **Embedding dimensionality (resolved).** `extract_embeddings` now registers a forward hook on an\n"
"> intermediate S-JEPA module instead of using the 2-D class logits. With `embedding_hook=\"feature_encoder\"`\n"
"> (default) it captures the rich local-token tensor `(batch, n_tokens, emb_dim=64)` and pools over the token\n"
"> axis (`embedding_pool` = `mean`/`max`/`meanmax`/`flatten`) to a fixed per-trial vector (D = 64 for `mean`,\n"
"> ~1024 for `flatten`). `embedding_hook=\"spatial_conv\"` instead captures the spatially-filtered signal.\n"
"> This is the meaningful representation the LDA probe and the hybrid's S-JEPA branch need; the old 2-D-logit\n"
"> path is gone. Leakage is unchanged: spatial_conv fine-tune, PCA, and LDA are still fit on the train fold only.\n"
))

# ---- Insert frozen-embedding cache cell after the run-all cell (index 19) ----
cache_cell = {
 "cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
 "source": (
'''# --- Cache FROZEN rich embeddings per subject (for the TWFB hybrid, Branch A) ---
# Deterministic, label-free: uses the frozen pretrained BASE_MODEL (no per-fold fine-tune),
# so each trial maps to one fixed embedding the hybrid can load directly.
if CONFIG.get("save_embeddings", False):
    emb_dir = ARTIFACT_ROOT / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)
    frozen = copy.deepcopy(BASE_MODEL).to(DEVICE).eval()
    manifest = {"hook": CONFIG.get("embedding_hook", "feature_encoder"),
                "pool": CONFIG.get("embedding_pool", "mean"),
                "window_samples": WINDOW_SAMPLES, "sfreq_model": SFREQ_MODEL,
                "mi_window_s": list(CONFIG["mi_window_s"]), "subjects": []}
    n_saved = 0
    for sid in SUBJECT_IDS:
        mat_path = sid_to_path.get(sid)
        if mat_path is None:
            continue
        try:
            Xs, ys = preprocess_subject(mat_path)
            Es = extract_embeddings(frozen, Xs, CONFIG, DEVICE)   # (40, D), frozen
            np.savez(emb_dir / f"sub-{sid:02d}.npz", X=Es, y=ys)
            manifest["subjects"].append(int(sid)); manifest["embedding_dim"] = int(Es.shape[1])
            n_saved += 1
        except Exception as exc:
            print(f"  Sub {sid:02d}: embedding cache error — {exc}")
    json.dump(manifest, open(emb_dir / "manifest.json", "w"), indent=2)
    print(f"Cached frozen embeddings for {n_saved} subjects (D={manifest.get('embedding_dim')}) -> {emb_dir}")
else:
    print("save_embeddings=False -> skipping frozen embedding cache")'''
 ).splitlines(keepends=True)
}
nb['cells'].insert(20, cache_cell)

json.dump(nb, open(NB, "w"), indent=1)
print(f"edited {NB}: now {len(nb['cells'])} cells")
