"""Part A (cont.): fix the data loader (descend into eeg struct) and onset-align the window."""
import json
NB = "liu2024_sjepa_embeddings_lda.ipynb"
nb = json.load(open(NB))
def src(i): return ''.join(nb['cells'][i]['source'])
def setsrc(i, s): nb['cells'][i]['source'] = s.splitlines(keepends=True)

# ---- Cell 4: window start 0.0 -> 1.5 s (onset is at ~2.0 s; matches the augmented notebook) ----
c4 = src(4)
old = '    "mi_window_s": (0.0, 4.0),     # seconds after MI onset\n'
assert old in c4
new = ('    # MI cue (marker==2) sits at ~2.0 s in the trial; start at 1.5 s so the 4.2 s window is\n'
       '    # onset-aligned (1.5-5.7 s) instead of the pre-imagery period. Matches the augmented notebook.\n'
       '    "mi_window_s": (1.5, 5.7),\n')
setsrc(4, c4.replace(old, new))

# ---- Cell 8: loader descends into the 'eeg' mat_struct for rawdata/label ----
c8 = src(8)
old_block = (
'    raw_data = mat.get("rawdata", mat.get("data", None))\n'
'    if raw_data is None:\n'
'        for k, v in mat.items():\n'
'            if not k.startswith("__") and isinstance(v, np.ndarray) and v.ndim == 3:\n'
'                raw_data = v\n'
'                break\n'
)
assert old_block in c8, "loader block not found"
new_block = (
'    raw_data = mat.get("rawdata", mat.get("data", None))\n'
'    labels   = mat.get("labels", mat.get("label", None))\n'
'    # Liu2024 figshare files nest arrays under an \'eeg\' struct: eeg.rawdata / eeg.label\n'
'    if raw_data is None or labels is None:\n'
'        for k, v in mat.items():\n'
'            if k.startswith("__"):\n'
'                continue\n'
'            if hasattr(v, "_fieldnames"):\n'
'                if raw_data is None and "rawdata" in v._fieldnames:\n'
'                    raw_data = getattr(v, "rawdata")\n'
'                if labels is None and "label" in v._fieldnames:\n'
'                    labels = getattr(v, "label")\n'
'            elif raw_data is None and isinstance(v, np.ndarray) and v.ndim == 3:\n'
'                raw_data = v\n'
)
c8 = c8.replace(old_block, new_block)
# remove the later re-fetch of labels that would overwrite the struct-resolved labels with None
c8 = c8.replace(
'    labels = mat.get("labels", mat.get("label", None))\n    y = np.asarray(labels, dtype=int).ravel()\n',
'    y = np.asarray(labels, dtype=int).ravel()\n')
setsrc(8, c8)

json.dump(nb, open(NB, "w"), indent=1)
print("loader + window fixed")
