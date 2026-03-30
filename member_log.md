# Group Member Progress Log
## Kylie Lin
1. Implemented deterministic reproducibility controls across the ML pipeline using a shared seed (`42`) and startup seed initialization.
2. Added `set_reproducibility(seed=42)` in `utils.py` to seed Python `random`, NumPy, and Torch (including CUDA when available).
3. Updated PCA flow in `viz.py` to use deterministic solver settings and fixed PCA sign indeterminacy so 2D map orientation is stable across runs.
4. Updated K-means in `clustering.py` to deterministic settings (`random_state`, fixed `n_init`, `lloyd`) and added canonical label remapping so cluster IDs stay stable.
5. Wired reproducibility controls into app startup in `app.py` via `RANDOM_SEED = 42` and passed the seed into clustering.
6. Added exact genre filtering logic in `app.py` by tokenizing and normalizing genre strings, then matching selected genres via set intersection (replacing substring matching).
7. Verified reproducibility with a Git Bash `check_reproducibility` function run (Exit Code: 0), confirming stable outputs: `labels_equal True` and `pca_equal True`. 