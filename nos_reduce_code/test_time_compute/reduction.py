"""Dictionary-reduction strategies for NOS test-time beam search.

A "dictionary" is represented by two parallel structures:

  latent   : torch.Tensor of shape (N, d), in the latent space consumed by
             test_time_compute.ttc_methods.beam_search_operator_selection_batch
  metadata : list of N dicts, one per operator, carrying physical labels
             (``advection_speed`` / ``diffusion`` for advection-diffusion) and
             a stable ``operator_id``.

Every strategy returns ``(reduced_latent, reduced_metadata, trace)``:

  reduced_latent   : (N', d) tensor in the same latent space as ``latent``
  reduced_metadata : list of N' dicts (labels used for parameter identification)
  trace            : dict describing how the N' operators map back to the full
                     dictionary, used for the traceability report

``random`` (with ``seed=42`` this reproduces the paper's Appendix C.5
subsampling) and ``frequency`` return an exact subset of the original
operators (traceable). ``cluster`` returns fused, averaged operators whose
membership is many-to-one (traceability degraded; recorded separately).
"""

from __future__ import annotations

import numpy as np
import torch


def _as_float(v):
    if hasattr(v, "item"):
        return float(v.item())
    return float(v)


def subset_from_indices(latent, metadata, indices):
    """Build a sub-dictionary defined by ``indices`` into the full dictionary."""
    if isinstance(indices, torch.Tensor):
        indices = indices.detach().cpu().tolist()
    indices = [int(i) for i in indices]

    reduced_latent = latent[indices]
    reduced_metadata = []
    for new_id, old_id in enumerate(indices):
        row = dict(metadata[old_id])
        row["operator_id"] = new_id
        row["source_operator_id"] = old_id
        reduced_metadata.append(row)

    trace = {
        "type": "subset",
        "indices": indices,
        "source_operator_ids": [metadata[i].get("operator_id", i) for i in indices],
    }
    return reduced_latent, reduced_metadata, trace


def random_reduce(latent, metadata, n_operators, seed=0):
    """Uniform random retention without replacement (indices sorted).

    This is also the paper's Appendix C.5 subsampling method: the released
    ``test_time_compute/analysis/sweep_ad_subsampling.py`` draws
    ``sorted(rng.choice(N, size=N', replace=False))`` with
    ``rng = np.random.default_rng(42)``. Use ``seed=42`` to reproduce C.5.
    """
    n_full = latent.shape[0]
    n = min(int(n_operators), n_full)
    if n >= n_full:
        indices = list(range(n_full))
    else:
        rng = np.random.default_rng(int(seed))
        indices = sorted(int(i) for i in rng.choice(n_full, size=n, replace=False))
    return subset_from_indices(latent, metadata, indices)


def frequency_reduce(model, latent, metadata, heldout_loader, n_operators,
                     beam_width=4, max_operators=5, dt=10.0 / 100,
                     beam_batch_size=32, device=None):
    """Keep the operators most frequently selected by beam search on held-out
    training-distribution trajectories.

    An operator is counted when it appears in the final selected composition
    (argmin) of the beam search for a held-out trajectory.
    """
    if device is None:
        from test_time_compute.ttc_utils import DEVICE
        device = DEVICE

    from test_time_compute.ttc_methods import beam_search_operator_selection_batch

    latent_dev = latent.detach().to(device)
    counts = np.zeros(latent.shape[0], dtype=np.float64)

    model.eval()
    with torch.no_grad():
        for batch in heldout_loader:
            inp = batch["input"].to(device)
            tgt = batch["target"].to(device)
            for i in range(inp.size(0)):
                composition, _err, _pred = beam_search_operator_selection_batch(
                    model, latent_dev, inp[i:i + 1], tgt[i:i + 1],
                    beam_width=beam_width, max_operators=max_operators,
                    dt=dt, batch_size=beam_batch_size)
                for op in composition:
                    counts[int(op)] += 1.0

    n = min(int(n_operators), latent.shape[0])
    order = np.argsort(-counts, kind="stable")[:n]
    kept = [int(i) for i in order]
    reduced_latent, reduced_metadata, trace = subset_from_indices(latent, metadata, kept)
    trace["type"] = "frequency"
    trace["counts"] = {str(k): float(counts[k]) for k in kept}
    return reduced_latent, reduced_metadata, trace


def _operator_val_errors(model, latent, val_loader, dt, device, dim,
                         max_val_samples=None):
    """Per-operator mean single-step NRMSE on the validation set."""
    from einops import rearrange
    from test_time_compute.ttc_methods import get_state_labels
    from test_time_compute.ttc_utils import get_relative_l2_error

    rel = get_relative_l2_error()
    xs, ys = [], []
    state_labels = None

    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            inp = batch["input"].to(device)
            tgt = batch["target"].to(device)
            if state_labels is None:
                state_labels = get_state_labels(inp)
            xs.append(rearrange(inp[:, :-1], "b t ... -> (b t) ..."))
            ys.append(rearrange(inp[:, 1:], "b t ... -> (b t) ..."))
            if max_val_samples is not None and sum(x.shape[0] for x in xs) >= max_val_samples:
                break

    x_val = torch.cat(xs, 0)
    y_val = torch.cat(ys, 0)
    if max_val_samples is not None:
        x_val = x_val[:max_val_samples]
        y_val = y_val[:max_val_samples]

    all_ops = model.decode_theta(latent.detach().to(device), dim)
    errors = np.empty(latent.shape[0], dtype=np.float64)
    with torch.no_grad():
        for i in range(latent.shape[0]):
            op = all_ops[i].unsqueeze(0).repeat(x_val.shape[0], 1)
            pred, _ = model.solve_ode(x_val, op, state_labels, dim=dim,
                                      integration_time=dt, n_future_steps=1, dt=dt)
            errors[i] = rel(pred, y_val).item()
    return errors


def cluster_reduce(model, latent, metadata, n_operators, val_loader=None,
                   pca_dim=None, dt=10.0 / 100, dim=1, device=None,
                   weighting="error", max_val_samples=64, random_state=0):
    """PCA -> k-means -> weighted average of the latent codes per cluster.

    Weights are proportional to ``1 / single-step NRMSE`` of each operator on
    the validation set. If ``val_loader`` is ``None`` or ``weighting`` is not
    ``"error"``, uniform averaging is used as a sensitivity check.
    """
    if device is None:
        from test_time_compute.ttc_utils import DEVICE
        device = DEVICE

    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    X = latent.detach().cpu().numpy().astype(np.float32)
    N = X.shape[0]
    n = min(int(n_operators), N)

    if pca_dim is None:
        pca_dim = min(32, n, X.shape[1])
    pca_dim = max(1, min(int(pca_dim), X.shape[1], N))
    Xp = PCA(n_components=pca_dim, random_state=random_state).fit_transform(X) \
        if pca_dim < X.shape[1] else X.copy()

    km = KMeans(n_clusters=n, n_init=10, random_state=random_state).fit(Xp)
    labels = km.labels_

    if weighting == "error" and val_loader is not None:
        errs = _operator_val_errors(model, latent, val_loader, dt, device, dim,
                                    max_val_samples=max_val_samples)
        weights = 1.0 / np.clip(errs, 1e-6, None)
    else:
        weights = np.ones(N, dtype=np.float64)

    lat = latent.detach().cpu().float()
    fused = []
    fused_meta = []
    members = []
    for c in range(n):
        idx = np.where(labels == c)[0]
        wc = weights[idx].astype(np.float64)
        wc = wc / wc.sum()
        wc_t = torch.tensor(wc, dtype=torch.float32).unsqueeze(1)
        fused.append((lat[idx] * wc_t).sum(0))

        member = [metadata[j] for j in idx]
        mrow = {"operator_id": c, "source_operator_ids": [int(j) for j in idx]}
        for key in member[0]:
            if key in ("operator_id", "source_operator_id", "source_operator_ids"):
                continue
            vals = [m[key] for m in member]
            if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
                mrow[key] = float(np.average([_as_float(v) for v in vals], weights=wc))
            else:
                mrow[key] = vals[0]
        fused_meta.append(mrow)
        members.append([int(j) for j in idx])

    fused_latent = torch.stack(fused).to(device)
    trace = {
        "type": "cluster",
        "members": members,
        "source_operator_ids": [
            [metadata[j].get("operator_id", j) for j in idx] for idx in members
        ],
    }
    return fused_latent, fused_meta, trace
