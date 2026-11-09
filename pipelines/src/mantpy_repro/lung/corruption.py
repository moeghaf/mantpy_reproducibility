"""Seeded synthetic corruption of ECM patch labels.

Main generator (:func:`corrupt_labels`): 10% of the signal-patch labels of an ROI are replaced.
``block`` writes one label, drawn uniformly from all clusters, into square blocks of side 50,
100 or 150 pixels; patches that already carry that label are not counted, and the last block
is cut at random so the budget is met exactly. ``isolated`` replaces single patches with a
label drawn from the other clusters. ``mixed`` (the primary type) spends half of the budget on
blocks and half on isolated patches.

The refit control compares two partitions of the same tiles, so it draws the corrupted
*positions* once on their common foreground (:func:`physical_proposal`) and maps the same
random numbers to labels in each partition (:func:`apply_proposal`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import BLOCK_SIDES, BUDGET_FRACTION, MIN_BLOCK_PATCHES


@dataclass
class Corruption:
    """One corrupted copy of an ROI's labels. Background patches keep the label -1."""

    is_artifact: np.ndarray      # bool: the label was replaced
    corrupted: np.ndarray        # observed labels after corruption
    pristine: np.ndarray
    blocks: list[dict]           # centre, side, new label and number of replaced patches of each block
    seed: int
    kind: str


def _draw_block(rng: np.random.Generator, xy: np.ndarray, eligible: np.ndarray, pristine: np.ndarray,
                n_labels: int, side: float, min_units: int, max_tries: int = 60):
    """Patches to replace in one block, its new label and its centre (no patches if none is found).

    The centre is a random eligible patch moved by up to half a side in x and y, so blocks land
    on tissue. A block needs at least ``min_units`` eligible patches and must change a label.
    """
    from scipy.spatial import cKDTree

    half = side / 2.0
    candidates = np.flatnonzero(eligible)
    nowhere = (np.zeros(0, int), -1, (float("nan"), float("nan")))
    if candidates.size == 0:
        return nowhere
    tree = cKDTree(xy[candidates])
    for _ in range(max_tries):
        anchor = xy[rng.choice(candidates)]
        cx, cy = anchor[0] + rng.uniform(-half, half), anchor[1] + rng.uniform(-half, half)
        near = tree.query_ball_point([cx, cy], r=half * np.sqrt(2.0))
        near = candidates[np.asarray(near, int)] if len(near) else np.zeros(0, int)
        inside = near[(np.abs(xy[near, 0] - cx) <= half) & (np.abs(xy[near, 1] - cy) <= half)]
        if inside.size < min_units:
            continue
        new = int(rng.integers(0, n_labels))
        flips = inside[pristine[inside] != new]
        if flips.size == 0:
            continue
        return flips, new, (float(cx), float(cy))
    return nowhere


def corrupt_labels(xy: np.ndarray, pristine: np.ndarray, n_labels: int, kind: str, seed: int, *,
                   budget_fraction: float = BUDGET_FRACTION, block_sides=BLOCK_SIDES,
                   min_units: int = MIN_BLOCK_PATCHES) -> Corruption:
    """Replace ``budget_fraction`` of the signal labels (``pristine >= 0``) of one ROI."""
    if kind not in ("mixed", "block", "isolated"):
        raise ValueError(f"kind must be 'mixed', 'block' or 'isolated', not {kind!r}")
    rng = np.random.default_rng(seed)
    xy, pristine = np.asarray(xy, float), np.asarray(pristine, int)
    eligible = pristine >= 0
    budget = int(round(budget_fraction * int(eligible.sum())))
    corrupted = pristine.copy()
    is_artifact = np.zeros(pristine.size, bool)
    blocks: list[dict] = []
    block_budget = budget if kind == "block" else (budget // 2 if kind == "mixed" else 0)
    isolated_budget = budget - block_budget

    tries = 0
    while is_artifact.sum() < block_budget and tries < 200:
        tries += 1
        side = float(rng.choice(np.asarray(block_sides, float)))
        flips, new, centre = _draw_block(rng, xy, eligible & ~is_artifact, pristine, n_labels, side, min_units)
        if flips.size == 0:
            continue
        remaining = block_budget - int(is_artifact.sum())
        if flips.size > remaining:
            flips = rng.choice(flips, size=remaining, replace=False)
        corrupted[flips] = new
        is_artifact[flips] = True
        blocks.append({"centre": centre, "side": side, "new_label": new, "n_flipped": int(flips.size)})

    pool = np.flatnonzero(eligible & ~is_artifact)
    if isolated_budget > 0 and pool.size:
        for i in rng.choice(pool, size=min(isolated_budget, pool.size), replace=False):
            others = np.arange(n_labels)
            others = others[others != pristine[i]]
            if others.size == 0:
                continue
            corrupted[i] = int(rng.choice(others))
            is_artifact[i] = True
    return Corruption(is_artifact=is_artifact, corrupted=corrupted, pristine=pristine, blocks=blocks, seed=seed,
                      kind=kind)


def make_realisation(cohort, kind: str, seed: int) -> dict[str, Corruption]:
    """One corruption realisation of every ROI. The generator of ROI ``i`` (cohort order) is
    seeded with ``seed * 1000 + i``."""
    return {name: corrupt_labels(roi.patch_xy, roi.label, cohort.n_labels, kind, seed * 1000 + i)
            for i, (name, roi) in enumerate(cohort.rois.items())}


# -------------------------------------------------------- paired corruption of the refit control
def physical_proposal(xy: np.ndarray, common: np.ndarray, kind: str, seed: int, roi: str) -> dict:
    """Corrupted positions on the common foreground of the two scaling arms.

    The generator is seeded with ``seed + 100003 x ROI number``. The budget is 10% of the common
    foreground. Blocks are placed without looking at labels, so a block can propose the label a
    patch already has and the realised corruption can be below 10%.
    """
    if kind not in ("mixed", "block", "isolated"):
        raise ValueError(kind)
    rng = np.random.default_rng(int(seed) + 100003 * int(roi[-3:]))
    budget = int(round(0.1 * int(common.sum())))
    block_budget = budget if kind == "block" else budget // 2 if kind == "mixed" else 0
    isolated_budget = budget - block_budget
    used = np.zeros(len(common), bool)
    blocks, count = [], 0
    for _ in range(200):
        if count >= block_budget:
            break
        pool = np.flatnonzero(common & ~used)
        if not len(pool):
            break
        side = float(rng.choice([50.0, 100.0, 150.0]))
        centre = xy[rng.choice(pool)] + rng.uniform(-side / 2, side / 2, size=2)
        inside = pool[np.all(np.abs(xy[pool] - centre) <= side / 2, axis=1)]
        if len(inside) < 20:
            continue
        if len(inside) > block_budget - count:
            inside = rng.choice(inside, size=block_budget - count, replace=False)
        used[inside] = True
        count += len(inside)
        blocks.append({"indices": inside, "centre": centre.tolist(), "side": side, "uniform": float(rng.random())})
    pool = np.flatnonzero(common & ~used)
    isolated = rng.choice(pool, size=min(isolated_budget, len(pool)), replace=False)
    uniforms = rng.random(len(isolated))
    proposed = used.copy()
    proposed[isolated] = True
    return {"blocks": blocks, "isolated": isolated, "uniforms": uniforms, "proposed": proposed,
            "common": common.copy(), "requested_nodes": budget, "proposed_nodes": int(proposed.sum()),
            "unplaced_block_nodes": block_budget - count}


def apply_proposal(pristine: np.ndarray, n_labels: int, proposal: dict, kind: str, seed: int) -> Corruption:
    """Write labels at the proposed positions of one partition with ``n_labels`` clusters.

    Each block maps its random number to one of the clusters; each isolated patch maps its
    random number to one of the other clusters.
    """
    observed, records = pristine.copy(), []
    for block in proposal["blocks"]:
        replacement = min(n_labels - 1, int(block["uniform"] * n_labels))
        indices = block["indices"]
        observed[indices] = replacement
        records.append({"centre": block["centre"], "side": block["side"], "new_label": replacement,
                        "n_proposed": len(indices), "n_flipped": int(np.sum(pristine[indices] != replacement))})
    indices = proposal["isolated"]
    if len(indices):
        other = np.minimum(n_labels - 2, (proposal["uniforms"] * (n_labels - 1)).astype(int))
        observed[indices] = other + (other >= pristine[indices])
    changed = (observed != pristine) & (pristine >= 0)
    if np.any(changed & ~proposal["common"]):
        raise RuntimeError("A label outside the common foreground was changed.")
    return Corruption(changed, observed, pristine.copy(), records, seed, kind)
