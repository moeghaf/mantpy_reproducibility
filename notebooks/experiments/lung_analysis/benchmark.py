"""Seeded synthetic label-corruption generator (ECM patches or cells) and the historical-overlay mode.

Pre-registered rules (spec/protocol.json):
* budget = fraction of eligible units (signal patches / cells) per ROI, hit exactly;
* ``block``: square blocks (side drawn from ``block_sides``) centred on positions whose block
  contains >= ``min_units`` eligible units; one replacement label per block drawn uniformly from the
  full label set; units already carrying that label are skipped; blocks are drawn until the budget is
  reached and the last block is truncated at random to hit it exactly;
* ``isolated``: units drawn uniformly, each with a replacement drawn uniformly from the *other* labels;
* ``mixed``: half of the budget as blocks, half isolated;
* ``within_group`` (reverse task variant): replacement drawn from the other labels of the same group.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass
class Corruption:
    is_artifact: np.ndarray        # bool (n,)
    corrupted: np.ndarray          # int (n,) observed labels after corruption (-1 kept for background)
    pristine: np.ndarray           # int (n,)
    blocks: list[dict]             # block records (centre, side, label, n_flipped)
    seed: int
    kind: str

    def transition_table(self, n_labels: int) -> np.ndarray:
        t = np.zeros((n_labels, n_labels), int)
        a = self.is_artifact
        np.add.at(t, (self.pristine[a], self.corrupted[a]), 1)
        return t


def _draw_block_flips(rng: np.random.Generator, xy: np.ndarray, eligible: np.ndarray, pristine: np.ndarray,
                      n_labels: int, side: float, min_units: int, max_tries: int = 60) -> tuple[np.ndarray, int, tuple[float, float]]:
    """Return (indices to flip, new label, centre) for one block or (empty, -1, centre) if none found.
    Block centres are drawn at the location of a random eligible unit (plus jitter within half a side),
    so blocks land on tissue; a cKDTree makes each attempt O(log n)."""
    from scipy.spatial import cKDTree
    half = side / 2.0
    elig_idx = np.flatnonzero(eligible)
    if elig_idx.size == 0:
        return np.zeros(0, int), -1, (float("nan"), float("nan"))
    tree = cKDTree(xy[elig_idx])
    for _ in range(max_tries):
        anchor = xy[rng.choice(elig_idx)]
        cx, cy = anchor[0] + rng.uniform(-half, half), anchor[1] + rng.uniform(-half, half)
        cand = tree.query_ball_point([cx, cy], r=half * np.sqrt(2.0))
        cand = elig_idx[np.asarray(cand, int)] if len(cand) else np.zeros(0, int)
        inside = cand[(np.abs(xy[cand, 0] - cx) <= half) & (np.abs(xy[cand, 1] - cy) <= half)]
        if inside.size < min_units:
            continue
        new = int(rng.integers(0, n_labels))
        flips = inside[pristine[inside] != new]
        if flips.size == 0:
            continue
        return flips, new, (float(cx), float(cy))
    return np.zeros(0, int), -1, (float("nan"), float("nan"))


def corrupt_labels(xy: np.ndarray, pristine: np.ndarray, n_labels: int, kind: str, seed: int, *,
                   budget_fraction: float = 0.10, block_sides: Sequence[float] = (50.0, 100.0, 150.0),
                   min_units: int = 20, groups: np.ndarray | None = None) -> Corruption:
    """``pristine`` uses -1 for ineligible units (background). ``groups`` (optional, same length) restricts
    replacement labels for kind='within_group' to labels sharing the unit's group (label->group map given
    as an array indexed by label)."""
    rng = np.random.default_rng(seed)
    xy = np.asarray(xy, float); pristine = np.asarray(pristine, int)
    eligible = pristine >= 0
    n_elig = int(eligible.sum())
    budget = int(round(budget_fraction * n_elig))
    corrupted = pristine.copy()
    is_art = np.zeros(pristine.size, bool)
    blocks: list[dict] = []

    def flip_isolated(n_flip: int, within_group: bool) -> None:
        pool = np.flatnonzero(eligible & ~is_art)
        if n_flip <= 0 or pool.size == 0:
            return
        pick = rng.choice(pool, size=min(n_flip, pool.size), replace=False)
        for i in pick:
            cands = np.arange(n_labels)
            if within_group and groups is not None:
                cands = cands[groups[cands] == groups[pristine[i]]]
            cands = cands[cands != pristine[i]]
            if cands.size == 0:
                continue
            corrupted[i] = int(rng.choice(cands)); is_art[i] = True

    n_block_budget = budget if kind == "block" else (budget // 2 if kind == "mixed" else 0)
    n_iso_budget = budget - n_block_budget if kind in ("mixed", "isolated", "within_group") else 0

    tries = 0
    while is_art.sum() < n_block_budget and tries < 200:
        tries += 1
        side = float(rng.choice(np.asarray(block_sides, float)))
        flips, new, centre = _draw_block_flips(rng, xy, eligible & ~is_art, pristine, n_labels, side, min_units)
        if flips.size == 0:
            continue
        remaining = n_block_budget - int(is_art.sum())
        if flips.size > remaining:
            flips = rng.choice(flips, size=remaining, replace=False)
        corrupted[flips] = new; is_art[flips] = True
        blocks.append({"centre": centre, "side": side, "new_label": new, "n_flipped": int(flips.size)})
    flip_isolated(n_iso_budget if kind != "within_group" else budget, within_group=(kind == "within_group"))
    return Corruption(is_artifact=is_art, corrupted=corrupted, pristine=pristine, blocks=blocks, seed=seed, kind=kind)


def corruption_frame(c: Corruption, ids: np.ndarray, roi: str) -> pd.DataFrame:
    return pd.DataFrame({"roi": roi, "unit_id": ids, "pristine": c.pristine, "corrupted": c.corrupted, "is_artifact": c.is_artifact,
                         "kind": c.kind, "seed": c.seed})


def historical_overlay_flips(xy: np.ndarray, pristine: np.ndarray, boxes: list[dict]) -> Corruption:
    """Reproduce the deposited (historical) benchmark from artifact_boxes.json records
    ({center:[x,y], half, new_cluster}) on the given pristine labels."""
    corrupted = pristine.copy(); is_art = np.zeros(pristine.size, bool); recs = []
    for b in boxes:
        cx, cy = b["center"]; half = b["half"]; new = int(b["new_cluster"])
        inside = (pristine >= 0) & (np.abs(xy[:, 0] - cx) <= half) & (np.abs(xy[:, 1] - cy) <= half) & (pristine != new)
        corrupted[inside] = new; is_art |= inside
        recs.append({"centre": (cx, cy), "side": 2 * half, "new_label": new, "n_flipped": int(inside.sum())})
    return Corruption(is_artifact=is_art, corrupted=corrupted, pristine=pristine, blocks=recs, seed=-1, kind="historical")
