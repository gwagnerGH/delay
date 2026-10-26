"""Prepare or reuse saved, specification-matched tree/damage robustness inputs.

The cluster dispatcher referenced this module but it was absent from the
checkout. Fixed-policy damage simulations are independent of preferences and
post-terminal growth; their cache key deliberately excludes these parameters.
Preparation happens before solver timing and under a per-grid file lock.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np

from src.climate import BPWClimate
from src.damage_simulation import DamageSimulation
from src.revised_experiment import FlatAfter2100Emissions
from src.tree import TreeModel

ROOT = Path(__file__).resolve().parents[1]


def ensure_grid(kind, spec, decision_times, prob_scale, damage_settings):
    source = ROOT / "data/revised_default/full_damages.npz"
    with np.load(source, allow_pickle=False) as saved:
        knots = saved["knots"].copy()
        draws = int(saved["draws"])
        seed = int(saved["seed"])
    inputs = dict(kind=kind, spec=spec, decision_times=list(decision_times),
                  prob_scale=float(prob_scale), damage_settings=damage_settings,
                  draws=draws, seed=seed, flat_after_2100=True, version=1)
    key = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()[:16]
    folder = ROOT / "data/revised_default/robustness" / f"{kind}_{key}"
    folder.mkdir(parents=True, exist_ok=True)
    damage_file, start_file = folder / "damages.npz", folder / "start.npz"
    with (folder / "prepare.lock").open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        if not damage_file.is_file():
            if os.environ.get("REQUIRE_DAMAGE_IMPORT") == "1":
                raise FileNotFoundError(
                    f"Missing matched robustness grid {damage_file}; prepare once "
                    "without REQUIRE_DAMAGE_IMPORT=1 before submitting import-only jobs")
            tree = TreeModel(decision_times, prob_scale=prob_scale)
            emissions = FlatAfter2100Emissions(tree, 2, emissions_time_step=5)
            emissions.baseline_emission_setup()
            climate = BPWClimate(tree, emissions, draws=draws)
            simulation = DamageSimulation(tree, emissions, climate, draws, knots,
                **{key: damage_settings[key] for key in ("dam_func", "tip_on", "d_unc", "t_unc")})
            # Serial generation makes numpy and Python RNG seeds reproducible
            # across machines; this cache is generated once, outside solves.
            numpy_state, python_state = np.random.get_state(), random.getstate()
            try:
                np.random.seed(seed)
                random.seed(seed)
                damages = np.array([simulation._run_path(m) for m in knots])
            finally:
                np.random.set_state(numpy_state)
                random.setstate(python_state)
            temporary = folder / "damages.tmp.npz"
            np.savez_compressed(temporary, damages=damages, knots=knots,
                draws=draws, seed=seed, decision_times=decision_times, prob_scale=prob_scale)
            temporary.replace(damage_file)
            (folder / "inputs.json").write_text(json.dumps(inputs, indent=2)+"\n")
        if not start_file.is_file():
            tree = TreeModel(decision_times, prob_scale=prob_scale)
            # Feasible neutral start, not a previously optimized comparator.
            temporary = folder / "start.tmp.npz"
            np.savez_compressed(temporary, mitigation=np.full(tree.num_decision_nodes, 0.6))
            temporary.replace(start_file)
    return damage_file, start_file
