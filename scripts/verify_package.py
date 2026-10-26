"""Check data coverage, manuscript figures, and packaged file checksums."""

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS = (
    "download-delay-frontier-main", "download-delay-frontier-robustness",
    "download-tree", "download-preference", "download-technology",
    "download-partial", "download-damage", "download-backstop",
    "download-ensemble", "download-mac-shift", "download-future-growth",
)
PAPER_FIGURES = (
    "delay_frontier_three_panel.eps",
    "policy_reentry_paths_first_panel.eps",
    "partial_mitigation_four_panel.eps",
    "uncertainty_main_text_panel.pdf",
    "nfl_mac_and_shift_outcomes_four_panel.eps",
    "appendix_carbon_price_trees_optimal_delayed.eps",
    "robustness_ranges_four_panel.pdf",
    "appendix_model_outputs.pdf",
    "appendix_model_outputs_growth.pdf",
)


def require(path):
    if not path.is_file():
        raise FileNotFoundError(path)


def require_csv(bundle, stem):
    folder = ROOT / "data" / "R_and_R" / bundle
    if not any((folder / name).is_file() for name in (stem, stem + ".gz")):
        raise FileNotFoundError(folder / stem)


def main():
    root_scripts = [path.name for path in ROOT.iterdir()
                    if path.is_file() and path.suffix in (".py", ".sh")]
    if root_scripts:
        raise ValueError(f"Executable scripts belong in scripts/: {root_scripts}")
    if list((ROOT / "scripts").glob("*shapley*")):
        raise ValueError("Unrelated Shapley runners are outside the paper package")
    require(ROOT / "data" / "revised_default" / "full_damages.npz")
    require(ROOT / "data" / "revised_default" / "full_baseline_start.npz")
    require(ROOT / "notebooks" / "paper_facing_plots.ipynb")
    for bundle in RUNS:
        require(ROOT / "data" / "R_and_R" / bundle / "README.json")
        require_csv(bundle, "certified_results.csv")
        if bundle != "download-ensemble":
            require_csv(bundle, "certified_node_metrics.csv")
            require_csv(bundle, "certified_node_prices.csv")
        else:
            require_csv(bundle, "certified_period_outcomes.csv")
    for figure in PAPER_FIGURES:
        require(ROOT / "paper_outputs" / "figures" / figure)
    from reproduce_main_spec import CLUSTER_RUNS
    for run in CLUSTER_RUNS:
        require(Path(run["command"]))
    for file in (
        "quantity_learning_summary.json", "cost_floor_2026-10-22/summary.csv",
        "cost_floor_2026-10-22/trajectories.csv",
    ):
        require(ROOT / "data" / "si_sensitivities" / file)
    manifest = ROOT / "SHA256SUMS"
    require(manifest)
    count = 0
    for line in manifest.read_text().splitlines():
        expected, relative = line.split("  ", 1)
        path = ROOT / relative
        require(path)
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError(f"Checksum mismatch: {relative}")
        count += 1
    print(f"Verified {len(RUNS)} certified bundles, {len(PAPER_FIGURES)} manuscript figures, and {count} file checksums.")


if __name__ == "__main__":
    main()
