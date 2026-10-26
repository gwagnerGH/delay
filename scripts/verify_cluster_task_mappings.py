#!/usr/bin/env python
"""Print first/last SGE task mappings for EZDelay cluster runners."""

import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def with_task_id(task_id, fn):
    old_task_id = os.environ.get('SGE_TASK_ID')
    os.environ['SGE_TASK_ID'] = str(task_id)
    try:
        return fn()
    finally:
        if old_task_id is None:
            os.environ.pop('SGE_TASK_ID', None)
        else:
            os.environ['SGE_TASK_ID'] = old_task_id


def print_mapping(name, first_task, last_task, mapping_fn):
    print(f"\n{name}")
    print(f"  first task {first_task}: {with_task_id(first_task, mapping_fn)}")
    print(f"  last task  {last_task}: {with_task_id(last_task, mapping_fn)}")


def main():
    from scripts import main_ensemble_delayed_cluster as ensemble
    from scripts import main_mean_parameter_cluster as mean
    from scripts import main_delay_frontier_cluster as frontier
    from scripts import main_delayed_partial as partial
    from scripts import main_tree_robustness_cluster as tree
    from scripts import main_preference_grid_cluster as preference
    from scripts import main_technology_grid_cluster as technology
    from scripts import main_damage_robustness_cluster as damage

    print_mapping(
        'ensemble',
        1,
        ensemble.N_SAMPLES * len(ensemble.delay_years),
        lambda: ensemble.get_cluster_config()[:3],
    )

    print_mapping(
        'mean parameter',
        1,
        len(ensemble.delay_years),
        lambda: mean.get_cluster_config()[:2],
    )

    frontier_vals, frontier_labels = frontier.parameter_values_and_labels()
    frontier_total = frontier.total_sample_count(frontier_vals) * len(frontier.frontier_delay_years())
    print_mapping(
        'delay frontier',
        1,
        frontier_total,
        lambda: frontier.get_cluster_config(frontier_vals, frontier_labels)[:4],
    )

    partial_total = len(partial.runs) * len(partial.delay_years) * len(partial.partial_mitigation_pct)
    print_mapping(
        'partial mitigation',
        1,
        partial_total,
        lambda: partial.get_cluster_config()[:5],
    )

    tree_total = len(ensemble.delay_years) * len(tree.selected_tree_spec_names())
    print_mapping(
        'tree robustness',
        1,
        tree_total,
        lambda: tree.get_cluster_config()[:4],
    )

    pref_vals, pref_labels, pref_levels = preference.preference_grid_values()
    print_mapping(
        'preference grid',
        1,
        len(pref_vals) * len(ensemble.delay_years),
        lambda: preference.get_cluster_config(pref_vals, pref_labels, pref_levels)[:3],
    )

    tech_vals, tech_labels, tech_levels = technology.technology_grid_values()
    print_mapping(
        'technology grid',
        1,
        len(tech_vals) * len(ensemble.delay_years),
        lambda: technology.get_cluster_config(tech_vals, tech_labels, tech_levels)[:3],
    )

    damage_specs = damage.damage_specs()
    print_mapping(
        'damage robustness',
        1,
        len(damage_specs) * len(ensemble.delay_years),
        lambda: damage.get_cluster_config(damage_specs)[:3],
    )


if __name__ == '__main__':
    main()
