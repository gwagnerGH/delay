"""Save physical and consumption states from a solved revised EZDelay policy.

Temperatures are not nodes in the saved damage tree. The TCRE column is a
clearly labeled best-estimate emissions-based diagnostic, not a realization
of the uncertain temperature/damage process used for optimization.
"""

import csv

import numpy as np


def _write_rows(path, rows):
    if not rows:
        raise ValueError(f"No diagnostic rows for {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export_state_trees(objective, policy, output_dir, expected_welfare):
    """Export decision-node physical variables and every five-year utility state."""
    tree = objective.tree
    emissions = objective.emit
    welfare, tape, consumption, utility, shares, _, _ = objective._forward(policy)
    if not np.isclose(welfare, expected_welfare, rtol=0, atol=1e-9):
        raise RuntimeError("Exported consumption does not reproduce saved welfare")

    concentration = np.asarray(objective.damage.ghg_level(policy), float)
    if concentration.shape != (tree.num_decision_nodes,):
        raise RuntimeError("Node concentration count differs from decision tree")
    blocks = {block["period"]: block for block in tape["local_blocks"]}
    calendar_blocks = tape.get("calendar_blocks")
    decision_rows = []
    for period, elapsed in enumerate(tree.decision_times[:-1]):
        first, last = tree.get_nodes_in_period(period)
        year = int(tree.base_year+elapsed)
        bau = float(emissions.baseline_gtco2_periods[period])
        block = (None if elapsed == 0 else calendar_blocks[elapsed]
                 if calendar_blocks is not None else blocks[period])
        for state, node in enumerate(range(first, last+1)):
            path, _ = emissions.get_mitigated_baseline(
                policy, node=node, baseline="cumemit")
            cumulative_1000_gtco2 = float(path[-1])
            current_m = float(policy[node])
            if calendar_blocks is not None:
                climate_state = (None if block is None else block["states"][state])
                low_ghg_fraction = (0.0 if climate_state is None else
                                    climate_state["cooling"])
                warming_fraction = (0.0 if climate_state is None else
                                    climate_state["warming"])
            else:
                low_ghg_fraction = (0.0 if node == 0 else float(1.0/(1.0+np.exp(
                    0.05*(concentration[node]-200.0)))))
                warming_fraction = (0.0 if node == 0 else float(
                    objective.damage.pchip_value_slope(
                        cumulative_1000_gtco2, node, period)[0]))
                if block is not None:
                    warming_fraction += float(block["states"][state].get("tip_damage", 0.0))
            if block is not None and not np.isclose(warming_fraction+low_ghg_fraction,
                              block["damage"][state], rtol=1e-9, atol=1e-9):
                raise RuntimeError("Physical damage components do not match optimizer")
            decision_rows.append(dict(
                year=year, node=node, state_index=state,
                probability=float(tree.node_prob[node]),
                ghg_ppm=float(concentration[node]),
                baseline_gtco2_per_year=bau,
                net_gtco2_per_year=bau*(1.0-current_m),
                gross_abatement_gtco2_per_year=bau*min(current_m, 1.0),
                removal_gtco2_per_year=bau*max(current_m-1.0, 0.0),
                cumulative_emissions_1000_gtco2=cumulative_1000_gtco2,
                tcre_best_estimate_warming_since_base_year_c=(
                    cumulative_1000_gtco2-emissions.CUMEMIT_BASE_YEAR)
                    *objective.damage.climate.TCRE_BEST_ESTIMATE,
                climate_damage_fraction_including_low_ghg=float(
                    warming_fraction+low_ghg_fraction),
                warming_damage_fraction=warming_fraction,
                low_ghg_penalty_fraction=low_ghg_fraction,
                abatement_share_of_pre_abatement_consumption=float(shares[elapsed][state]),
                pre_abatement_consumption_model_units=float(tape["cons_tree"][elapsed][state]),
                consumption_model_units=float(consumption[elapsed][state]),
                continuation_utility_model_units=float(utility[elapsed][state]),
            ))
    _write_rows(output_dir/"decision_state_nodes.csv", decision_rows)

    five_year_rows = []
    for elapsed in consumption.periods:
        period = min(int(np.searchsorted(tree.decision_times, elapsed,
                                         side="right")-1), tree.num_periods-1)
        first, last = tree.get_nodes_in_period(period)
        controls = last-first+1
        values = consumption[elapsed]
        expanded = len(values) == 2*controls
        if not expanded and len(values) != controls:
            raise RuntimeError("Cannot map five-year consumption to decision nodes")
        probability_period = period+1 if expanded else period
        prob_first, _ = tree.get_nodes_in_period(probability_period)
        prob = tree.node_prob[prob_first:prob_first+len(values)]
        if len(prob) != len(values) or not np.isclose(prob.sum(), 1.0):
            raise RuntimeError("Five-year state probabilities do not sum to one")
        base = tape["cons_tree"][elapsed]
        unit_utility = utility[elapsed]
        share = shares.get(elapsed)
        bau = (None if share is None else float(np.interp(
            tree.base_year+elapsed, emissions.times, emissions.baseline_gtco2)))
        for state in range(len(values)):
            control_node = first+(state//2 if expanded else state)
            mitigation = float(policy[control_node])
            five_year_rows.append(dict(
                year=int(tree.base_year+elapsed), state_index=state,
                state_tree_node=prob_first+state, controlling_decision_node=control_node,
                probability=float(prob[state]),
                is_terminal_continuation=bool(elapsed == consumption.last_period),
                baseline_gtco2_per_year=bau,
                net_gtco2_per_year=(None if bau is None else bau*(1.0-mitigation)),
                removal_gtco2_per_year=(None if bau is None else
                                         bau*max(mitigation-1.0, 0.0)),
                pre_abatement_consumption_model_units=float(base[state]),
                abatement_share_of_pre_abatement_consumption=(
                    None if share is None else float(share[state])),
                consumption_model_units=float(values[state]),
                continuation_utility_model_units=float(unit_utility[state]),
            ))
    _write_rows(output_dir/"five_year_state_nodes.csv", five_year_rows)

    terminal_rows = []
    last_first, last_end = tree.get_nodes_in_period(tree.num_periods-1)
    terminal_block = tape["terminal_block"]
    for state, node in enumerate(range(last_first, last_end+1)):
        cumulative, _ = emissions.get_mitigated_baseline(
            policy, node=node, baseline="cumemit", is_last=True)
        cumulative_1000_gtco2 = float(cumulative[-1])
        terminal_rows.append(dict(
            year=int(tree.base_year+tree.decision_times[-1]),
            state_index=state, controlling_decision_node=node,
            probability=float(tree.node_prob[node]),
            ghg_ppm=float(objective.damage.climate.get_conc_at_node(
                policy, node, is_last=True)),
            cumulative_emissions_1000_gtco2=cumulative_1000_gtco2,
            tcre_best_estimate_warming_since_base_year_c=(
                cumulative_1000_gtco2-emissions.CUMEMIT_BASE_YEAR)
                *objective.damage.climate.TCRE_BEST_ESTIMATE,
            terminal_damage_fraction_including_low_ghg=float(
                terminal_block["damage"][state]),
            terminal_consumption_model_units=float(consumption.last[state]),
            terminal_continuation_utility_model_units=float(utility.last[state]),
        ))
    _write_rows(output_dir/"terminal_state_nodes.csv", terminal_rows)
    return len(decision_rows), len(five_year_rows), len(terminal_rows)
