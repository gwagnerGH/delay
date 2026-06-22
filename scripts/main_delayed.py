#!/usr/bin/env python

import pickle
import pprint
import os

import numpy as np

from _project_paths import configure_paths
configure_paths()

from src.tree import TreeModel
from src.emit_baseline import BPWEmissionBaseline
from src.cost import BPWCost
from src.climate import BPWClimate
from src.damage import BPWDamage
from src.utility import EZUtility
from src.analysis.climate_output import ClimateOutput
from src.analysis.delayed_action import get_delay_nodes, ConstraintAnalysis
from src.config import DEFAULT_DECISION_TIMES
from src.tools import import_csv
from src.optimization import GeneticAlgorithm, GradientSearch


def damage_cache_tag(decision_times, prob_scale=1.0):
    dt_tag = '-'.join(str(int(x)) for x in decision_times)
    prob_tag = f"{float(prob_scale):g}".replace('.', 'p')
    return f"_DT{dt_tag}_PS{prob_tag}"


def main():
    optimize = True
    test_run = False
    import_damages = False

    delay_years = 5

    if test_run:
        print("\n\n***WARNING --- RUNNING WITH LIMITED NUMBER OF ITERATIONS FOR\
              TEST PURPOSES***\n\n")
        N_generations_ga = 2
        N_iters_gs = 2
    else:
        N_generations_ga = 150 # 150
        N_iters_gs = 100 # 100

    data_csv_file = 'research_runs'
    header, indices, data = import_csv(data_csv_file, delimiter=',', indices=2)

    desired_runs = [0]
    for i in desired_runs:
        name = indices[i][1]

        ra, eis, pref, growth, tech_chg, tech_scale, dam_func,\
            baseline_num, tip_on, bs_premium, d_unc, t_unc,\
            no_free_lunch = data[i]

        baseline_num = int(baseline_num)
        dam_func = int(dam_func)
        tip_on = int(tip_on)
        d_unc = int(d_unc)
        t_unc = int(t_unc)
        no_free_lunch = int(no_free_lunch)

        print('**Running job:       ', name, '\n**Model Parameters are:')
        model_params = [ra, eis, pref, growth, tech_chg, tech_scale,\
                        dam_func, baseline_num, tip_on, bs_premium, d_unc,
                        t_unc, no_free_lunch]
        pprint.pprint(set(zip(header, model_params)))

        decision_times_delay = DEFAULT_DECISION_TIMES.copy()
        if delay_years > 0:
            decision_times_delay[1] = delay_years
        decision_times_baseline = decision_times_delay.copy()

        t_baseline = TreeModel(decision_times=decision_times_baseline,
                      prob_scale=1.0)

        t_delay = TreeModel(decision_times=decision_times_delay,
                      prob_scale=1.0)

        baseline_emission_model_baseline = BPWEmissionBaseline(tree=t_baseline,
                                                      baseline_num=baseline_num)
        baseline_emission_model_baseline.baseline_emission_setup()

        baseline_emission_model_delay = BPWEmissionBaseline(tree=t_delay,
                                                      baseline_num=baseline_num)
        baseline_emission_model_delay.baseline_emission_setup()

        draws = 3 * 10**6
        climate_baseline = BPWClimate(t_baseline, baseline_emission_model_baseline, draws=draws)
        climate_delay = BPWClimate(t_delay, baseline_emission_model_delay, draws=draws)

        emit_at_0_baseline = np.interp(2030, baseline_emission_model_baseline.times,
                              baseline_emission_model_baseline.baseline_gtco2)
        c_baseline = BPWCost(t_baseline, emit_at_0=emit_at_0_baseline, baseline_num=baseline_num,
                    tech_const=tech_chg, tech_scale=tech_scale,
                    cons_at_0=86252.0, # 2025 estimated from https://data.worldbank.org/indicator/NE.CON.TOTL.CD
                    backstop_premium=bs_premium,
                    no_free_lunch=no_free_lunch)

        emit_at_0_delay = np.interp(2030, baseline_emission_model_delay.times,
                              baseline_emission_model_delay.baseline_gtco2)
        c_delay = BPWCost(t_delay, emit_at_0=emit_at_0_delay, baseline_num=baseline_num,
                    tech_const=tech_chg, tech_scale=tech_scale,
                    cons_at_0=86252.0, # 2025 estimated from https://data.worldbank.org/indicator/NE.CON.TOTL.CD
                    backstop_premium=bs_premium,
                    no_free_lunch=no_free_lunch)



        d_m = 0.1
        mitigation_constants = np.arange(0, 1 + d_m, d_m)[::-1]

        print("Unconstrained Damage Simulation:")
        df_baseline = BPWDamage(tree=t_baseline, emit_baseline=baseline_emission_model_baseline,
                       climate=climate_baseline, mitigation_constants=mitigation_constants,
                       draws=draws)

        print("Delayed Damage Simulation:")
        df_delay = BPWDamage(tree=t_delay, emit_baseline=baseline_emission_model_delay,
                       climate=climate_delay, mitigation_constants=mitigation_constants,
                       draws=draws)


        baseline_damage_file_tag = damage_cache_tag(
            decision_times_baseline, t_baseline.prob_scale
        )
        delay_damage_file_tag = damage_cache_tag(
            decision_times_delay, t_delay.prob_scale
        )

        damsim_filename_baseline = ''.join(["simulated_damages_df", str(dam_func),
                                   "_TP", str(tip_on), "_SSP", str(baseline_num),
                                   "_BY", str(t_baseline.base_year),
                                   "_dunc", str(d_unc), "_tunc", str(t_unc),
                                   baseline_damage_file_tag])

        damsim_filename_delay = ''.join(["simulated_damages_df", str(dam_func),
                                   "_TP", str(tip_on), "_SSP", str(baseline_num),
                                   "_BY", str(t_delay.base_year),
                                   "_dunc", str(d_unc), "_tunc", str(t_unc),
                                   delay_damage_file_tag])

        if import_damages:
            df_baseline.import_damages(file_name=damsim_filename_baseline)
            df_delay.import_damages(file_name=damsim_filename_delay)
        else:
            df_baseline.damage_simulation(filename=damsim_filename_baseline, save_simulation=True,
                                 dam_func=dam_func, tip_on=tip_on, d_unc=d_unc,
                                 t_unc=t_unc)
            df_delay.damage_simulation(filename=damsim_filename_delay, save_simulation=True,
                                 dam_func=dam_func, tip_on=tip_on, d_unc=d_unc,
                                 t_unc=t_unc)


        if optimize:

            u_baseline = EZUtility(tree=t_baseline, damage=df_baseline, cost=c_baseline, period_len=5.0, eis=eis, ra=ra,
                          time_pref=pref, cons_growth=growth)

            u_delay = EZUtility(tree=t_delay, damage=df_delay, cost=c_delay, period_len=5.0, eis=eis, ra=ra,
                          time_pref=pref, cons_growth=growth)

            # No constraints for optimal scenario
            fixed_indices_opt = None
            fixed_values_opt = None

            ga_model_opt = GeneticAlgorithm(
                pop_amount=400,
                num_generations=N_generations_ga,
                cx_prob=0.8,
                mut_prob=0.50,
                bound=1.5,
                num_feature=t_baseline.num_decision_nodes,
                utility=u_baseline,
                fixed_values=fixed_values_opt,
                fixed_indices=fixed_indices_opt,
                print_progress=True
            )

            gs_model_opt = GradientSearch(
                var_nums=t_baseline.num_decision_nodes,
                utility=u_baseline,
                accuracy=5.e-7,
                iterations=N_iters_gs,
                fixed_values=fixed_values_opt,
                fixed_indices=fixed_indices_opt,
                print_progress=True
            )

            final_pop_opt, fitness_opt = ga_model_opt.run()
            sort_pop_opt = final_pop_opt[np.argsort(fitness_opt)][::-1]

            # Run gradient search
            m_optimal, u_optimal = gs_model_opt.run(initial_point_list=sort_pop_opt, topk=1)

            # Validate mitigation vector shape and values.
            # Values above 1.0 represent negative emissions via the backstop.
            assert m_optimal.ndim == 1 and m_optimal.size == t_baseline.num_decision_nodes, \
                f"m_optimal shape mismatch: expected ({t_baseline.num_decision_nodes},), got {m_optimal.shape}"
            assert np.isfinite(m_optimal).all(), \
                f"m_optimal contains non-finite values: {m_optimal[~np.isfinite(m_optimal)]}"

            assert (m_optimal >= 0).all() and (m_optimal <= 1.5).all(), \
                f"m_optimal out of bounds [0,1.5]: min={m_optimal.min():.6f}, max={m_optimal.max():.6f}"

            print(f"Mitigation vector validated: shape={m_optimal.shape}, range=[{m_optimal.min():.6f}, {m_optimal.max():.6f}]")

            print("Unconstrained scenario:")
            print(f"First-period mitigation:  {m_optimal[0]:.6f}")
            print(f"Carbon price:             ${c_baseline.price(0, m_optimal[0], 0):.2f} per ton")
            print(f"Utility:                  {u_optimal:.10f}")

            output_prefix_opt = f'{name}_optimal'

            filelist = [f for f in os.listdir('data/') if f.startswith(output_prefix_opt) and
                        f.endswith('.csv')]
            for f in filelist:
                print(f'Removing file data/{f}')
                os.remove('data/'+f)

            co_opt = ClimateOutput(u_baseline)
            co_opt.calculate_output(m_optimal)
            co_opt.save_output(m_optimal, prefix=output_prefix_opt)

            p_opt = {}
            p_opt['df.d_rcomb'] = df_baseline.d_rcomb
            p_opt['m_opt'] = m_optimal
            p_opt['u_opt'] = u_optimal
            p_opt['delay_action'] = False
            p_opt['delay_periods'] = 0

            picklename_opt = f'data/{name}_optimal_log.pickle'
            with open(picklename_opt, 'wb') as handle:
                pickle.dump(p_opt, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(f'Optimal scenario saved as {picklename_opt}\n')

            print(f"Running delayed action scenario with {delay_years}-year delay...\n")

            fixed_indices_delay = get_delay_nodes(t_delay, 1)
            fixed_values_delay = np.zeros(len(fixed_indices_delay))

            print(f"Total decision nodes:        {t_delay.num_decision_nodes}")
            print(f"Number of nodes constrained: {len(fixed_indices_delay)}")
            print(f"Constrained node indices:    {fixed_indices_delay}\n")

            ga_model_delay = GeneticAlgorithm(
                pop_amount=400,
                num_generations=N_generations_ga,
                cx_prob=0.8,
                mut_prob=0.50,
                bound=1.5,
                num_feature=t_delay.num_decision_nodes,
                utility=u_delay,
                fixed_values=fixed_values_delay,
                fixed_indices=fixed_indices_delay,
                print_progress=True
            )

            gs_model_delay = GradientSearch(
                var_nums=t_delay.num_decision_nodes,
                utility=u_delay,
                accuracy=5.e-7,
                iterations=N_iters_gs,
                fixed_values=fixed_values_delay,
                fixed_indices=fixed_indices_delay,
                print_progress=True
            )

            final_pop_delay, fitness_delay = ga_model_delay.run()
            sort_pop_delay = final_pop_delay[np.argsort(fitness_delay)][::-1]

            m_delayed, u_delayed = gs_model_delay.run(initial_point_list=sort_pop_delay, topk=1)

            assert m_delayed.ndim == 1 and m_delayed.size == t_delay.num_decision_nodes, \
                f"m_delayed shape mismatch: expected ({t_delay.num_decision_nodes},), got {m_delayed.shape}"
            assert np.isfinite(m_delayed).all(), \
                f"m_delayed contains non-finite values: {m_delayed[~np.isfinite(m_delayed)]}"
            assert (m_delayed >= 0).all() and (m_delayed <= 1.5).all(), \
                f"m_delayed out of bounds [0,1.5]: min={m_delayed.min():.6f}, max={m_delayed.max():.6f}"

            for idx in fixed_indices_delay:
                assert abs(m_delayed[idx]) < 1e-10, \
                    f"Constrained node {idx} not zero: m_delayed[{idx}]={m_delayed[idx]:.10f}"
            print(f"Mitigation vector validated: shape={m_delayed.shape}, range=[{m_delayed.min():.6f}, {m_delayed.max():.6f}]")
            print(f"All {len(fixed_indices_delay)} constrained nodes verified at zero")

            print("Delayed action scenario:")
            print(f"First-period mitigation: {m_delayed[0]:.6f} (constrained to 0)")
            print(f"Carbon price: ${c_delay.price(0, m_delayed[0], 0):.2f} per ton")
            print(f"Utility: {u_delayed:.10f}")

            output_prefix_delay = f'{name}_delayed_{delay_years}periods'

            filelist = [f for f in os.listdir('data/') if f.startswith(output_prefix_delay) and
                        f.endswith('.csv')]
            for f in filelist:
                print(f'Removing file data/{f}')
                os.remove('data/'+f)

            co_delay = ClimateOutput(u_delay)
            co_delay.calculate_output(m_delayed)
            co_delay.save_output(m_delayed, prefix=output_prefix_delay)

            p_delay = {}
            p_delay['df.d_rcomb'] = df_delay.d_rcomb
            p_delay['m_opt'] = m_delayed
            p_delay['u_opt'] = u_delayed
            p_delay['delay_action'] = True
            p_delay['delay_periods'] = 1

            picklename_delay = f'data/{name}_delayed_{delay_years}periods_log.pickle'
            with open(picklename_delay, 'wb') as handle:
                pickle.dump(p_delay, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(f'Delayed scenario saved as {picklename_delay}\n')

            print("Constraint Analysis:")

            ca = ConstraintAnalysis(u_delay, u_baseline, m_delayed, m_optimal)
            ca.save_output(prefix=f'{name}_delayed_{delay_years}periods')


            print("Final summary")
            print(f"\nOptimization Results:")
            print(f"  Optimal first-period mitigation:   {m_optimal[0]:.6f}")
            print(f"  Delayed first-period mitigation:   {m_delayed[0]:.6f}")
            print(f"  Mitigation foregone:               {m_optimal[0] - m_delayed[0]:.6f}")

            print(f"\nUtility Comparison:")
            print(f"  Optimal utility:                   {u_optimal:.10f}")
            print(f"  Delayed utility:                   {u_delayed:.10f}")
            print(f"  Utility loss:                      {ca.con_cost:.10f}")
            print(f"  Relative loss:                     {(ca.con_cost/u_optimal)*100:.4f}%")

            print(f"\nEconomic Impacts:")
            if ca.delta_c is not None:
                print(f"  Consumption compensation (abs):    {ca.delta_c:.6f}")
                print(f"  Compensation (% of year 0 cons):   {ca.delta_c_pct:.4f}%")
                print(f"  Compensation (billions $):         ${ca.delta_c_billions:.2f}B")
                print(f"  5-year DWL (% of year 0 cons):     {ca.delta_c_5yr_pct:.4f}%")
                print(f"  5-year DWL annual flow ($B):       ${ca.delta_c_5yr_billions:.2f}B")
            else:
                print(f"  Consumption compensation:          FAILED TO CONVERGE")
            print(f"  Emission reduction foregone:       {ca.delta_emission_gton:.4f} Gt CO2")

            if ca.deadweight is not None:
                print(f"\nDeadweight Analysis:")
                print(f"  Deadweight cost:                   ${ca.deadweight:.2f} per ton CO2")

            print(f"  Optimal scenario:      data/{output_prefix_opt}_*.csv")
            print(f"  Delayed scenario:      data/{output_prefix_delay}_*.csv")
            print(f"  Constraint analysis:   data/{output_prefix_delay}_constraint_output.csv")
            print(f"  Optimal pickle:        {picklename_opt}")
            print(f"  Delayed pickle:        {picklename_delay}")


if __name__ == "__main__":
    main()
