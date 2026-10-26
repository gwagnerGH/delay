#!/usr/bin/env python3
"""Plot completed full-tree EZDelay cost-floor sensitivity results."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    args = parser.parse_args()
    summary = read(args.results/'summary.csv')
    trajectories = read(args.results/'trajectories.csv')
    colors = {0.0: '#424242', 0.25: '#2474a6', 0.4: '#d67622', 0.6: '#84388d'}
    plt.rcParams.update({'figure.dpi': 160, 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})

    fig, ax = plt.subplots(figsize=(6.2, 4.1))
    for floor in sorted({float(r['cost_floor']) for r in summary}):
        rows = sorted((r for r in summary if float(r['cost_floor']) == floor),
                      key=lambda r: int(r['delay_years']))
        ax.plot([int(r['delay_years']) for r in rows],
                [float(r['dwl_pct_per_year']) for r in rows],
                '-o', color=colors[floor], label=f'{floor:.0%} floor')
    ax.set(xlabel='Delay (years)', ylabel='Annual consumption-equivalent DWL (%)',
           xticks=[5, 10, 15])
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.results/'dwl_by_cost_floor.png')
    plt.close(fig)

    grouped = defaultdict(list)
    for row in trajectories:
        grouped[(float(row['cost_floor']), row['policy'])].append(row)
    for policy, label in [('optimal', 'no_delay'), ('delay_10', 'ten_year_delay')]:
        for metric, title, scale in [
            ('mean_mitigation', 'Mitigation rate', 1),
            ('mean_net_emissions', 'Net emissions (GtCO₂/year)', 1),
            ('mean_removal', 'Carbon removal (GtCO₂/year)', 1)]:
            fig, ax = plt.subplots(figsize=(7.0, 4.1))
            for floor in sorted({key[0] for key in grouped if key[1] == policy}):
                rows = sorted(grouped[floor, policy], key=lambda r: int(r['year']))
                rows = [r for r in rows if r.get(metric, '') != '']
                if not rows:
                    continue
                years = [int(r['year']) for r in rows]
                key = metric.replace('mean_', '')
                mean = [float(r[metric])*scale for r in rows]
                low = [float(r[f'p10_{key}'])*scale for r in rows]
                high = [float(r[f'p90_{key}'])*scale for r in rows]
                ax.plot(years, mean, color=colors[floor], lw=1.8,
                        label=f'{floor:.0%} floor')
                ax.fill_between(years, low, high, color=colors[floor], alpha=.10,
                                linewidth=0)
            ax.set(xlabel='Year', ylabel=title, xlim=(2025, 2250))
            if metric == 'mean_mitigation':
                ax.axhline(1, color='black', lw=.7, alpha=.4)
            ax.legend(frameon=False, ncol=2)
            fig.tight_layout()
            fig.savefig(args.results/f'{label}_{key}.png')
            plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.1))
    for floor in sorted({key[0] for key in grouped if key[1] == 'optimal'}):
        rows = [r for r in grouped[floor, 'optimal'] if r.get('mean_interval_mac', '') != '']
        ax.plot([int(r['year']) for r in rows],
                [float(r['mean_interval_mac']) for r in rows], '-o',
                color=colors[floor], label=f'{floor:.0%} floor')
    ax.set(xlabel='Decision year', ylabel='Mean interval carbon price ($/tCO₂)',
           xlim=(2025, 2250))
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.results/'optimal_carbon_prices.png')
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.2), sharex=True, sharey=True)
    styles = [('optimal', 'No delay', '#303030', '-'),
              ('delay_5', '5-year delay', '#2474a6', '--'),
              ('delay_10', '10-year delay', '#d67622', '-.'),
              ('delay_15', '15-year delay', '#84388d', ':')]
    legend_lines = []
    for ax, floor in zip(axes.flat, sorted({key[0] for key in grouped})):
        for policy, label, color, linestyle in styles:
            rows = sorted((row for row in grouped[floor, policy]
                           if row.get('mean_interval_mac', '') != ''),
                          key=lambda row: int(row['year']))
            line, = ax.plot([int(row['year']) for row in rows],
                            [float(row['mean_interval_mac']) for row in rows],
                            marker='o', ms=3.6, lw=1.8, linestyle=linestyle,
                            color=color, label=label)
            if len(legend_lines) < len(styles):
                legend_lines.append(line)
        ax.set(title=f'{floor:.0%} cost floor', xlim=(2025, 2200), ylim=(0, 1100),
               xticks=[2025, 2060, 2100, 2150, 2200])
        ax.grid(axis='y', alpha=.16)
    for ax in axes[:, 0]:
        ax.set_ylabel('Mean interval carbon price ($/tCO₂)')
    for ax in axes[-1, :]:
        ax.set_xlabel('Decision year')
    fig.legend(legend_lines, [style[1] for style in styles],
               loc='upper center', ncol=4, frameon=False, bbox_to_anchor=(.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, .94), h_pad=1.4, w_pad=1.2)
    fig.savefig(args.results/'optimal_and_delayed_carbon_prices_by_floor.png')
    plt.close(fig)


if __name__ == '__main__':
    main()
