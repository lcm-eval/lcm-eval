from typing import List

import matplotlib.pyplot as plt
import pandas as pd

from cross_db_benchmark.datasets.datasets import Database
from classes.classes import FlatModelConfig, ScaledPostgresModelConfig, MSCNModelConfig, E2EModelConfig, \
    ZeroShotModelConfig, DACEModelConfig, QPPNetModelConfig, QPPModelNoCostsConfig, DACEModelNoCostsConfig
from classes.workloads import EvalWorkloads
from evaluation.eval import Evaluator
from evaluation.evaluation_metrics import QError, PickRate, SelectedRuntime
import seaborn as sns
sns.set_theme(style="whitegrid", font_scale=1.8)

def plot_with_gap(ax, data, fmt, gap_after=5, gap_size=0.5, color=[],  **kwargs):
    bar_locations = []
    for i, (label, value) in enumerate(data.items()):
        c = color[i]
        if i > gap_after:
            i += gap_size  # Add gap_size to the x-coordinate of the bars after the 5th
        bar = ax.bar(i, value, color=c, **kwargs)
        bar_locations.append(bar[0].get_x() + bar[0].get_width() / 2.0)  # Get the center of the bar
        if i == 0:
            ax.axhline(value, color='black', linestyle='--')
        ax.bar_label(bar, label_type='edge', fmt=fmt)

    # Set the x-ticks to the center of the bars
    ax.set_xticks(bar_locations)
    ax.set_xticklabels(data.index)


def summary_plot_seaborn(results: List[dict], databases: List[Database], model_confs):
    fontsize = 14
    results = pd.DataFrame.from_dict(results)
    results["database"] = results["workload"].str.rsplit('_', n=1).str[0]
    results["workload"] = results["workload"].str.rsplit('_', n=1).str[-1]

    model_colors = {model_config.display_name: model_config.color for model_config in model_confs if
                    hasattr(model_config, 'name') and model_config.name is not None}

    fig, (upper_axs, lower_axs) = plt.subplots(2, 3, figsize=(15, 4), sharex="col", dpi=100)

    for i, database in enumerate(databases):
        pick_rate, runtime = upper_axs[i], lower_axs[i]

        # Prepare dataframes
        result_df = results[results["database"] == database.db_name]
        percentage_true = result_df.groupby('model_name')['pick_rate'].mean() * 100
        percentage_true = percentage_true.reindex(model_colors.keys())
        runtimes = result_df.groupby('model_name')['runtime'].sum()
        runtimes = runtimes.reindex(model_colors.keys())

        runtimes = runtimes.to_frame()
        percentage_true = percentage_true.to_frame()
        print(percentage_true)
        # Plot the pick rate
        sns.barplot(data=percentage_true, x="model_name", y="pick_rate", ax=pick_rate, palette=model_colors, width=1.0, log_scale=(False, False), hue="model_name")
        xlim = pick_rate.get_xlim()
        pick_rate.set_ylim(0, 100)
        pick_rate.set_xlabel('')
        pick_rate.grid(True, axis='x', which='minor')
        pick_rate.grid(True, axis='x', which='major')
        pick_rate.axhline(percentage_true[percentage_true["model_name"] == "Sc. Postgres"].value, color='black', linestyle='--')
        pick_rate.axvspan(xmin=6.5, xmax=xlim[1], alpha=0.1, color='gray')
        pick_rate.set_title(database.display_name)
        pick_rate.legend()

        # Plot the runtimes
        sns.barplot(x=runtimes.index, y=runtimes.values, ax=runtime, palette=model_colors, width=1.0, log_scale=(False, False), hue=runtimes.index)
        runtime.set_xlabel('')
        runtime.set_xticklabels([])
        runtime.axhline(runtimes[0], color='black', linestyle='--')
        runtime.grid(True, axis='x', which='minor')
        runtime.grid(True, axis='x', which='major')
        runtime.axvspan(xmin=6.5, xmax=xlim[1], alpha=0.1, color='gray')
        runtime.legend()
    upper_axs[0].set_ylabel('Pick Rate (%)', fontsize=fontsize*1.5)
    lower_axs[0].set_ylabel('Runtime (s)', fontsize=fontsize*1.5)
    #upper_axs[2].legend(title='Model', loc='upper left', bbox_to_anchor=(1, 1), fontsize=fontsize*1.5)
    plt.subplots_adjust(hspace=0.1)
    plt.show()

if __name__ == '__main__':
    evaluator = Evaluator()
    metrics = [QError(), PickRate(), SelectedRuntime()]
    workloads =[EvalWorkloads.PhysicalPlan.imdb, EvalWorkloads.PhysicalPlan.tpc_h, EvalWorkloads.PhysicalPlan.baseball]
    databases = [Database("scale", display_name="IMDB"), Database("tpc_h", display_name="TPC-H"), Database("baseball", display_name="Baseball")]

    model_confs = [
        ScaledPostgresModelConfig(),
        FlatModelConfig(),
        MSCNModelConfig(),
        E2EModelConfig(),
        QPPNetModelConfig(),
        ZeroShotModelConfig(),
        DACEModelConfig(),
        QPPModelNoCostsConfig(),
        DACEModelNoCostsConfig(),
    ]

    seeds = [0, 1, 2]
    for workload in workloads:
        evaluator.eval(workloads=workload,
                       metrics=metrics,
                       plot_single_workloads=True,
                       plot_limit=5,
                       seeds=[0, 1, 2],
                       model_configs=model_confs)
    #summary_plot_seaborn(evaluator.metric_collection, databases, model_confs)
