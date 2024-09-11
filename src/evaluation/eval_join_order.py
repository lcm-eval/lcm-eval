from typing import List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cross_db_benchmark.datasets.datasets import Database
from classes.classes import MODEL_CONFIGS
from classes.workloads import EvalWorkloads
from evaluation.eval import Evaluator
from evaluation.evaluation_metrics import QError, SpearmanCorrelation, Metric, SelectedRuntime
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.0)


def summary_plot(results: List[dict], databases: List[Database], metrics: List[Metric]):
    results = pd.DataFrame.from_dict(results)
    for database in databases:
        result_df = results[results["database"] == database.db_name].sort_values(by="num_tables")
        result_df["workload"] = result_df["workload"].str.rsplit('_', n=1).str[-1]

        model_colors = {model_config.display_name: model_config.color for model_config in MODEL_CONFIGS if
                        hasattr(model_config, 'name') and model_config.name is not None}

        fig, axs = plt.subplots(nrows=len(metrics),
                                ncols=2,
                                figsize=(10, 6),
                                dpi=100,
                                gridspec_kw={'width_ratios': [5, 1]},
                                sharex="col")

        for i, metric in enumerate(metrics):
            ax = axs[i, 0]
            scores = result_df.groupby(['workload', "num_tables", 'model_name'])[metric.metric_name].mean().unstack()
            scores = scores.sort_values(by="num_tables").reset_index(level=1, drop=True)
            scores = scores[list(model_colors.keys())]
            scores.plot(kind='bar',
                         ax=ax,
                         color= [model_colors[column] for column in scores.columns.tolist() if column in model_colors],
                         width=0.8,
                         edgecolor=None)

            if metric.logscale:
                ax.set_yscale("log")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=10)
            ax.legend(ncol=2, fontsize=6, loc="upper left")
            ax.set_ylabel(metric.display_name)
            ax.set_axisbelow(True)
            ax.set_ylim(metric.y_min, metric.y_max)
            ax.set_xlabel("Workload #ID")
            ax.set_xticks(np.arange(len(scores)) + 0.5, minor=True)
            ax.set_xticks(np.arange(len(scores)))
            ax.set_xticklabels(scores.index, rotation=0, fontsize=10)
            ax.set_xlim(-0.5, 8.5)
            ax.grid(True, axis='x', which='minor')
            ax.grid(False, axis='x', which='major')

            # Plot the average scores for each model
            ax2 = axs[i, 1]

            avg_scores = result_df.groupby('model_name').apply(
                lambda x: pd.Series(
                    {metric.metric_name: getattr(x[metric.metric_name], metric.aggregation)() for metric in metrics}))
            avg_scores = avg_scores.reindex(model_colors.keys())
            avg_scores[metric.metric_name].plot(kind='bar',
                                                ax=ax2,
                                                color=model_colors.values(),
                                                width=0.8,
                                                edgecolor=None)

            if metric.logscale:
                ax2.set_yscale("log")
            ax2.set_ylim(metric.y_min, metric.y_max)
            ax2.set_axisbelow(True)
            ax2.set_xticks([])
            ax2.bar_label(ax2.containers[0], label_type='center', fmt='{:,.2f}', rotation=90, padding=0)
            ax2.set_xlabel("")
            ax2.grid(color="black", linewidth=0.2, which='both')
            ax2.yaxis.tick_right()

            if metric.aggregation == 'mean':
                ax2.set_ylabel(f'Avg. {metric.display_name}')
            else:
                ax2.set_ylabel(f'Total {metric.display_name}')

        # Draw additional lines
        for i in range(0, axs.shape[0]):
            axs[i, 0].axvline(x=2.5, color='black', linewidth=1.5, linestyle='--')
            axs[i, 0].axvline(x=5.5, color='black', linewidth=1.5, linestyle='--')

        axs[0, 0].annotate(text="3 Tables",
                           xy=(0.2, 1.0),
                           xycoords=axs[0, 0].transAxes,
                           fontsize=12,
                           ha='center',
                           va='bottom')
        axs[0, 0].annotate(text="4 Tables",
                           xy=(0.5, 1.0),
                           xycoords=axs[0, 0].transAxes,
                           fontsize=12,
                           ha='center',  # This centers the text horizontally
                           va='bottom')
        axs[0, 0].annotate(text="5 Tables",
                           xy=(0.85, 1.0),
                           xycoords=axs[0, 0].transAxes,
                           fontsize=12,
                           ha='center',  # This centers the text horizontally
                           va='bottom')

        axs[1, 0].get_legend().remove()
        axs[2, 0].get_legend().remove()

        plt.subplots_adjust(wspace=0.01, hspace=0.3)
        if database.display_name is not None:
            plt.suptitle(database.display_name)
        else:
            plt.suptitle(database.db_name)
        fig.align_labels()
        plt.tight_layout()
        plt.show()


def summary_plot_overall(results: List[dict], databases: List[Database], metrics: List[Metric]):
    results = pd.DataFrame.from_dict(results)
    model_colors = {model_config.display_name: model_config.color for model_config in MODEL_CONFIGS if
                    hasattr(model_config, 'name') and model_config.name is not None}

    fig, axs = plt.subplots(nrows=len(metrics),
                            ncols=len(databases),
                            figsize=(12, 6),
                            dpi=100,
                            sharex="col")

    for i, metric in enumerate(metrics):
        for j, database in enumerate(databases):
            # Plot the average scores for each model for each database
            ax = axs[i, j]

            avg_scores = results[results["database"] == database.db_name].groupby('model_name').apply(
                lambda x: pd.Series(
                    {metric.metric_name: getattr(x[metric.metric_name], metric.aggregation)() for metric in metrics}))
            avg_scores = avg_scores.reindex(model_colors.keys())
            avg_scores[metric.metric_name].plot(kind='bar',
                                                ax=ax,
                                                color=model_colors.values(),
                                                width=0.8,
                                                edgecolor=None)

            if metric.logscale:
                ax.set_yscale("log")
            ax.set_ylim(metric.y_min, metric.y_max)
            ax.set_axisbelow(True)
            #ax.bar_label(ax.containers[0], label_type='center', fmt='{:,.2f}', rotation=90, padding=0)
            ax.set_xlabel("")
            ax.grid(color="black", linewidth=0.2, which='both')
            if i == 0:
                ax.set_title(database.display_name)
            if j == 0:
                if metric.aggregation == 'mean':
                    ax.set_ylabel(f'Avg.\n{metric.display_name}')
                else:
                    ax.set_ylabel(f'Total\n{metric.display_name}')
            else:
                ax.set_ylabel("")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=10)

    plt.suptitle('Join Order - Summary over all databases')
    fig.align_labels()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    evaluator = Evaluator()
    metrics = [QError(), SpearmanCorrelation(), SelectedRuntime()]
    workloads = [EvalWorkloads.FullJoinOrder.imdb]

    databases = [
        #Database("imdb", display_name="IMDB"),
         #Database("tpc_h", display_name="TPC_H"),
         Database("imdb", display_name="IMDB")
    ]

    seeds = [0, 1, 2]

    for workload in workloads:
        evaluator.eval(workloads=workload,
                       metrics=metrics,
                       model_configs=MODEL_CONFIGS,
                       seeds=seeds,
                       plot_single_workloads=True,
                       plot_limit=10)
    summary_plot(evaluator.metric_collection, databases, metrics)
    summary_plot_overall(evaluator.metric_collection, databases, metrics)
