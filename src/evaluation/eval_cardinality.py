from typing import List

import matplotlib.pyplot as plt
import pandas as pd

from cross_db_benchmark.benchmark_tools.utils import load_json
from cross_db_benchmark.datasets.datasets import Database
from classes.classes import MODEL_CONFIGS, ScaledPostgresModelConfig, FlatModelConfig, E2EModelConfig, MSCNModelConfig, \
    QPPNetModelConfig, ZeroShotModelConfig, DACEModelConfig
from classes.paths import LocalPaths
from classes.workloads import EvalWorkloads
from evaluation.eval import Evaluator
from evaluation.evaluation_metrics import QError, RMSE, Metric, SpearmanCorrelation

HEADINGS = ["SELECT * FROM [TABLE] WHERE [COLUMN] = X", "SELECT * FROM [TABLE] WHERE [COLUMN] >= X", ]
QUERY_TYPES = ["cardinality_plan/seq_point_filter", "cardinality_plan/range_point_filter"]


def summary_plot(results: List[dict], databases: List[Database], metrics: List[Metric]):
    result_df = pd.DataFrame.from_dict(results)
    result_df["table"] = result_df["workload"].str.rsplit('.', n=1).str[0]
    result_df["column"] = result_df["workload"].str.rsplit('.', n=1).str[1]

    for database in databases:
        database_df = result_df[result_df["database"] == database.db_name]

        # Assign table lengths to table
        table_length_json_path = LocalPaths().code / "cross_db_benchmark" / "datasets" / database.db_name / "table_lengths.json"
        database_df['table_lengths'] = database_df['table'].map(load_json(table_length_json_path, namespace=False))

        # Assign unique values to table
        column_statistics_path = LocalPaths().code / "cross_db_benchmark" / "datasets" / database.db_name / "column_statistics.json"
        column_stats = load_json(column_statistics_path, namespace=False)

        def get_unique(row):
            table = row['table']
            column = row['column']
            return column_stats.get(table, {}).get(column, None)["num_unique"]

        database_df['num_unique'] = database_df.apply(get_unique, axis=1)

        # Do a plot for different query types
        for wl_type, heading in zip(QUERY_TYPES, HEADINGS):
            workload_df = database_df[database_df["workload_type"] == wl_type]
            fig, axs = plt.subplots(4, 1, figsize=(8, 6), sharex=True, dpi=100)

            # Do a row plot for each metric
            for metric, ax in zip(metrics, axs):
                metric_df = workload_df.groupby(['workload', "table_lengths", 'model_name'])[
                    [metric.metric_name]].mean().unstack()
                metric_df = metric_df.sort_values(by='table_lengths')
                metric_df.columns = metric_df.columns.map('{0[1]}'.format)
                metric_df.plot(kind='bar',
                               ax=ax,
                               color=[m.color for m in MODEL_CONFIGS],
                               width=0.8,
                               edgecolor='black')

                if metric.logscale:
                    ax.set_yscale("log")

                ax.set_xlabel("Table/Column")
                ax.set_ylabel(metric.display_name)

                # Reformat xticks
                xticks = [item.get_text() for item in ax.get_xticklabels()]
                formatted_labels = [label.replace('.', '\n') for label in xticks]
                ax.set_xticklabels(formatted_labels, rotation=45, fontsize=9, ha='right')

                ax.legend(ncol=6, loc='upper center', bbox_to_anchor=(0.5, 1.5))
                ax.set_axisbelow(True)
                ax.set_ylim(metric.y_min, metric.y_max)
                ax.grid()

            # Adding extra plot in the end
            metric_df = workload_df.groupby(['workload'])[["table_lengths", "num_unique"]].mean()
            metric_df = metric_df.sort_values(by="table_lengths")
            metric_df.plot(kind='bar',
                           ax=axs[3],
                           color=["black", "gray"],
                           label=["num_vals", "num_unique"],
                           width=0.8,
                           edgecolor='black')

            axs[3].legend(ncol=6, loc='upper center', bbox_to_anchor=(0.5, 1.5))
            axs[3].set_axisbelow(True)
            axs[3].set_yscale("log")
            axs[3].grid()

            # Reformat x-ticks of last row
            xticks = [item.get_text() for item in axs[3].get_xticklabels()]
            formatted_labels = [label.replace('.', '\n') for label in xticks]
            axs[3].set_xticklabels(formatted_labels, rotation=45, fontsize=9, ha='right')

            # Set legends
            axs[0].legend(loc='upper right', prop={'family': 'monospace', 'size': 6}, ncol=3)
            axs[3].legend(loc='upper right', prop={'family': 'monospace', 'size': 6})
            for ax in axs[1:-1]:
                ax.get_legend().remove()

            plt.suptitle(database.db_name + "\n" + heading)
            plt.tight_layout()
            plt.show()


if __name__ == '__main__':
    evaluator = Evaluator()
    metrics = [QError(), RMSE(), SpearmanCorrelation()]

    workloads = [EvalWorkloads.ScanCost.FourWayJoin.tpc, EvalWorkloads.ScanCost.FourWayJoin.imdb, EvalWorkloads.ScanCost.FourWayJoin.baseball]
    databases = [Database("tpc_h"), Database("imdb"), Database("baseball")]
    model_configs = [
        ScaledPostgresModelConfig(),
        FlatModelConfig(),
        MSCNModelConfig(),
        E2EModelConfig(),
        QPPNetModelConfig(),
        ZeroShotModelConfig(),
        DACEModelConfig()
    ]
    for workload in workloads:
        evaluator.eval(workload,
                       metrics,
                       plot_single_workloads=True,
                       plot_limit=10,
                       model_configs=model_configs,
                       seeds=[0, 1, 2])

    summary_plot(evaluator.metric_collection, databases, metrics)
