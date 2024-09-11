import argparse

from models.tabular.train_tabular_baseline import train_tabular_model
from models.workload_driven.preprocessing.sample_vectors import augment_sample_vectors
from models.workload_driven.preprocessing.sentence_creation import create_sentences
from models.workload_driven.preprocessing.word_embeddings import compute_word_embeddings
from models.workload_driven.training.train_baseline import train_baseline_default

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--workload_runs', default=None, nargs='+')
    parser.add_argument('--test_workload_runs', default=None, nargs='+')
    parser.add_argument('--statistics_file', default=None)
    parser.add_argument('--column_statistics', default=None)
    parser.add_argument('--word_embeddings', default=None)
    parser.add_argument('--dataset', default=None)
    parser.add_argument('--data_dir', default=None)
    parser.add_argument('--source', default=None)
    parser.add_argument('--target', default=None)
    parser.add_argument('--filename_model', default=None)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--max_epoch_tuples', type=int, default=100000)
    parser.add_argument('--cap_training_samples', type=int, default=None)
    parser.add_argument('--model_name', default='TPool')

    # Benchmark Steps
    parser.add_argument('--train_model', action='store_true')
    parser.add_argument('--augment_sample_vectors', action='store_true')
    parser.add_argument('--construct_sentences', action='store_true')
    parser.add_argument('--compute_word_embeddings', action='store_true')
    parser.add_argument('--train_tabular_model', action='store_true')
    parser.add_argument('--hyperparameter_path', default=None)
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()

    if args.construct_sentences:
        create_sentences(args.dataset, args.workload_runs, args.data_dir, args.target)

    if args.compute_word_embeddings:
        compute_word_embeddings(args.source, args.target)

    if args.augment_sample_vectors:
        augment_sample_vectors(args.dataset, args.data_dir, args.source, args.target)

    if args.train_model:
        if args.hyperparameter_path is None:
            # for testing
            train_baseline_default(args.workload_runs, args.test_workload_runs, args.statistics_file,
                                   args.column_statistics, args.word_embeddings, args.target,
                                   args.filename_model, cap_training_samples=args.cap_training_samples,
                                   model_name=args.model_name, device=args.device,
                                   num_workers=args.num_workers, max_epoch_tuples=args.max_epoch_tuples,
                                   seed=args.seed)

    if args.train_tabular_model:
        train_tabular_model(args.workload_runs,
                            args.test_workload_runs,
                            args.statistics_file,
                            args.target,
                            args.filename_model,
                            seed=args.seed)
