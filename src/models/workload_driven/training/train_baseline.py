import os
import time
from copy import copy

import numpy as np
import optuna
import torch
import torch.optim as opt

from models.workload_driven.dataset.dataset_creation import create_baseline_dataloader
from models.workload_driven.dataset.plan_tree_batching import plan_batch_to
from models.workload_driven.model.mscn_model import MSCNModel
from models.workload_driven.model.e2e_model import E2EModel
from training.training.checkpoint import save_csv, load_checkpoint, save_checkpoint
from training.training.metrics import RMSE, MAPE, QError
from training.training.train import validate_model, optuna_intermediate_value, train_epoch
from training.training.utils import find_early_stopping_metric, flatten_dict
from training.batch_to_funcs import batch_to


def train_baseline_default(workload_runs,
                           test_workload_runs,
                           statistics_file,
                           column_statistics,
                           word_embeddings,
                           target_dir,
                           filename_model,
                           cap_training_samples=None,
                           model_name=None,
                           device='cpu',
                           max_epoch_tuples=100000,
                           num_workers=1,
                           seed=0):
    """
    Sets default parameters and trains model
    """

    # hyperparameters from Sun et al. paper
    if model_name in {'TPool', 'TLstm', 'TestSum'}:
        train_kwargs = dict(optimizer_class_name='Adam',
                            optimizer_kwargs=dict(
                                lr=1e-3,
                            ),
                            model_name=model_name,
                            hidden_dim_plan=256,
                            hidden_dim_pred=256,
                            batch_size=64,
                            epochs=20,
                            early_stopping_patience=20,
                            max_epoch_tuples=max_epoch_tuples,
                            device=device,
                            num_workers=num_workers,
                            cap_training_samples=cap_training_samples,
                            seed=seed
                            )
    elif model_name in {'MSCN'}:
        train_kwargs = dict(optimizer_class_name='Adam',
                            optimizer_kwargs=dict(
                                lr=1e-3,
                            ),
                            model_name=model_name,
                            hidden_dim_plan=256,
                            batch_size=64,
                            epochs=100,
                            mscn_enc_layers=1,
                            mscn_est_layers=1,
                            early_stopping_patience=20,
                            max_epoch_tuples=max_epoch_tuples,
                            device=device,
                            num_workers=num_workers,
                            cap_training_samples=cap_training_samples,
                            seed=seed
                            )
    else:
        raise NotImplementedError

    param_dict = flatten_dict(train_kwargs)

    train_baseline_model(workload_runs, test_workload_runs, statistics_file, column_statistics, word_embeddings,
                         target_dir, filename_model, param_dict=param_dict, **train_kwargs)


def train_baseline_model(workload_runs,
                         test_workload_runs,
                         statistics_file,
                         column_statistics,
                         word_embeddings,
                         target_dir,
                         filename_model,
                         optimizer_class_name='Adam',
                         optimizer_kwargs=None,
                         model_name='TPool',
                         hidden_dim_plan=32,
                         hidden_dim_pred=32,
                         mscn_enc_layers=1,
                         mscn_est_layers=1,
                         loss_class_name='QLoss',
                         loss_class_kwargs=None,
                         batch_size=32,
                         epochs=0,
                         device='cpu',
                         max_epoch_tuples=100000,
                         param_dict=None,
                         num_workers=1,
                         cap_training_samples=None,
                         early_stopping_patience=20,
                         trial=None,
                         seed=0):
    if loss_class_kwargs is None:
        loss_class_kwargs = dict()

    # seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    target_test_csv_paths = []
    for p in test_workload_runs:
        test_workload = os.path.basename(p).replace('.json', '')
        target_test_csv_paths.append(os.path.join(target_dir, f'test_{filename_model}_{test_workload}.csv'))

    if all([os.path.exists(p) for p in target_test_csv_paths]):
        print(f"Model was already trained and tested ({target_test_csv_paths} exists)")
        return

    # create a dataset
    feature_statistics, train_loader, val_loader, test_loaders, input_dims = \
        create_baseline_dataloader(workload_runs, test_workload_runs, statistics_file, column_statistics,
                                   word_embeddings, model_name, cap_training_samples=cap_training_samples,
                                   val_ratio=0.15, batch_size=batch_size, shuffle=True,
                                   num_workers=num_workers, pin_memory=False, dim_bitmaps=1000)

    metrics = [RMSE(), MAPE(), QError(percentile=50, early_stopping_metric=True), QError(percentile=95),
               QError(percentile=100)]

    if model_name in {'TPool', 'TLstm', 'TestSum'}:
        model = E2EModel(hidden_dim_pred=hidden_dim_pred, input_dim_pred=input_dims.input_dim_pred,
                          hidden_dim_plan=hidden_dim_plan, input_dim_plan=input_dims.input_dim_plan,
                          device=device, model_name=model_name, loss_class_name=loss_class_name,
                          loss_class_kwargs=loss_class_kwargs)
        batch_to_func = plan_batch_to
    elif model_name in {'MSCN'}:
        model = MSCNModel(hidden_dim=hidden_dim_plan, input_dim_table=input_dims.input_dim_table,
                          input_dim_pred=input_dims.input_dim_pred, input_dim_join=input_dims.input_dim_join,
                          input_dim_agg=input_dims.input_dim_agg, loss_class_name=loss_class_name,
                          loss_class_kwargs=loss_class_kwargs, n_enc_layers=mscn_enc_layers,
                          n_add_est_layers=mscn_est_layers, device=device)
        batch_to_func = batch_to
    else:
        raise NotImplementedError

    # move to gpu
    model = model.to(model.device)
    print(model)
    optimizer = opt.__dict__[optimizer_class_name](model.parameters(), **optimizer_kwargs)

    csv_stats, epochs_wo_improvement, epoch, model, optimizer, metrics, finished = \
        load_checkpoint(model, target_dir, filename_model, optimizer=optimizer, metrics=metrics, filetype='.pt')

    # train an actual model (q-error? or which other loss?)
    while epoch < epochs:
        print(f"Epoch {epoch}")

        epoch_stats = copy(param_dict)
        epoch_stats.update(epoch=epoch)
        epoch_start_time = time.perf_counter()

        try:
            train_epoch(epoch_stats, train_loader, model, optimizer, max_epoch_tuples, custom_batch_to=batch_to_func)

            any_best_metric = validate_model(val_loader, model, epoch=epoch, epoch_stats=epoch_stats, metrics=metrics,
                                             max_epoch_tuples=max_epoch_tuples, custom_batch_to=batch_to_func,
                                             verbose=True)
            epoch_stats.update(epoch=epoch, epoch_time=time.perf_counter() - epoch_start_time)

            # report to optuna
            if trial is not None:
                intermediate_value = optuna_intermediate_value(metrics)
                epoch_stats['optuna_intermediate_value'] = intermediate_value

                print(f"Reporting epoch_no={epoch}, intermediate_value={intermediate_value} to optuna "
                      f"(Trial {trial.number})")
                trial.report(intermediate_value, epoch)

            # see if we can already stop the training
            stop_early = False
            if not any_best_metric:
                epochs_wo_improvement += 1
                if early_stopping_patience is not None and epochs_wo_improvement > early_stopping_patience:
                    stop_early = True
            else:
                epochs_wo_improvement = 0
            if trial is not None and trial.should_prune():
                stop_early = True
            # also set finished to true if this is the last epoch
            if epoch == epochs - 1:
                stop_early = True

            epoch_stats.update(stop_early=stop_early)
            print(f"epochs_wo_improvement: {epochs_wo_improvement}")

            # save stats to file
            csv_stats.append(epoch_stats)

            # save current state of training allowing us to resume if this is stopped
            save_checkpoint(epochs_wo_improvement, epoch, model, optimizer, target_dir,
                            filename_model, metrics=metrics, csv_stats=csv_stats, finished=stop_early)

            epoch += 1

            # Handle pruning based on the intermediate value.
            if trial is not None and trial.should_prune():
                raise optuna.TrialPruned()

            if stop_early:
                print(f"Early stopping kicked in due to no improvement in {early_stopping_patience} epochs")
                break
        except:
            print("Error during epoch. Trying again.")

    # if we are not doing hyperparameter search, evaluate test set
    if trial is None and test_loaders is not None:
        if not (target_dir is None or filename_model is None):
            assert len(target_test_csv_paths) == len(test_loaders)
            for test_path, test_loader in zip(target_test_csv_paths, test_loaders):
                print(f"Starting validation for {test_path}")
                test_stats = copy(param_dict)

                early_stop_m = find_early_stopping_metric(metrics)
                print("Reloading best model")
                model.load_state_dict(early_stop_m.best_model)
                validate_model(test_loader, model, epoch=epoch, epoch_stats=test_stats, metrics=metrics,
                               custom_batch_to=batch_to_func, log_all_queries=True)

                save_csv([test_stats], test_path)

        else:
            print("Skipping saving the test stats")

    if trial is not None:
        return optuna_intermediate_value(metrics)
