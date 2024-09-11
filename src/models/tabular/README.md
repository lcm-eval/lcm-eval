# Documentation for Tabular Baseline

At first, we have to parse all plans

```
python3 train.py
  --gather_feature_statistics
  --workload_runs 
        ../zero-shot-data/e2e_baseline/augmented_plans/imdb/imdb_workload_400k_s2_c8220.json
        ../zero-shot-data/e2e_baseline/augmented_plans/imdb/imdb_workload_400k_s3_c8220.json
        ../zero-shot-data/e2e_baseline/augmented_plans/imdb/job-light_c8220.json
        ../zero-shot-data/e2e_baseline/augmented_plans/imdb/synthetic_c8220.json
        ../zero-shot-data/e2e_baseline/augmented_plans/imdb/scale_c8220.json
  --target ../zero-shot-data/e2e_baseline/augmented_plans/imdb/statistics_combined.json

python3 baseline.py 
  --train_tabular_model
  --workload_runs 
        ../zero-shot-data/e2e_baseline/augmented_plans/imdb/imdb_workload_400k_s2_c8220.json
        ../zero-shot-data/e2e_baseline/augmented_plans/imdb/imdb_workload_400k_s3_c8220.json
  --test_workload_runs
        ../zero-shot-data/e2e_baseline/augmented_plans/imdb/job-light_c8220.json
        ../zero-shot-data/e2e_baseline/augmented_plans/imdb/synthetic_c8220.json
        ../zero-shot-data/e2e_baseline/augmented_plans/imdb/scale_c8220.json
  --statistics_file ../zero-shot-data/e2e_baseline/augmented_plans/imdb/statistics_combined.json
  --target todo
  --filename_model lightgbm_test
  --model_name LightGBM
  --cap_training_samples 50000

python3 baseline.py 
  --train_tabular_model
  --workload_runs 
        ../zero-shot-data/e2e_baseline/augmented_plans/imdb/imdb_workload_400k_s2_c8220.json
        ../zero-shot-data/e2e_baseline/augmented_plans/imdb/imdb_workload_400k_s3_c8220.json
  --test_workload_runs
        ../zero-shot-data/e2e_baseline/augmented_plans/imdb/job-light_c8220.json
        ../zero-shot-data/e2e_baseline/augmented_plans/imdb/synthetic_c8220.json
        ../zero-shot-data/e2e_baseline/augmented_plans/imdb/scale_c8220.json
  --statistics_file ../zero-shot-data/e2e_baseline/augmented_plans/imdb/statistics_combined.json
  --target todo
  --filename_model lightgbm_test
  --model_name AnalyticalEstCard
  --cap_training_samples 50000

```


# Todo
          
# Done

- implement baseline algorithm and save csv
- experiment infrastructure
- results
    - 100 queries: 
                todo/test_lightgbm_test_job-light_c8220.csv
                    val_median_q_error_50: 1.5804 [best: inf]
                    val_median_q_error_95: 3.0111 [best: inf]
                    val_median_q_error_100: 3.6944 [best: inf]
                todo/test_lightgbm_test_synthetic_c8220.csv
                    val_median_q_error_50: 1.8009 [best: inf]
                    val_median_q_error_95: 4.0485 [best: inf]
                    val_median_q_error_100: 30.6286 [best: inf]
                todo/test_lightgbm_test_scale_c8220.csv
                    val_median_q_error_50: 1.9751 [best: inf]
                    val_median_q_error_95: 4.4375 [best: inf]
                    val_median_q_error_100: 5.4513 [best: inf]
      - 1k queries
                todo/test_lightgbm_test_job-light_c8220.csv
                    val_median_q_error_50: 1.4141 [best: inf]
                    val_median_q_error_95: 2.1587 [best: inf]
                    val_median_q_error_100: 3.1921 [best: inf]
                todo/test_lightgbm_test_synthetic_c8220.csv
                    val_median_q_error_50: 1.6267 [best: inf]
                    val_median_q_error_95: 3.6661 [best: inf]
                    val_median_q_error_100: 4.7421 [best: inf]
                todo/test_lightgbm_test_scale_c8220.csv
                    val_median_q_error_50: 1.5780 [best: inf]
                    val_median_q_error_95: 3.4670 [best: inf]
                    val_median_q_error_100: 11.1935 [best: inf]
        - 3k queries
                todo/test_lightgbm_test_job-light_c8220.csv
                    val_median_q_error_50: 1.3959 [best: inf]
                    val_median_q_error_95: 2.1073 [best: inf]
                    val_median_q_error_100: 2.7953 [best: inf]
                todo/test_lightgbm_test_synthetic_c8220.csv
                    val_median_q_error_50: 1.6324 [best: inf]
                    val_median_q_error_95: 3.4921 [best: inf]
                    val_median_q_error_100: 12.8728 [best: inf]
                todo/test_lightgbm_test_scale_c8220.csv
                    val_median_q_error_50: 1.6300 [best: inf]
                    val_median_q_error_95: 3.0576 [best: inf]
                    val_median_q_error_100: 7.4207 [best: inf]
        - 10k queries
                todo/test_lightgbm_test_job-light_c8220.csv
                    val_median_q_error_50: 1.2964 [best: inf]
                    val_median_q_error_95: 1.8875 [best: inf]
                    val_median_q_error_100: 2.7336 [best: inf]
                todo/test_lightgbm_test_synthetic_c8220.csv
                    val_median_q_error_50: 1.5933 [best: inf]
                    val_median_q_error_95: 4.0905 [best: inf]
                    val_median_q_error_100: 15.3074 [best: inf]
                todo/test_lightgbm_test_scale_c8220.csv
                    val_median_q_error_50: 1.5915 [best: inf]
                    val_median_q_error_95: 2.6629 [best: inf]
                    val_median_q_error_100: 8.1252 [best: inf]
        - 50k queries
                todo/test_lightgbm_test_job-light_c8220.csv
                    val_median_q_error_50: 1.2977 [best: inf]
                    val_median_q_error_95: 1.9130 [best: inf]
                    val_median_q_error_100: 3.6845 [best: inf]
                todo/test_lightgbm_test_synthetic_c8220.csv
                    val_median_q_error_50: 1.5617 [best: inf]
                    val_median_q_error_95: 4.7928 [best: inf]
                    val_median_q_error_100: 8.5124 [best: inf]
                todo/test_lightgbm_test_scale_c8220.csv
                    val_median_q_error_50: 1.5864 [best: inf]
                    val_median_q_error_95: 2.7058 [best: inf]
                    val_median_q_error_100: 6.3112 [best: inf]