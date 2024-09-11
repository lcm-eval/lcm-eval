# Documentation for Workload-Driven Baseline

At first, we have to parse all plans

```
python3 run_benchmark.py \
  --parse_run \
  --database postgres \
  --source ../zero-shot-data/runs/raw/airline/test_workload.json \
  --target ../zero-shot-data/e2e_baseline/parsed_plans/airline/test_workload.json \
  --parse_baseline
  
python3 run_benchmark.py
  --parse_run
  --database postgres
  --source ../zero-shot-data/runs/raw/imdb/workload_100k_s1_c8220.json
  --target ../zero-shot-data/e2e_baseline/parsed_plans/imdb/workload_100k_s1_c8220.json
  --parse_baseline

# for all imdb experiments
DATASETS=("imdb_workload_400k_s2_c8220.json" "imdb_workload_400k_s3_c8220.json" "job-light_c8220.json" "scale_c8220.json" "synthetic_c8220.json")
for DS in ${DATASETS[*]}; do
    echo $DS;
    python3 run_benchmark.py --parse_run --database postgres --source ../zero-shot-data/runs/raw/imdb/$DS --target ../zero-shot-data/e2e_baseline/parsed_plans/imdb/$DS --parse_baseline
done;

# JOB full
python3 run_benchmark.py --parse_baseline --parse_run --include_zero_card --database postgres --source ../zero-shot-data/runs/raw/imdb_full/job_full_c8220.json --target ../zero-shot-data/e2e_baseline/parsed_plans/imdb_full/job_full_c8220.json
python3 run_benchmark.py --max_query_ms 600000 --parse_baseline --parse_run --include_zero_card --database postgres --source ../zero-shot-data/runs/raw/imdb_full/job_full_c8220.json --target ../zero-shot-data/e2e_baseline/parsed_plans/imdb_full/long_running_job_full_c8220.json
python3 run_benchmark.py --parse_baseline --parse_run --database postgres --source ../zero-shot-data/runs/raw/imdb_full/job_full_c8220.json --target ../zero-shot-data/e2e_baseline/parsed_plans/imdb_full/nonzero_job_full_c8220.json
python3 run_benchmark.py --max_query_ms 600000 --parse_baseline --parse_run --database postgres --source ../zero-shot-data/runs/raw/imdb_full/job_full_c8220.json --target ../zero-shot-data/e2e_baseline/parsed_plans/imdb_full/long_running_nonzero_job_full_c8220.json
DATASETS=("complex_workload_400k_s4_c8220.json" "complex_workload_400k_s5_c8220.json" "complex_workload_400k_s6_c8220.json")
for DS in ${DATASETS[*]}; do
    echo $DS;
    python3 run_benchmark.py --parse_run --database postgres --source ../zero-shot-data/runs/raw/imdb_full/$DS --target ../zero-shot-data/e2e_baseline/parsed_plans/imdb_full/$DS --parse_baseline
done;

# for updates
DATASETS=("synthetic_repl_3_c8220.json" "scale_repl_3_c8220.json" "job-light_repl_3_c8220.json" "synthetic_repl_2_c8220.json" "scale_repl_2_c8220.json" "job-light_repl_2_c8220.json" "synthetic_repl_1_c8220.json" "scale_repl_1_c8220.json" "job-light_repl_1_c8220.json")
for DS in ${DATASETS[*]}; do
    echo $DS;
    python3 run_benchmark.py --parse_run --database postgres --source ../zero-shot-data/runs/raw/imdb/$DS --target ../zero-shot-data/e2e_baseline/parsed_plans/imdb/$DS --parse_baseline
done;
```

Add sample bitmaps

```
python3 baseline.py
  --augment_sample_vectors
  --dataset airline
  --data_dir ../zero-shot-data/datasets/airline
  --source ../zero-shot-data/e2e_baseline/parsed_plans/airline/test_workload.json
  --target ../zero-shot-data/e2e_baseline/augmented_plans/airline/test_workload.json
  
python3 baseline.py
  --augment_sample_vectors
  --dataset airline
  --data_dir ../zero-shot-data/datasets/airline
  --source ../zero-shot-data/e2e_baseline/parsed_plans/airline/complex_workload_200k_s1_c8220.json
  --target ../zero-shot-data/e2e_baseline/augmented_plans/airline/complex_workload_200k_s1_c8220.json
  
python3 baseline.py
  --augment_sample_vectors
  --dataset imdb
  --data_dir ../zero-shot-data/datasets/imdb
  --source ../zero-shot-data/e2e_baseline/parsed_plans/imdb/job-light_c8220.json
  --target ../zero-shot-data/e2e_baseline/augmented_plans/imdb/job-light_c8220.json
  
python3 baseline.py
  --augment_sample_vectors
  --dataset imdb
  --data_dir ../zero-shot-data/datasets/imdb
  --source ../zero-shot-data/e2e_baseline/parsed_plans/imdb/workload_100k_s1_c8220.json
  --target ../zero-shot-data/e2e_baseline/augmented_plans/imdb/workload_100k_s1_c8220.json
  
# for all imdb experiments
DATASETS=("imdb_workload_400k_s2_c8220.json" "imdb_workload_400k_s3_c8220.json" "job-light_c8220.json" "scale_c8220.json" "synthetic_c8220.json")
for DS in ${DATASETS[*]}; do
    echo $DS;
    python3 baseline.py --augment_sample_vectors --dataset imdb --data_dir ../zero-shot-data/datasets/imdb --source ../zero-shot-data/e2e_baseline/parsed_plans/imdb/$DS --target ../zero-shot-data/e2e_baseline/augmented_plans/imdb/$DS
done;

# for JOB full
DATASETS=("job_full_c8220.json" "long_running_job_full_c8220.json" "nonzero_job_full_c8220.json" "long_running_nonzero_job_full_c8220.json" "complex_workload_400k_s4_c8220.json" "complex_workload_400k_s5_c8220.json" "complex_workload_400k_s6_c8220.json")
DATASETS=("complex_workload_400k_s4_c8220.json" "complex_workload_400k_s5_c8220.json" "complex_workload_400k_s6_c8220.json")
for DS in ${DATASETS[*]}; do
    echo $DS;
    python3 baseline.py --augment_sample_vectors --dataset imdb_full --data_dir ../zero-shot-data/datasets/imdb --source ../zero-shot-data/e2e_baseline/parsed_plans/imdb_full/$DS --target ../zero-shot-data/e2e_baseline/augmented_plans/imdb_full/$DS &
done; 

# for updates
DATASETS=("synthetic_repl_3_c8220.json" "scale_repl_3_c8220.json" "job-light_repl_3_c8220.json" "synthetic_repl_2_c8220.json" "scale_repl_2_c8220.json" "job-light_repl_2_c8220.json" "synthetic_repl_1_c8220.json" "scale_repl_1_c8220.json" "job-light_repl_1_c8220.json")
for DS in ${DATASETS[*]}; do
    echo $DS;
    python3 baseline.py --augment_sample_vectors --dataset imdb --data_dir ../zero-shot-data/datasets/imdb --source ../zero-shot-data/e2e_baseline/parsed_plans/imdb/$DS --target ../zero-shot-data/e2e_baseline/augmented_plans/imdb/$DS
done; 

# sliced join experiments [3] ~max 10k train queries
python3 run_benchmark.py --slice_no_tables --min_no_tables 4 --source ../zero-shot-data/e2e_baseline/augmented_plans/imdb/imdb_workload_400k_s2_c8220.json --target ../zero-shot-data/e2e_baseline/augmented_plans/imdb/imdb_workload_400k_s2_c8220_4plus.json
python3 run_benchmark.py --slice_no_tables --min_no_tables 4 --source ../zero-shot-data/e2e_baseline/augmented_plans/imdb/imdb_workload_400k_s3_c8220.json --target ../zero-shot-data/e2e_baseline/augmented_plans/imdb/imdb_workload_400k_s3_c8220_4plus.json

WORKLOADS=("synthetic" "scale" "job-light")
for WL in ${WORKLOADS[*]}; do
    echo $WL;
    python3 run_benchmark.py --slice_no_tables --min_no_tables 4 --source ../zero-shot-data/e2e_baseline/augmented_plans/imdb/${WL}_c8220.json --target ../zero-shot-data/e2e_baseline/augmented_plans/imdb/${WL}_c8220_4plus.json
done;

```

Train word embeddings for the predicate encoding. For this, we first construct the sentences.

```
python3 baseline.py
  --construct_sentences
  --dataset airline
  --data_dir ../zero-shot-data/datasets/airline
  --target ../zero-shot-data/e2e_baseline/sentences/airline/sentences.json
  --workload_runs ../zero-shot-data/e2e_baseline/parsed_plans/airline/complex_workload_200k_s1_c8220.json
  
python3 baseline.py
  --compute_word_embeddings
  --source ../zero-shot-data/e2e_baseline/sentences/airline/sentences.json
  --target ../zero-shot-data/e2e_baseline/sentences/airline/word2vec.m

# for all imdb experiments
python3 baseline.py
  --construct_sentences
  --dataset imdb
  --data_dir ../zero-shot-data/datasets/imdb
  --workload_runs 
    ../zero-shot-data/e2e_baseline/parsed_plans/imdb/imdb_workload_400k_s2_c8220.json
    ../zero-shot-data/e2e_baseline/parsed_plans/imdb/imdb_workload_400k_s3_c8220.json
  --target ../zero-shot-data/e2e_baseline/sentences/imdb/sentences.json

python3 baseline.py
  --compute_word_embeddings
  --source ../zero-shot-data/e2e_baseline/sentences/imdb/sentences.json
  --target ../zero-shot-data/e2e_baseline/sentences/imdb/word2vec.m

# JOB
python3 baseline.py
  --construct_sentences
  --dataset imdb_full
  --data_dir ../zero-shot-data/datasets/imdb
  --workload_runs 
    ../zero-shot-data/e2e_baseline/augmented_plans/imdb_full/complex_workload_400k_s4_c8220.json
    ../zero-shot-data/e2e_baseline/augmented_plans/imdb_full/complex_workload_400k_s5_c8220.json
    ../zero-shot-data/e2e_baseline/augmented_plans/imdb_full/complex_workload_400k_s6_c8220.json
  --target ../zero-shot-data/e2e_baseline/sentences/imdb_full/sentences.json

python3 baseline.py
  --compute_word_embeddings
  --source ../zero-shot-data/e2e_baseline/sentences/imdb_full/sentences.json
  --target ../zero-shot-data/e2e_baseline/sentences/imdb_full/word2vec.m

python3 baseline.py 
  --compute_word_embeddings 
  --source ../zero-shot-data/e2e_baseline/sentences/imdb/sentences.json 
  --target ../zero-shot-data/e2e_baseline/sentences/imdb/word2vec.m
```

Train the model

```
# local test airbnb simple workload
python3 train.py
  --gather_feature_statistics
  --workload_runs 
        ../zero-shot-data/e2e_baseline/augmented_plans/airline/test_workload.json
  --target ../zero-shot-data/e2e_baseline/augmented_plans/airline/statistics.json

python3 baseline.py 
  --train_model
  --workload_runs 
        ../zero-shot-data/e2e_baseline/augmented_plans/airline/test_workload.json
  --test_workload_runs
        ../zero-shot-data/e2e_baseline/augmented_plans/airline/test_workload.json
  --statistics_file ../zero-shot-data/e2e_baseline/augmented_plans/airline/statistics.json
  --column_statistics ./cross_db_benchmark/datasets/airline/column_statistics.json
  --word_embeddings ../zero-shot-data/e2e_baseline/sentences/airline/word2vec.m
  --target todo
  --filename_model e2eb
  --model_name MSCN

# local test airbnb complex workload
python3 train.py
  --gather_feature_statistics
  --workload_runs 
        ../zero-shot-data/e2e_baseline/augmented_plans/airline/complex_workload_200k_s1_c8220.json
  --target ../zero-shot-data/e2e_baseline/augmented_plans/airline/statistics_complex_workload_200k_s1_c8220.json

python3 baseline.py 
  --train_model
  --workload_runs 
        ../zero-shot-data/e2e_baseline/augmented_plans/airline/complex_workload_200k_s1_c8220.json
  --test_workload_runs
        ../zero-shot-data/e2e_baseline/augmented_plans/airline/complex_workload_200k_s1_c8220.json
  --statistics_file ../zero-shot-data/e2e_baseline/augmented_plans/airline/statistics_complex_workload_200k_s1_c8220.json
  --column_statistics ./cross_db_benchmark/datasets/airline/column_statistics.json
  --word_embeddings ../zero-shot-data/e2e_baseline/sentences/airline/word2vec.m
  --target todo
  --filename_model e2eb

# local test imdb
python3 train.py
  --gather_feature_statistics
  --workload_runs 
        ../zero-shot-data/e2e_baseline/augmented_plans/imdb/workload_100k_s1_c8220.json
  --target ../zero-shot-data/e2e_baseline/augmented_plans/imdb/statistics.json

python3 baseline.py 
  --train_model
  --workload_runs 
        ../zero-shot-data/e2e_baseline/augmented_plans/imdb/workload_100k_s1_c8220.json
  --test_workload_runs
        ../zero-shot-data/e2e_baseline/augmented_plans/imdb/workload_100k_s1_c8220.json
  --statistics_file ../zero-shot-data/e2e_baseline/augmented_plans/imdb/statistics.json
  --column_statistics ./cross_db_benchmark/datasets/imdb/column_statistics.json
  --word_embeddings ../zero-shot-data/e2e_baseline/sentences/imdb/word2vec.m
  --target todo
  --filename_model e2ebe
  --model_name MSCN
  --device cuda:1

python3 baseline.py 
    --train_model 
    --workload_runs 
        ../zero-shot-data/e2e_baseline/augmented_plans/imdb/workload_100k_s1_c8220.json 
    --test_workload_runs 
        ../zero-shot-data/e2e_baseline/augmented_plans/imdb/workload_100k_s1_c8220.json 
    --statistics_file ../zero-shot-data/e2e_baseline/augmented_plans/imdb/statistics.json 
    --column_statistics ./cross_db_benchmark/datasets/imdb/column_statistics.json 
    --word_embeddings ../zero-shot-data/e2e_baseline/sentences/imdb/word2vec.m 
    --target todo 
    --filename_model e2ebe 
    --model_name MSCN 
    --device cuda:1

# local test imdb job light

python3 train.py
  --gather_feature_statistics
  --workload_runs 
        ../zero-shot-data/e2e_baseline/augmented_plans/imdb/job-light_c8220.json
  --target ../zero-shot-data/e2e_baseline/augmented_plans/imdb/statistics_job-light_c8220.json

python3 baseline.py 
  --train_model
  --workload_runs 
        ../zero-shot-data/e2e_baseline/augmented_plans/imdb/job-light_c8220.json
  --test_workload_runs
        ../zero-shot-data/e2e_baseline/augmented_plans/imdb/job-light_c8220.json
  --statistics_file ../zero-shot-data/e2e_baseline/augmented_plans/imdb/statistics_job-light_c8220.json
  --column_statistics ./cross_db_benchmark/datasets/imdb/column_statistics.json
  --word_embeddings ../zero-shot-data/e2e_baseline/sentences/imdb/word2vec.m
  --target todo
  --filename_model e2ebe
  --model_name MSCN
  --device cuda:1

# for all imdb experiments
"job-light_c8220.json" "scale_c8220.json" "synthetic_c8220.json")
python3 train.py
  --gather_feature_statistics
  --workload_runs 
        ../zero-shot-data/e2e_baseline/augmented_plans/imdb/imdb_workload_400k_s2_c8220.json
        ../zero-shot-data/e2e_baseline/augmented_plans/imdb/imdb_workload_400k_s3_c8220.json
        ../zero-shot-data/e2e_baseline/augmented_plans/imdb/job-light_c8220.json
        ../zero-shot-data/e2e_baseline/augmented_plans/imdb/scale_c8220.json
        ../zero-shot-data/e2e_baseline/augmented_plans/imdb/synthetic_c8220.json
  --target ../zero-shot-data/e2e_baseline/augmented_plans/imdb/statistics_combined.json

python3 train.py
  --gather_feature_statistics
  --workload_runs 
        ../zero-shot-data/e2e_baseline/augmented_plans/imdb_full/complex_workload_400k_s4_c8220.json
        ../zero-shot-data/e2e_baseline/augmented_plans/imdb_full/complex_workload_400k_s5_c8220.json
        ../zero-shot-data/e2e_baseline/augmented_plans/imdb_full/complex_workload_400k_s6_c8220.json
        ../zero-shot-data/e2e_baseline/augmented_plans/imdb_full/job_full_c8220.json
        ../zero-shot-data/e2e_baseline/augmented_plans/imdb_full/long_running_job_full_c8220.json
        ../zero-shot-data/e2e_baseline/augmented_plans/imdb_full/nonzero_job_full_c8220.json
        ../zero-shot-data/e2e_baseline/augmented_plans/imdb_full/long_running_nonzero_job_full_c8220.json
  --target ../zero-shot-data/e2e_baseline/augmented_plans/imdb/job_full_statistics_combined.json

# this is just for testing (will be generated by experiment scripts)
python3 baseline.py 
  --train_model
  --workload_runs 
        ../zero-shot-data/e2e_baseline/augmented_plans/imdb/workload_100k_s1_c8220.json
  --test_workload_runs
        ../zero-shot-data/e2e_baseline/augmented_plans/imdb/job-light_c8220.json
  --statistics_file ../zero-shot-data/e2e_baseline/augmented_plans/imdb/statistics_combined.json
  --column_statistics ./cross_db_benchmark/datasets/imdb/column_statistics.json
  --word_embeddings ../zero-shot-data/e2e_baseline/sentences/imdb/word2vec.m
  --target todo
  --filename_model e2ebe
  --model_name Optimizer
  --cap_training_samples 1000

run_exp "numactl --cpunodebind=1 --membind=1 
python3 baseline.py 
    --train_model 
    --workload_runs 
    ../zero-shot-data/e2e_baseline/augmented_plans/imdb/imdb_workload_400k_s2_c8220.json 
    ../zero-shot-data/e2e_baseline/augmented_plans/imdb/imdb_workload_400k_s3_c8220.json 
    --test_workload_runs 
    ../zero-shot-data/e2e_baseline/augmented_plans/imdb/job-light_c8220.json 
    ../zero-shot-data/e2e_baseline/augmented_plans/imdb/scale_c8220.json 
    ../zero-shot-data/e2e_baseline/augmented_plans/imdb/synthetic_c8220.json 
    --statistics_file ../zero-shot-data/e2e_baseline/augmented_plans/imdb/statistics_combined.json 
    --column_statistics ./cross_db_benchmark/datasets/imdb/column_statistics.json 
    --word_embeddings ../zero-shot-data/e2e_baseline/sentences/imdb/word2vec.m 
    --target ../zero-shot-data/evaluation/e2e_baseline/imdb/  
    --max_epoch_tuples 100000 
    --filename_model TPool_500_0 
    --cap_training_samples 500 
    --model_name TPool 
    --num_workers 16 --seed 0 "

run_exp -m "Zero-shot GPU 1, Socket 1" "numactl --cpunodebind=1 --membind=1 "

run_exp -m "Zero-shot GPU 1, Socket 1" "numactl --cpunodebind=1 --membind=1 
python3 baseline.py 
--train_model 
--workload_runs ../zero-shot-data/e2e_baseline/augmented_plans/imdb/workload_100k_s1_c8220.json 
--test_workload_runs ../zero-shot-data/e2e_baseline/augmented_plans/imdb/workload_100k_s1_c8220.json 
--statistics_file ../zero-shot-data/e2e_baseline/augmented_plans/imdb/statistics.json
 --column_statistics ./cross_db_benchmark/datasets/imdb/column_statistics.json 
 --word_embeddings ../zero-shot-data/e2e_baseline/sentences/imdb/word2vec.m 
 --target todo --filename_model e2ebe 
 --model_name MSCN --device cuda:1"

run_exp -m "Zero-shot GPU 1, Socket 1" "numactl --cpunodebind=1 --membind=1 
python3 baseline.py --train_model 
--workload_runs ../zero-shot-data/e2e_baseline/augmented_plans/imdb/workload_100k_s1_c8220.json 
--test_workload_runs ../zero-shot-data/e2e_baseline/augmented_plans/imdb/workload_100k_s1_c8220.json 
--statistics_file ../zero-shot-data/e2e_baseline/augmented_plans/imdb/statistics.json 
--column_statistics ./cross_db_benchmark/datasets/imdb/column_statistics.json 
--word_embeddings ../zero-shot-data/e2e_baseline/sentences/imdb/word2vec.m 
--target todo --filename_model e2ebe 
--model_name MSCN --device cuda:1"

```

# Todo

# Done

- for multiple strings add embeddings - then max/min to top node
    - model
        - one predicate graph
            - predicate encoding using DFS with nulls
                - per-predicate encoding of
                    - string words/literal
                    - operator
                    - column
            - bottom-up pass predicates
        - one graph plans
            - plan node encoding
                - sample bitmap
                - operation
                - predicate
                - output column (binary vector)
                - table (binary vector)
            - lstm-pass
                - implement using: avg of Gt, Rt
                - then lstm op per nodes in current depth
- Operation, Metadata, Predicate and Sample Bitmap
- integrate naively in learning - predicate graph - node features:
- boolean op - predicate column - predicate operator - predicate operand
- parse plans
    - column number
    - bitmaps
    - operation
    - table number
    - index number
    - output columns
    - literals
- first end-to-end training
- add bitmap samples for predicates
- word embeddings for strings
    - construct sentences
    - for encoding take mean of word vectors + this hash version original in code
- flexible derivation of dims
- check that we really use the same hyperparameters as Sun et al.
- bugfix predicate parsing
- check that queries for imdb make sense (for training)
- check GPU execution
- TPool/ TLSTM described in the paper
- variants with pooling etc. described in the paper
- MSCN baseline
    - parse baseline: extract which joins are executed
    - new plan batching and model
    - set convolution
        - table set (table one hot, bitmap)
        - join set (one hot: which tables are joined)
        - predicate set (column one hot, operator one hot, predicate one hot)
        - output column set (which aggregation and which column)
- Postgres Baseline
- manage experiment on cloudlab
- for train != test workload make sure that column indexes, table indexes and one hot encodings match!
    - column stats
    - samples for bitmaps should be drawn with seed
- checkout: scale optimizer (1.54 wo scale 1.67 with scale)
- recompute all files required for experiment
- setup exp commands
- experiment setup
    - different models: MSCN, TPool, Optimizer
    - seed this (repeat for 8 different seeds)
    - test datasets: job light scale etc.
    - for different caps: 500, 1k, 5k, 10k, 50k
- setup venv on all nodes
- sync files
- evaluation notebook
- Experiment setup and scripts
    - TPool on Job-light: (vs us 1.39 with pg and 1.17 exact cards after zero queries!)
        - for 0.5k queries q error of 1.52 on job light
        - for 1k queries q error of 1.49 on job light
        - for 5k queries q error of 1.38 on job light
        - for 10k queries q error of 1.29 on job light
- some kind of workload for updates?