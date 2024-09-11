# How good a (learned) cost models, really? 
This repository contains the evaluation **code** of the submission "How good a (learned) cost models, really?"
**The training data, trained models and model predictions are available at [OSF](https://osf.io/mahku/?view_only=a5255c9b7aad4c198aadcd4aa22c87fc).**

## Hardware Instances
This code distinguishes between different type of machines.
- **Cloudlab-Instances**: The machines of type [c8220](https://www.clemson.cloudlab.us/portal/show-nodetype.php?type=c8220&_gl=1*1goap78*_ga*MTIyNzE5ODc0My4xNzIxOTIwMTMw*_ga_6W2Y02FJX6*MTcyMzE4MzIwNi42LjEuMTcyMzE4MzIyMy4wLjAuMA..) 
  used in the Cloudlab environment to run training and test queries. These are the target instances, i.e. for these machines and database environments, LCMs are trained.
- **Training-Instances**: In order to accelerate the training LCMs, we use a set of cluster nodes that especially have GPU support (CUDA).
- **Localhost**: Used as central coordinator to distribute code and evaluate results.

Moreover, we integrated and recommend [Weights & Biases](https://wandb.ai/site) to monitor the training process and evaluate the results.

## Environment Variables
To make the scripts running, the user has to provide the following information by setting the corresponding environment variables:
- `LOCAL_ROOT_PATH`: The path to the root directory of the repository on the local host.
- `LOCAL_KNOWN_HOSTS_PATH`: The path to the known_hosts file on the local host.
- `CLOUDLAB_ROOT_PATH`: The path to the root directory of the repository on the cloudlab machines.
- `CLOUDLAB_SSH_USERNAME`: The username to log in to the cloudlab machines.
- `CLOUDLAB_SSH_KEY_PATH`: The path to the ssh key to log in to the cloudlab machines.
- `CLOUDLAB_SSH_PASSPHRASE`: The passphrase of the ssh key to log in to the cloudlab machines.
- `CLUSTER_ROOT_PATH`: The path to the root directory of the repository on the training machines.
- `CLUSTER_STORAGE_PATH`: The path to the storage directory on the training machines.
- `CLUSTER_SSH_USERNAME`: The username to login to the training machines.
- `CLUSTER_SSH_KEY_PATH`: The path to the ssh key to login to the training machines.
- `CLUSTER_SSH_PASSPHRASE`: The passphrase of the ssh key to login to the training machines.
- `WANDB_USER`: The username of the Weights & Biases account.
- `WANDB_PROJECT`: The project name of the Weights & Biases account.
- `OSF_USERNAME`: The username of the OSF account where the data is stored.
- `OSF_PASSWORD`: The password of the OSF account where the data is stored.
- `OSF_PROJECT`: The project name of the OSF account where the data is stored.
- `NODE00`: Node Information of the first training machine in the form of: `{'hostname': 'hostname.com', 'python': '3.9'}`.
- Add more nodes if necessary.

## Evaluation Steps
The repository contains a set of scripts to automate the evaluation process.
They are located at `src/scripts/exp_runner` and realize the following tasks:
- **Setup**: Install all necessary dependencies and download the required data on a set of given cloudlab machines.
- **Run Training Workload**: Execute the training workloads on the target machines.
- **Run Evaluation Workload**: Execute selected evaluation workloads on the target machines.
- **Train Models**: Train the LCMs on the training machines with the training data.
- **Predict Models**: Predict the cost of selected evaluation workloads with the trained LCMs.
- **Remove Data**: Remove selected workloads data from localhost and training machines.

### 1. Setup Target Machines
Make sure that the target machines (i.e. Cloudlab Instances) can be reached over SSH.
Also make sure to put an `.env` file in the root directory of the repository with the environment variables.
The following command installs all necessarey dependencies and downloads the required data from a OSF repository.
Moreover, it scales the data (if tables are too small), installs PostgreSQL (with hinting extension) and ingests 
the data to postgres. Moreover, training queries (SQL-Strings) are generated. 
Note that this script needs to be executed **twice**, as the hardware requires a reboot.
The node names (e.g. `clnode032.clemson.cloudlab.us`) are read out from a file.
This file and the SSH-configuration can be extracted from Cloudlabs `rspec` file with `parse_cloudlab_manifest.py`
To set-up the target machines, please run:

```
python3 exp_setup.py --task setup deliver start monitor
```

### 2. Run Training Workload
The following command executes the training workloads on the target machines.
```
python3 exp_run_training_workload.py --task setup deliver start monitor
```
Note the different data formats of the query plans that are generated:
- `raw`: This is the output of Postgres `EXPLAIN ANALYZE`command. Moreover, database and column statistics are stored.
- `parsed_plan`: This is a parsed version of the query plan that is fed into the LCMs.
- `parsed_plan_baseline`: These query plans are also parsed and additionally contain sample bitmaps which are required by some LCM.
- `json`: As identified later, Postgres also directly provides the query plan in JSON format.
This format contains more features than the standard plans and is required particularly by `QPPNet`.

### 3. Pick-Up Training Workload
The following command picks up the training workloads from the target machines and stores them on the local host.
```
python3 exp_run_training_workload.py --task pickup
```


### 4. Run Evaluation Workload
Similar as before, the following command executes the evaluation workloads on the target machines.
```
python3 exp_run_evaluation_workloads.py --task setup deliver start monitor
```
They can be fetched with:
```
python3 exp_run_training_workload.py --task pickup
```

### 5. Extract Feature Statistics
In order to train models, feature statistics are required to describe the data distribution and 
range of features. They are derived from the training workloads.
```
python3 gather_feature_statistics.py 
--database imdb 
--target ./data/runs/json/tpc_h_pk/feature_statistics.json 
--workload workload_100k_s1.json 
```
Note that QPPNet requires a different format of the feature statistics

--> ToDo: Describe exact assignment of feature statistics

### 5. Train Models
The following command trains the LCMs on the training machines with the training data.
Note that you require feature statistics and training data (parsed query plans) to train the models.
```
python3 exp_train_model.py --task setup deliver start monitor
```
In the script you can select the corresponding LCMs and databases that should be trained.
See the paper for more details on the training data selection for the LCMs.
The training script automatically evaluates the LCM against a test set as written in the paper and stores
its results under `/data/evaluation`
The models are stored (remotely) under `/data/models`.

### 6. Predict Workloads
To predict unseen workloads, please call:
```
python3 exp_predict_all.py --task setup deliver start monitor
```
Make sure to configure the desired evaluation workload and target database in the script.
The evaluation workloads are stored under `/data/evaluation`.

### 7. Fetch Predictions
The following command fetches the predictions from the target machines and stores them on the localhost under `/data/evaluation`.
```
python3 exp_predict_all.py --task pickup
```

### 8. Plot Evaluations
The evaluation strategy is described in the paper and the corresponding jupyter notebooks can be found at:
`/src/evaluation`. The notebooks are used to evaluate the results of the LCMs and compare them to the ground truth.