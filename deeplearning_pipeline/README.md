# Pythia project

## Setup environment
We create a python 3.6 virtual environment and we activate it:
```
virtualenv -p python3.6 venv
source venv/bin/activate
```
Now we are ready to install the packages we need:
```
pip install -r requirements.txt
```

For more flexibility we can create 2 folders inside the repo containing the data and the model output.
It's ok because they are gitgnored:
```
mkdir data
mkdir model
```

## Training process
For the whole pipeline we can modify the following parameters inside `src/params.json` to make 
a lot of experiments.

### Parameters explanation
```
{
 "train_lstm": {
  "train_flag": false,
  "batch_size": 512,
  "epochs": 20,
  "learning_rate": 0.001,
  "activation": "relu",
  "optimizer": "adam",
  "lstm_units": [200, 100],
  "dropouts": [0.5, 0.5],
  "hyperparameter_search": false,
  "num_hp_experiments": 2
 },
 "train_mlp": {
  "batch_size": 1024,
  "epochs": 70,
  "learning_rate": 0.001,
  "activation": "relu",
  "optimizer": "adam",
  "dense_units": [256, 128, 64, 32],
  "dropouts": [0.2, 0.2, 0.3, 0.3],
  "hyperparameter_search": false,
  "num_hp_experiments": 2
 },
 "data_lstm": {
  "data_dir": "data/raoula.csv",
  "test_start_date": "2019-03-29",
  "window_size": 100
 },
 "data_mlp": {
  "data_dir": "data/raoula.csv",
  "test_split": 0.3
 },
 "callbacks": {
  "adaptive_learning_rate": false,
  "adaptive_lr_patience_epochs": 1,
  "adaptive_lr_decay": 0.2,
  "min_adaptive_lr": 0.000001,
  "exponential_lr": false,
  "num_epochs_per_decay": 3,
  "lr_decay_factor": 0.4,
  "early_stopping": false,
  "early_stopping_min_change": 0,
  "early_stopping_patience_epochs": 10,
  "save_model_per_epoch": false,
  "epochs_per_save": 1,
  "model_dir": "model"
 },
 "more": {
   "save_final_model": true,
   "final_model_dir": "model",
   "results_to_csv": true,
   "csv_path": "model",
   "gpus": 0,
   "fine-tuning-file": ""
}
}

```
Inside the `train` field we can modify the parameters for different experiments. How to run hyper-parameter search:
* We set the `hyperparameter_seach` flag to true.
* We can change the following parameters into lists to take find the best experiment: `batch_size`, `epochs`,
`learning_rate`, `activation`(only relu, sigmoid and tanh) and `optimizer`(adam, sgd, adagrad, adadelta, rmsprop).
* We should also set the number of experiments to run. Be careful because the Keras backend does not support
cross-experiments between the lists we set. But if we set a bigger number for `num_hp_experiments` than the 
expected number of experiments we are safe and also it's not bad running again the same experiment. For example
if we set batch size into `[2, 4]` and learning rate into `[0.1, 0.01]` we expect 4 experiments so it's good to set
`num_hp_experiments=10`!

Inside the `callbacks` field we can set some useful Keras callbacks during the training process.
Inside `more` field we can set some useful flags for storing model file and results. Pay attention to the last flag
`fine-tuning-file` to restart the training process from a checkpoint file. Be careful because callbacks
are activated from the beginning. When we set the flags correctly we can run experimets inside `src` like that:

```
python train_eval.py
```

## Tensorflow serving pipeline

```
docker pull tensorflow/serving
docker run -p 8501:8501 --name tfserving_pythia --mount type=bind,source=/home/christos/Desktop/Pythia/model/serving,target=/models/pythia -e MODEL_NAME=pythia -t tensorflow/serving &
```
