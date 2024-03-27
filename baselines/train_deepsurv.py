import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper

import torch
import torchtuples as tt

from pycox.models import CoxPH
from pycox.evaluation import EvalSurv
import argparse
import os
import json
import pickle

parser = argparse.ArgumentParser(prog="Train")
parser.add_argument("directory", type=str)
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--learning_rate", type=float, required=True)
parser.add_argument("--middle_size", type=int, default=0)
parser.add_argument("--dropout", type=float, default=0)

args = parser.parse_args()

os.mkdir(args.directory)

with open(os.path.join(args.directory, "config.json"), "w") as f:
    json.dump(
        {
            "path": args.data_path,
            "learning_rate": args.learning_rate,
            "middle_size": args.middle_size,
            "dropout": args.dropout,
            "models": "deep_surv",
        },
        f,
    )

np.random.seed(1234)
_ = torch.manual_seed(123)

feature_dir = args.data_path

features = np.load(os.path.join(feature_dir, "features.npy"))
times = np.load(os.path.join(feature_dir, "times.npy"))
train_indices = np.load(os.path.join(feature_dir, "train_indices.npy"))
val_indices = np.load(os.path.join(feature_dir, "val_indices.npy"))
test_indices = np.load(os.path.join(feature_dir, "test_indices.npy"))
is_event = np.load(os.path.join(feature_dir, "is_event.npy"))
deltas = np.load(os.path.join(feature_dir, "deltas.npy"))

x_train = features[train_indices, :]
y_train = deltas[train_indices], is_event[train_indices]

x_val = features[val_indices, :]
y_val = deltas[val_indices], is_event[val_indices]
val = x_val, y_val

x_test = features[test_indices, :]
y_test = deltas[test_indices], is_event[test_indices]

in_features = x_train.shape[1]
num_nodes = [args.middle_size, 32]
out_features = 1
batch_norm = True
dropout = args.dropout
output_bias = False
batch_size = 256

net = tt.practical.MLPVanilla(
    in_features,
    num_nodes,
    out_features,
    batch_norm,
    dropout,
    output_bias=output_bias,
)

model = CoxPH(net, tt.optim.Adam)

model.optimizer.set_lr(args.learning_rate)

epochs = 500
callbacks = [tt.callbacks.EarlyStopping()]
verbose = True

log = model.fit(
    x_train,
    y_train,
    batch_size,
    epochs,
    callbacks,
    verbose,
    val_data=val,
    val_batch_size=batch_size,
)

predictions = model.predict(features)

with open(os.path.join(args.directory, "model.pkl"), "wb") as f:
    pickle.dump(net.state_dict(), f)

with open(os.path.join(args.directory, "predictions.npy"), "wb") as f:
    np.save(f, predictions)


with open(os.path.join(args.directory, "done"), "w") as f:
    f.write("\n")
