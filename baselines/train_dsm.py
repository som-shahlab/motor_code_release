import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
import dsm

import torch
import torchtuples as tt

from pycox.models import DeepHitSingle
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
parser.add_argument("--k", type=int)
parser.add_argument("--distribution", type=str)

args = parser.parse_args()

os.mkdir(args.directory)

with open(os.path.join(args.directory, "config.json"), "w") as f:
    json.dump(
        {
            "path": args.data_path,
            "learning_rate": args.learning_rate,
            "middle_size": args.middle_size,
            "k": args.k,
            "distribution": args.distribution,
            "models": "dsm",
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

train_indices = train_indices[deltas[train_indices] > 0]
val_indices = val_indices[deltas[val_indices] > 0]

x_train = features[train_indices, :]
y_train = deltas[train_indices], is_event[train_indices]


x_val = features[val_indices, :]
y_val = deltas[val_indices], is_event[val_indices]

model = dsm.DeepSurvivalMachines(k=args.k, distribution=args.distribution, layers=[args.middle_size, 32], cuda=1)

device = torch.device('cuda')

val = x_val, y_val

model.fit(x_train, y_train[0], y_train[1], iters=500, learning_rate=args.learning_rate, batch_size=256, optimizer='Adam', val_data=(x_val, y_val[0], y_val[1]))

times_to_predict = list(np.quantile(deltas[is_event], np.linspace(0, 1, num=128 + 1))[:-1])
assert len(times_to_predict) == 128

with open(os.path.join(args.directory, "model.pkl"), "wb") as f:
    pickle.dump(model, f)

predictions = model.predict_survival(features, times_to_predict)
#print(predictions[:10, :10])
delta = -np.log(predictions[:, 1:] / predictions[:, :-1])
#print(delta[:10, :10])
times_to_predict = np.array(times_to_predict)
time_per_bin = times_to_predict[1:] - times_to_predict[:-1]

hazard = np.log2(delta / time_per_bin).astype(np.float16)
#print(hazard[:10, :10])

with open(os.path.join(args.directory, "hazard.npy"), "wb") as f:
    np.save(f, hazard)

with open(os.path.join(args.directory, "bins.npy"), "wb") as f:
    np.save(f, times_to_predict)


with open(os.path.join(args.directory, "done"), "w") as f:
    f.write("\n")

if False:
    pro_features = np.load(os.path.join(feature_dir, "pro_features.npy"))
    pro_predictions = model.predict_survival(pro_features, list(times_to_predict))
    pro_delta = -np.log(pro_predictions[:, 1:] / pro_predictions[:, :-1])

    pro_hazard = np.log2(pro_delta / time_per_bin).astype(np.float16)
    with open(os.path.join(args.directory, "pro_hazard.npy"), "wb") as f:
        np.save(f, pro_hazard)
