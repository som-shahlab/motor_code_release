"""
    This script walks through the various steps to apply MOTOR.

    In order to use this script, the assumption is that you already have a set of labels and an extract
"""

import os
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('EXTRACT_LOCATION')
parser.add_argument('STORAGE')
parser.add_argument('LABELS')

args = parser.parse_args()

EXTRACT_LOCATION = args.EXTRACT_LOCATION
STORAGE = args.STORAGE
LABELS = args.LABELS

DICTIONARY_PATH = os.path.join(STORAGE, "dictionary")

SURVIVAL_DICTIONARY_PATH = os.path.join(STORAGE, "survival_dictionary")
MODEL_PATH = os.path.join(STORAGE, "motor_model")


"""
First, we need to prepare batches for the labels.
"""

MOTOR_BATCHES = os.path.join(STORAGE, "motor_batches")
TASK_BATCHES = os.path.join(STORAGE, "task_batches")

# Replace motor with next_code for next_code objective
assert 0 == os.system(
    f"femr_create_batches {TASK_BATCHES} --data_path {EXTRACT_LOCATION} --dictionary {DICTIONARY_PATH} --task labeled_patients --labeled_patients_path {LABELS}"
)

"""
Given the batches, we can now train a linear probe
"""

PROBE_PATH = os.path.join(STORAGE, "probe.pkl")
PROBE_PREDICTIONS_PATH = os.path.join(STORAGE, "probe_predictions.pkl")

assert 0 == os.system(
    f"python femr/native/finetune_head_model.py {PROBE_PREDICTIONS_PATH} --probe {PROBE_PATH} --data_path {EXTRACT_LOCATION} \
        --batch_info_path {TASK_BATCHES}/batch_info.msgpack --model_dir {MODEL_PATH} \
        --labeled_patients_path {LABELS} --survival_batches {MOTOR_BATCHES}"
)

"""
We can now fully finetune the model
"""

TASK_MODEL_PATH = os.path.join(STORAGE, "task_motor_model")

assert 0 == os.system(
    # Set extra hyperparameters at this line. Also increase max_iter before training for real
    f"femr_train_model {TASK_MODEL_PATH} --data_path {EXTRACT_LOCATION} --batches_path {TASK_BATCHES} --learning_rate 1e-4 --rotary_type per_head --num_batch_threads 3 --max_iter 1000 --start_from_checkpoint {MODEL_PATH} --linear_probe_head {PROBE_PATH} --survival_batches {MOTOR_BATCHES}"
)

"""
Evaluate the finetuned model
"""

FINETUNE_PREDICTIONS_PATH = os.path.join(STORAGE, "finetune_predictions.pkl")

assert 0 == os.system(
    f"python -u femr/native/compute_predictions.py {FINETUNE_PREDICTIONS_PATH} --data_path {EXTRACT_LOCATION} \
        --batch_info_path {TASK_BATCHES}/batch_info.msgpack --model_dir {TASK_MODEL_PATH} \
        --labeled_patients_path {LABELS}"
)
