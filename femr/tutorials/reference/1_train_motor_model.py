"""
    This script walks through the various steps to train and use MOTOR.

    In order to use this script, the assumption is that you already have a set of labels and an extract
"""

import os
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('EXTRACT_LOCATION')
parser.add_argument('STORAGE')

args = parser.parse_args()

EXTRACT_LOCATION = args.EXTRACT_LOCATION
STORAGE = args.STORAGE

"""
The first step of training MOTOR is creating dictionaries, that helps map codes to integers that can be used within a neural network.
"""

DICTIONARY_PATH = os.path.join(STORAGE, "dictionary")

assert 0 == os.system(f"femr_create_dictionary {DICTIONARY_PATH} --data_path {EXTRACT_LOCATION}")
    
RAW_SURVIVAL_DICTIONARY_PATH = os.path.join(STORAGE, "survival_dictionary")
SURVIVAL_DICTIONARY_PATH = os.path.join(STORAGE, "survival_dictionary")

assert 0 == os.system(f"femr_create_survival_dictionary {RAW_SURVIVAL_DICTIONARY_PATH} --data_path {EXTRACT_LOCATION} --num_buckets 8 --size 8192")

# For MOTOR paper only: Remove certain tasks to make things more out of domain
assert 0 == os.system(f"python postprocessing/remove_from_dictionary.py {EXTRACT_LOCATION} {RAW_SURVIVAL_DICTIONARY_PATH} {SURVIVAL_DICTIONARY_PATH}")


"""
The second step of training MOTOR is to prepare the batches that will actually get fed into the neural network.
"""

MOTOR_BATCHES = os.path.join(STORAGE, "motor_batches")

# Replace motor with next_code for next_code objective
assert 0 == os.system(
    f"femr_create_batches {MOTOR_BATCHES} --data_path {EXTRACT_LOCATION} --dictionary {DICTIONARY_PATH} --survival_dictionary {SURVIVAL_DICTIONARY_PATH} --task motor"
)

"""
Given the batches, it is now possible to train MOTOR. By default it will train for 10 epochs, with early stopping.
"""

MODEL_PATH = os.path.join(STORAGE, "motor_model")

assert 0 == os.system(
    # Set extra hyperparameters at this line. Also increase max_iter before training for real
    f"femr_train_model {MODEL_PATH} --data_path {EXTRACT_LOCATION} --batches_path {MOTOR_BATCHES} --learning_rate 1e-4 --rotary_type per_head --num_batch_threads 3 --max_iter 1000 --motor_dim 512"
)
