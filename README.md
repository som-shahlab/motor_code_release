**Overview**

This is the code release for the paper "MOTOR: A Time-To-Event Foundation Model For Structured Medical Records".

Note that the code here is somewhat outdated and unmaintained. If you are interested in running MOTOR on your own data, please see the femr repository: https://github.com/som-shahlab/femr


This code release consists of five main components:

- **femr**: A fork of a lab-internal medical ML library with our data processing,  MOTOR, a next code baseline (see Appendix I) and count featurization.

- **baselines**: An implementation of the Cox PH, RSF, and DeepSurv baselines.

- **labels_and_features**: Our labeling and featurization code.

- **postprocessing**: Code for removing our labels from the pretraining task to help measure out of domain performance.

- **evaluation**: Code for performing evaluation of our three metrics, with confidence intervals.

**Steps to Reproduce Experiments**

1. Create a FEMR extract given your EHR/claims data, following the instructions in the femr package.

2. Run labels_and_features/generate_survival_labels.py to generate labels.

3. Run labels_and_features/subset_labels.py to perform the subsetting.

4. Generate features with both labels_and_features/generate_feature_matrices.py and labels_and_features/generate_final_features.py.

5. Run baselines with baselines/run_r_baseline.py, baselines/train_deepsurv.py, baselines/train_deephit.py, and baselines/train_dsm.py.

6. Pretrain MOTOR and the next_code model, following instructions in femr/tutorials/reference/1_train_motor_model.py. Note the step to apply postprocessing/remove_from_dictionary.py.

7. Apply MOTOR and next_code model, following instructions in femr/tutorials/reference/2_apply_motor_model.py.

8. Run evaluations using evaluation/eval_models.py.


**Python / R environment details**

For every step except 4, use a conda environment set up according to the instructions in femr/pyproject.toml, with additional packages as specified in femr_requirements.txt

For step 4, deepsurv_requirements.txt contains the Python requirements for deepsurv and the packrat folder specifies the R dependencies.
