# JointNET: A Deep Model for Predicting Active Sacroiliitis from Sacroiliac Joint Radiography

## About

JointNET is a deep learning model developed to predict active inflammation from sacroiliac joint radiographs. The model was trained and validated on a dataset of 1,537 (augmented to 1,752) grade 0 Sacroiliac Joints (SIJs) from 768 patients. The gold-standard MRI exams showed active inflammation in 330 joints according to ASAS criteria. The model's performance was compared with two radiologists, and it showcased superior accuracy in detecting active inflammation based solely on radiographs.

[Link to the paper](https://arxiv.org/abs/2301.10769)

## Algorithm Description

The JointNET model is a convolutional neural network designed to detect MRI-based active inflammation labels using only radiographs. The model was trained on a dataset of 1,537 grade 0 SIJs, which was augmented to 1,752 samples. The gold-standard MRI exams identified active inflammation in 330 joints based on ASAS criteria. The model's performance metrics are as follows:
- Mean AUROC: 89.2% (95% CI: 86.8%, 91.7%)
- Sensitivity: 69.0% (95% CI: 65.3%, 72.7%)
- Specificity: 90.4% (95% CI: 87.8%, 92.9%)
- Mean Accuracy: 90.2% (95% CI: 87.6%, 92.8%)

The model's positive predictive value was 74.6% (95% CI: 72.5%, 76.7%) and the negative predictive value was 87.9% (95% CI: 85.4%, 90.5%) when the prevalence was considered 1%.

## Repository Files

- **RunLogGenerator.py**: Generates logs for model runs.
- **all_ensemble_table.py**: Compiles results from ensemble methods.
- **count.py**: Utility for counting dataset samples.
- **create_folders.py**: Script to create necessary directories.
- **ensemble_gender_age_cv.py**: Ensemble method considering gender and age.
- **optimal_epoch_finder.py**: Finds the optimal epoch for training.
- **psnr_ssim_pgan.py**: Computes PSNR and SSIM metrics for PGAN.
- **renamer.py**: Utility for renaming files.
- **result_finder.py**: Retrieves model results.
- **result_finder2.py**: Extended utility for retrieving model results.
- **roc_final.py**: Computes ROC metrics for the final model.
- **roc_final2.py**: Extended utility for computing ROC metrics.



## Paper Summary

The primary objective of the research was to develop a deep-learning model capable of predicting active inflammation from sacroiliac joint radiographs. The study involved a retrospective analysis of 1,537 grade 0 SIJs from 768 patients. Gold-standard MRI exams identified active inflammation in 330 joints. The convolutional neural network model, JointNET, was developed to detect MRI-based active inflammation labels using only radiographs. The model achieved a mean AUROC of 89.2% with a sensitivity of 69.0% and specificity of 90.4%. The study concluded that JointNET effectively predicts active inflammation from sacroiliac joint radiographs, outperforming human observers.

## Authors

- Sevcan Turk
- Ahmet Demirkaya
- M Yigit Turali
- Cenk Hepdurgun
- Salman UH Dar
- Ahmet K Karabulut
- Aynur Azizova
- Mehmet Orman
- Ipek Tamsel
- Ustun Aydingoz
- Mehmet Argin
- Tolga Cukur

