# JointNET: A Deep Model for Predicting Active Sacroiliitis from Sacroiliac Joint Radiography

## About

JointNET is a deep learning model developed to predict active inflammation from sacroiliac joint radiographs. The model was trained and validated on a dataset of 1,537 (augmented to 1,752) grade 0 Sacroiliac Joints (SIJs) from 768 patients. The gold-standard MRI exams showed active inflammation in 330 joints according to ASAS criteria. The model's performance was compared with two radiologists, and it showcased superior accuracy in detecting active inflammation based solely on radiographs.

[Link to the paper](https://arxiv.org/abs/2301.10769)

## Repository Files

- **RunLogGenerator.py**: [Description based on file content]
- **all_ensemble_table.py**: [Description based on file content]
- **count.py**: [Description based on file content]
- **create_folders.py**: [Description based on file content]
- **ensemble_gender_age_cv.py**: [Description based on file content]
- **optimal_epoch_finder.py**: [Description based on file content]
- **psnr_ssim_pgan.py**: [Description based on file content]
- **renamer.py**: [Description based on file content]
- **result_finder.py**: [Description based on file content]
- **result_finder2.py**: [Description based on file content]
- **roc_final.py**: [Description based on file content]
- **roc_final2.py**: [Description based on file content]

## Paper Summary

The primary objective of the research was to develop a deep learning model capable of predicting active inflammation from sacroiliac joint radiographs. The study involved a retrospective analysis of 1,537 grade 0 SIJs from 768 patients. Gold-standard MRI exams identified active inflammation in 330 joints. The convolutional neural network model, JointNET, was developed to detect MRI-based active inflammation labels using only radiographs. The model achieved a mean AUROC of 89.2% with a sensitivity of 69.0% and specificity of 90.4%. The study concluded that JointNET effectively predicts active inflammation from sacroiliac joint radiographs, outperforming human observers.

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

