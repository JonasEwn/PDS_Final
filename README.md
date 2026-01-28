# Predicting Domain and Seniority from LinkedIn Data

This Repository conatins the Code and Data for our Final Project, focuse on predicting the Domain and Seniority of the provided LinkedIn Data using different Machine Learning approaches. The Repository is structured according to the suggested apporoaches in the task describtion.

HF-profile: https://huggingface.co/Luu200


# Predicting Domain and Seniority from LinkedIn Data

This repository contains the code and data for the **Practical Data Science Final Project**.  
The goal in this assignmnet is to predict the **professional domain (department)** and **seniority level** of individuals based on LinkedIn CV data using machine learning approaches.

The repository is organised according to the structure of the task description, with our different approaches categorised accordingly.

---

## Repository Structure

'''
PDS_Final/
├── data/
│   ├── raw/                         # Original LinkedIn CV data (JSON)
│   ├── processed/                   # Cleaned and preprocessed CSV/JSON files
│   └── snapaddy_labels.csv          # Hand-labeled evaluation dataset
│
├── task_1_rule_based/
│   ├── rule_based_domain.ipynb
│   └── rule_based_seniority.ipynb
│
├── task_2_embeddings/
│   ├── embedding_zero_shot_domain.ipynb
│   └── embedding_zero_shot_seniority.ipynb
│
├── task_3_finetuning/
│   ├── finetune_domain.ipynb
│   └── finetune_seniority.ipynb
│
├── task_4_programmatic_labeling/
│   ├── pseudo_label_generation.ipynb
│   └── supervised_training.ipynb
│
├── task_6_feature_engineering/
│   ├── feature_extraction.ipynb
│   └── classical_ml_models.ipynb
│
├── ensemble/
│   ├── department.ipynb
│   ├── seniority.ipynb
│   └── plots/                       # Evaluation plots and confusion matrices
│
├── reports/
│   ├── final_report.pdf
│   └── figures/
│
├── src/
│   ├── preprocessing.py
│   ├── modeling.py
│   └── evaluation.py
│
├── requirements.txt
└── README.md
'''
